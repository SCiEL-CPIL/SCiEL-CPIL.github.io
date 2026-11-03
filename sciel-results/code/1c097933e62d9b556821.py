"""Combined QAP solver: continuous relaxation + Hungarian + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors.
    2. Continuous relaxation via projected gradient descent on assignment matrix.
    3. Hungarian-based recovery from relaxed solution.
    4. Fast local search with precomputed swap deltas.
    5. 3-swap moves to escape 2-swap local optima.
    6. Multi-restart with perturbation if time permits.
    """
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    def elapsed():
        return time.time() - t0

    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    # --- Spectral initialization ---
    def spectral_init():
        """Initialize using eigenvectors of flow and distance matrices."""
        # For QAP, we want to align high-flow facilities with low-distance locations
        # Use leading eigenvectors of symmetric parts
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get leading eigenvectors
        try:
            evals_a, evecs_a = np.linalg.eigh(a_sym)
            evals_b, evecs_b = np.linalg.eigh(b_sym)
            
            # For minimization, we want to match high eigenvalues of a with low eigenvalues of b
            # Sort a eigenvectors by descending eigenvalue, b by ascending
            idx_a = np.argsort(-evals_a)
            idx_b = np.argsort(evals_b)
            
            # Take top k eigenvectors
            k = min(10, n)
            V_a = evecs_a[:, idx_a[:k]]
            V_b = evecs_b[:, idx_b[:k]]
            
            # Compute alignment score
            # We want permutation that maximizes trace(V_a^T P V_b) or similar
            # For now, use rank matching on projected values
            proj_a = V_a.T @ np.arange(n)  # This is a heuristic
            proj_b = V_b.T @ np.arange(n)
            
            # Better: use the eigenvector components directly
            # Facility i has "importance" from a's eigenvectors
            # Location j has "centrality" from b's eigenvectors
            a_scores = np.abs(evecs_a[:, idx_a[0]])  # Leading eigenvector magnitude
            b_scores = np.abs(evecs_b[:, idx_b[0]])
            
            heavy_idx = np.argsort(-a_scores, kind="stable")
            central_idx = np.argsort(b_scores, kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[heavy_idx] = central_idx
            return p
            
        except Exception:
            # Fallback to rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p

    # --- Continuous relaxation via projected gradient descent ---
    def continuous_relaxation(p_init, max_time):
        """Solve continuous relaxation of QAP using projected gradient descent.
        
        Minimize F(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        subject to X being a doubly stochastic matrix.
        
        Gradient: grad[i,k] = 2 * sum_j a[i,j] * (B @ X)[:, j] ... 
        More precisely: grad = 2 * (A @ (B @ X)) + 2 * (A.T @ (B.T @ X.T)).T
        
        Actually, for QAP with cost sum a[i,j] b[p[i], p[j]]:
        The continuous relaxation objective is:
        F(X) = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] X[i,k] X[j,l]
             = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        
        Gradient w.r.t. X[i,k]:
        dF/dX[i,k] = 2 * sum_{j,l} a[i,j] b[k,l] X[j,l] + 2 * sum_{j,l} a[j,i] b[l,k] X[j,l]
                   = 2 * (A @ (B @ X.T)).T[i,k] + 2 * (A.T @ (B.T @ X.T)).T[i,k]
        
        Let's define:
        G1 = A @ (B @ X.T)  -> shape (n,n), G1[i,l] = sum_j A[i,j] sum_k B[k,l] X.T[j,k] = sum_j A[i,j] sum_k B[k,l] X[k,j]
        Wait, X.T[j,k] = X[k,j]. So G1[i,l] = sum_j A[i,j] (B @ X)[l,j]
        Actually B @ X has shape (n,n): (B@X)[k,j] = sum_l B[k,l] X[l,j]
        So G1 = A @ (B @ X).T? No.
        
        Let's be careful.
        F(X) = trace(X.T @ A @ X @ B.T) is one form, but let's stick to the sum.
        F(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        
        dF/dX[i,k] = 2 * sum_{j,l} a[i,j] b[k,l] X[j,l]
        
        Let M[i,j] = a[i,j]. Let N[k,l] = b[k,l].
        dF/dX[i,k] = 2 * sum_j sum_l a[i,j] b[k,l] X[j,l]
        
        Define C = A @ X @ B.T. Then C[i,l] = sum_j sum_k A[i,j] X[j,k] B.T[k,l] = sum_j sum_k A[i,j] X[j,k] B[l,k]
        That's not quite right either.
        
        Let's use matrix calculus properly.
        F(X) = vec(X).T @ (B ⊗ A) @ vec(X) approximately, but with constraints.
        
        Simpler: 
        grad[i,k] = 2 * sum_j a[i,j] * (B @ X)[:, j].T[k] ... no.
        
        Let S = B @ X. S[k,j] = sum_l B[k,l] X[l,j].
        Then sum_l b[k,l] X[j,l] = (X @ B.T)[j,k] = (B @ X.T).T[j,k].
        
        Let's compute:
        T1 = A @ (X @ B.T)  # T1[i,k] = sum_j A[i,j] (X@B.T)[j,k] = sum_j A[i,j] sum_l X[j,l] B.T[l,k] = sum_j A[i,j] sum_l X[j,l] B[k,l]
        T2 = A.T @ (X @ B.T)  # T2[j,k] = sum_i A.T[i,j] (X@B.T)[i,k] = sum_i A[j,i] sum_l X[i,l] B[k,l]
        
        So dF/dX[i,k] = 2 * (T1[i,k] + T2[i,k])? 
        Actually, since the sum is over all i,j,k,l, and X appears twice, the gradient is:
        dF/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
                   = sum_j (sum_l a[i,j] b[k,l] X[j,l]) + sum_j (sum_l a[j,i] b[l,k] X[j,l])
                   
        First term: for fixed i,k, sum_j sum_l a[i,j] b[k,l] X[j,l] = (A @ (X @ B.T))[i,k]
        Second term: for fixed i,k, sum_j sum_l a[j,i] b[l,k] X[j,l] = (A.T @ (X @ B.T))[i,k]
        
        So grad = 2 * (A @ (X @ B.T) + A.T @ (X @ B.T))
        
        But wait, X @ B.T has shape (n,n). (X @ B.T)[j,l] = sum_m X[j,m] B.T[m,l] = sum_m X[j,m] B[l,m].
        So (A @ (X @ B.T))[i,k] = sum_j A[i,j] sum_m X[j,m] B[k,m].
        This matches sum_j sum_m a[i,j] b[k,m] X[j,m]. Yes.
        
        So grad = 2 * (A @ (X @ B.T) + A.T @ (X @ B.T))
        
        We project onto the Birkhoff polytope (doubly stochastic matrices) using the Hungarian algorithm
        or iterative scaling. For simplicity and speed, we can use the fact that the projection
        onto doubly stochastic matrices is given by the Hungarian algorithm on the negative gradient,
        or we can use Sinkhorn-Knopp scaling for a soft projection.
        
        Given time constraints, we'll do a few steps of gradient descent with Sinkhorn scaling
        to maintain approximate doubly-stochasticity, then recover via Hungarian.
        """
        if time.time() > max_time:
            return p_init
        
        # Initialize X as the assignment matrix of p_init
        X = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            X[i, p_init[i]] = 1.0
        
        # Precompute B.T
        BT = b.T
        
        # Gradient descent steps
        num_steps = 200
        lr = 0.01
        step_time_budget = (max_time - time.time()) / num_steps
        
        for step in range(num_steps):
            if time.time() > max_time:
                break
            
            # Compute gradient
            XB = X @ BT  # (n,n)
            grad = 2.0 * (a @ XB + a.T @ XB)
            
            # Gradient step
            X -= lr * grad
            
            # Project onto doubly stochastic polytope using Sinkhorn-Knopp
            # Iterate row and column normalization
            for _ in range(10):
                # Row normalization
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                X = X / row_sums
                
                # Column normalization
                col_sums = X.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                X = X / col_sums
            
            # Clip to [0,1]
            X = np.clip(X, 0, 1)
            
            # Renormalize to be exactly doubly stochastic (approximate)
            # Final scaling
            for _ in range(5):
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                X = X / row_sums
                col_sums = X.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                X = X / col_sums
        
        # Recover permutation via Hungarian algorithm
        # Minimize sum X[i,j] * (-something) or just use X as cost matrix
        # Actually, we want to find the permutation closest to X
        # Use Hungarian on -X to maximize sum X[i,p[i]]
        row_ind, col_ind = linear_sum_assignment(-X)
        p_relax = np.empty(n, dtype=np.int64)
        p_relax[row_ind] = col_ind
        
        return p_relax

    # --- Initial solution via spectral initialization ---
    p = spectral_init()
    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # --- Fast swap_delta using precomputed b_p = b[p[...]] ---
    def _update_b_p(best_p):
        """Precompute b_p[k,l] = b[p[k],l] and b_pT[k,l] = b[l,p[k]]."""
        return b[best_p, :], b[:, best_p]

    b_p, b_pT = _update_b_p(best_p)

    def swap_delta_fast(i, j):
        """Compute delta for swapping positions i and j in best_p. O(n) time."""
        u = best_p[i]
        v = best_p[j]
        if u == v:
            return 0
        
        delta = 0.0
        
        # Diagonal/cross terms
        delta += a[i,i]*(b[v,v] - b[u,u])
        delta += a[i,j]*(b[v,u] - b[u,v])
        delta += a[j,i]*(b[u,v] - b[v,u])
        delta += a[j,j]*(b[u,u] - b[v,v])
        
        # Row i, k not in {i,j}
        row_i_diff = b_pT[:, v] - b_pT[:, u]
        delta += np.dot(a[i], row_i_diff) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,u]-b[u,u])
        
        # Row j, k not in {i,j}
        row_j_diff = b_pT[:, u] - b_pT[:, v]
        delta += np.dot(a[j], row_j_diff) - a[j,i]*(b[u,u]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        # Col i, k not in {i,j}
        col_i_diff = b_p[:, v] - b_p[:, u]
        delta += np.dot(a[:, i], col_i_diff) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[u,v]-b[v,u])
        
        # Col j, k not in {i,j}
        col_j_diff = b_p[:, u] - b_p[:, v]
        delta += np.dot(a[:, j], col_j_diff) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        return delta

    # --- Local search with fast swap_delta ---
    def local_search_2swap(p, cost, b_p, b_pT, max_time):
        """Hill-climbing 2-swap local search with precomputed b_p, b_pT."""
        nonlocal best_p, best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            num_trials = 30000
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j)
                if d < -0.5:
                    u, v = p[i], p[j]
                    p[i], p[j] = v, u
                    cost += d
                    b_p[i, :], b_p[j, :] = b_p[j, :].copy(), b_p[i, :].copy()
                    b_pT[:, i], b_pT[:, j] = b_pT[:, j].copy(), b_pT[:, i].copy()
                    best_p = p
                    best_cost = cost
                    improved = True
        return p, cost

    # --- Hungarian refinement ---
    def hungarian_refine(p, cost, max_time):
        """Use Hungarian on linearized cost to improve solution."""
        nonlocal best_p, best_cost
        if time.time() > max_time:
            return p, cost
        
        b_p, b_pT = _update_b_p(p)
        
        # Build linear cost matrix
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < cost:
            p = p_hung
            cost = cost_hung
            best_p = p.copy()
            best_cost = cost
        return p, cost

    # --- 3-swap moves ---
    def swap3_delta(i, j, k):
        """Compute delta for 3-cycle: p[i]->p[j]->p[k]->p[i]."""
        u, v, w = best_p[i], best_p[j], best_p[k]
        if u == v or v == w or u == w:
            return 0
        
        delta = 0.0
        p_new = best_p.copy()
        p_new[i], p_new[j], p_new[k] = v, w, u
        
        for x in [i, j, k]:
            for y in [i, j, k]:
                delta += a[x,y]*(b[p_new[x],p_new[y]] - b[best_p[x],best_p[y]])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = mask[j] = mask[k] = False
        others = np.where(mask)[0]
        
        for x in [i, j, k]:
            diff = b[p_new[x], others] - b[best_p[x], others]
            delta += np.dot(diff, a[x, others])
        
        for y in [i, j, k]:
            diff = b[best_p[others], p_new[y]] - b[best_p[others], best_p[y]]
            delta += np.dot(diff, a[others, y])
        
        return delta

    def local_search_3swap(p, cost, b_p, b_pT, max_time):
        """Try 3-cycle moves to escape 2-swap local optima."""
        nonlocal best_p, best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            num_trials = 5000
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j, k = np.random.randint(0, n, 3)
                if len(set([i, j, k])) < 3:
                    continue
                d = swap3_delta(i, j, k)
                if d < -0.5:
                    u, v, w = p[i], p[j], p[k]
                    p[i], p[j], p[k] = v, w, u
                    cost += d
                    b_p[i, :], b_p[j, :], b_p[k, :] = b_p[j, :].copy(), b_p[k, :].copy(), b_p[i, :].copy()
                    b_pT[:, i], b_pT[:, j], b_pT[:, k] = b_pT[:, j].copy(), b_pT[:, k].copy(), b_pT[:, i].copy()
                    best_p = p
                    best_cost = cost
                    improved = True
        return p, cost

    # --- Main loop ---
    # Phase 1: Continuous relaxation to get a better starting point
    if time.time() < deadline - 60.0:
        p_relax = continuous_relaxation(p, deadline - 60.0)
        cost_relax = compute_cost(p_relax)
        if cost_relax < best_cost:
            best_p = p_relax.copy()
            best_cost = cost_relax
            p = p_relax
    
    # Phase 2: Initial local search
    if time.time() < deadline - 40.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 40.0)
    
    # Phase 3: Hungarian refinement
    if time.time() < deadline - 20.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost, deadline - 20.0)
    
    # Phase 4: More local search with 3-swaps
    if time.time() < deadline - 10.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_3swap(best_p, best_cost, b_p, b_pT, deadline - 10.0)
    
    # Phase 5: Final 2-swap polish
    if time.time() < deadline - 2.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 2.0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
