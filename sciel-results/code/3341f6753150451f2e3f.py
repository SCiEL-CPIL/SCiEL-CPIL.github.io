"""Frank-Wolfe / Birkhoff relaxation for QAP tai256c, with Graph Partitioning initialization and 2-opt polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact integer QAP cost for permutation p."""
    n = len(p)
    # b_p = b[p[:, None], p[None, :]] is the permuted distance matrix
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _spectral_bisection(A, n, t_start, time_budget):
    """
    Recursive spectral bisection to create a structured permutation.
    Minimizes cut in graph defined by A.
    Returns a permutation list where similar facilities are close in index.
    """
    n = len(A)
    
    def bisect(indices):
        if len(indices) <= 1:
            return indices
        
        # Check time
        if time.time() - t_start > time_budget:
            return sorted(indices)
        
        # Build subgraph
        A_sub = A[np.ix_(indices, indices)]
        
        # Compute degree
        deg = A_sub.sum(axis=1)
        
        # Laplacian L = D - A
        L = np.diag(deg) - A_sub
        
        # Solve for Fiedler vector (eigenvector of smallest non-zero eigenvalue)
        # Use eigh for symmetric matrix
        # For speed, we can use sparse or just dense if n is small enough
        # In recursive bisection, sizes drop quickly. 256 -> 128 -> 64 ...
        # 256x256 eigh is fast.
        
        try:
            # We want the eigenvector corresponding to the second smallest eigenvalue.
            # scipy.sparse.linalg.eigsh is better for large, but dense is fine for <= 256
            # We only need the smallest few.
            # Using numpy.linalg.eigh gives all.
            # Let's use scipy.sparse.linalg.eigsh for better performance if available and matrix is sparse-ish?
            # A is likely dense.
            # Let's just use numpy.linalg.eigh for simplicity and reliability on dense matrices up to 256.
            eigvals, eigvecs = np.linalg.eigh(L)
            
            # Fiedler vector is eigenvector of eigvals[1]
            fiedler = eigvecs[:, 1]
        except Exception:
            # Fallback to random or simple sort
            return sorted(indices, key=lambda x: np.random.rand())
        
        # Partition based on sign of Fiedler vector
        # Mean-centering is sometimes used, but sign is standard
        # We want balanced partitions
        # Sort by fiedler values
        order = np.argsort(fiedler)
        
        mid = len(indices) // 2
        left_indices = [indices[i] for i in order[:mid]]
        right_indices = [indices[i] for i in order[mid:]]
        
        # Recurse
        left_res = bisect(left_indices)
        right_res = bisect(right_indices)
        
        return left_res + right_res

    all_indices = list(range(n))
    return bisect(all_indices)

def _graph_partitioning_init(a, b, time_budget=10.0):
    """
    Generate a permutation using recursive spectral bisection on the flow matrix A.
    Then map these ordered facilities to locations.
    To map to locations, we can use a similar logic or just map the i-th facility in the 
    bisection order to the i-th location in a distance-ordered sequence?
    
    A better approach for QAP:
    The bisection gives an ordering of facilities that respects the flow structure.
    We should assign these to locations that also respect the distance structure.
    
    Simple strategy:
    1. Get facility order F from spectral bisection on A.
    2. Get location order L from spectral bisection on B.
    3. Assign F[0] -> L[0], F[1] -> L[1], etc.
    This aligns the "hierarchical" structures.
    """
    t_start = time.time()
    
    # Facility ordering
    fac_order = _spectral_bisection(a, len(a), t_start, time_budget / 2)
    
    if time.time() - t_start > time_budget:
        return list(range(len(a)))
        
    # Location ordering
    loc_order = _spectral_bisection(b, len(b), t_start, time_budget)
    
    # Construct permutation p such that p[i] is the location for facility i.
    # fac_order[k] is the k-th facility in the hierarchy.
    # loc_order[k] is the k-th location in the hierarchy.
    # We assign facility fac_order[k] to location loc_order[k].
    # So p[fac_order[k]] = loc_order[k].
    
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    for k in range(n):
        p[fac_order[k]] = loc_order[k]
        
    return p.tolist()

def _frank_wolfe_qap(a, b, init_perm, max_iters=300, time_budget=100.0):
    """
    Frank-Wolfe (conditional gradient) on the Birkhoff polytope.
    We minimize f(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    over doubly stochastic X.
    
    The gradient of f w.r.t. X[i,k] is:
    grad[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    (since f is quadratic and symmetric in the two indices for the cost structure)
    
    Actually, let's be more careful. The cost is:
    C(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    
    grad_{i,k} C = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    
    This is because X[i,k] appears in terms where it's the first factor (i,k) and 
    where it's the second factor (j,l) with j=i, l=k.
    
    At each iteration:
    1. Compute gradient G
    2. Solve linear assignment on G to get vertex V (a permutation matrix)
    3. Line search: X_new = (1 - gamma) * X + gamma * V
    4. Project back to Birkhoff polytope (Hungarian on -X_new, or just keep convex combo)
       Actually, for Frank-Wolfe on Birkhoff, the step is X_{t+1} = (1-gamma_t) X_t + gamma_t V_t
       and we don't need to re-project because the convex combo of doubly stochastic matrices
       is doubly stochastic.
    """
    n = len(a)
    t_start = time.time()
    
    # Initialize X as the permutation matrix of init_perm
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), init_perm] = 1.0
    
    best_cost = _qap_cost(a, b, init_perm)
    best_perm = list(init_perm)
    
    prev_grad = None
    step_size = 1.0
    
    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > time_budget:
            break
        
        # Compute gradient: grad[i,k] = sum_j a[i,j] * (B @ X)[j, k] ... let me re-derive
        # C = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        # dC/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
        
        # First term: for fixed i,k, sum_j a[i,j] * sum_l b[k,l] X[j,l] = sum_j a[i,j] (B X)[k,j]... 
        # Wait, let me define M = X @ B^T, so M[j,l] = sum_k X[j,k] B[k,l]? No.
        # Let's use matrix notation. Let S = X (n x n).
        # Term 1: (A @ S @ B)[i, k] where we need to check indices.
        # (A @ S)[i, l] = sum_j A[i,j] S[j,l]
        # (A @ S @ B)[i, k] = sum_l (A@S)[i,l] B[l,k] = sum_j,l A[i,j] S[j,l] B[l,k]
        # But we want sum_j,l A[i,j] B[k,l] S[j,l]. Note B[k,l] vs B[l,k].
        # If B is symmetric, B[k,l] = B[l,k], so Term 1 = (A @ S @ B)[i,k].
        # 
        # Term 2: sum_{j,l} A[j,i] B[l,k] S[j,l]
        # = sum_l B[l,k] sum_j A[j,i] S[j,l]
        # (A^T @ S^T)[l, i] = sum_j A^T[l,j] S^T[i,j] = sum_j A[j,l] S[j,i]... this is getting messy.
        # Let's just do it explicitly with numpy for clarity, then optimize if needed.
        
        # For n=256, explicit computation might be slow. Let's use matrix mults.
        # Assume A and B are symmetric (common in QAP). We'll handle general case too.
        
        # Term 1: T1[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l]
        # = sum_j A[i,j] * (sum_l B[k,l] X[j,l])
        # Let Y = X @ B^T, so Y[j,l'] = sum_l X[j,l] B[l,l']... no.
        # Y[j,k] = sum_l X[j,l] B[l,k] = (X @ B^T)[j,k] if B is not symmetric... 
        # Actually (X @ B)[j,k] = sum_l X[j,l] B[l,k]. So Y = X @ B.
        # Then T1[i,k] = sum_j A[i,j] Y[j,k] = (A @ Y)[i,k] = (A @ X @ B)[i,k].
        
        # Term 2: T2[i,k] = sum_{j,l} A[j,i] B[l,k] X[j,l]
        # = sum_l B[l,k] * (sum_j A[j,i] X[j,l])
        # Z[i,l] = sum_j A[j,i] X[j,l] = (A^T @ X^T)[l,i]^T... 
        # (A^T @ X)[l,i] = sum_j A^T[l,j] X[j,i] = sum_j A[j,l] X[j,i]. Not quite.
        # Let W = A^T @ X^T. W[l,i] = sum_j A^T[l,j] X^T[i,j] = sum_j A[j,l] X[j,i].
        # We want sum_j A[j,i] X[j,l]. That's (A^T @ X)[l,i]? 
        # (A^T @ X)[l,i] = sum_j A^T[l,j] X[j,i] = sum_j A[j,l] X[j,i]. 
        # We want sum_j A[j,i] X[j,l] = sum_j A^T[i,j] X[j,l] = (A^T @ X)[i,l].
        # So sum_j A[j,i] X[j,l] = (A^T @ X)[i,l].
        # Then T2[i,k] = sum_l B[l,k] * (A^T @ X)[i,l] = ((A^T @ X) @ B^T)[i,k].
        
        # So grad = A @ X @ B + (A^T @ X) @ B^T
        
        # For symmetric A and B: grad = A @ X @ B + A @ X @ B = 2 * A @ X @ B
        
        AX = a @ X
        if np.allclose(a, a.T) and np.allclose(b, b.T):
            grad = 2.0 * (AX @ b)
        else:
            ATX = a.T @ X
            grad = (AX @ b) + (ATX @ b.T)
        
        # Solve linear assignment on grad to get next vertex
        row_ind, col_ind = linear_sum_assignment(grad)
        V = np.zeros((n, n), dtype=np.float64)
        V[row_ind, col_ind] = 1.0
        
        # Line search (simple backtracking or fixed step)
        # For Frank-Wolfe, gamma_t = 2/(t+2) is a standard choice
        gamma = 2.0 / (it + 2.0)
        
        X_new = (1.0 - gamma) * X + gamma * V
        
        # Check convergence: if X_new is close to a permutation, extract it
        # Simple check: max deviation from 0/1
        # For now, just update X
        X = X_new
        
        # Periodically extract a permutation and evaluate
        if it % 5 == 0:
            # Extract permutation from X using Hungarian on -X
            r_idx, c_idx = linear_sum_assignment(-X)
            perm = np.empty(n, dtype=np.int64)
            perm[r_idx] = c_idx
            cost = _qap_cost(a, b, perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        
        # Also try the current vertex V as a permutation
        perm_v = np.empty(n, dtype=np.int64)
        perm_v[row_ind] = col_ind
        cost_v = _qap_cost(a, b, perm_v)
        if cost_v < best_cost:
            best_cost = cost_v
            best_perm = list(perm_v)
    
    return best_perm, best_cost


def _local_search_2opt(a, b, p, time_budget=20.0, max_passes=50):
    """
    2-opt local search with O(n) incremental cost delta for swaps.
    
    For swap of positions i and j (facilities at loc_i=p[i], loc_j=p[j]):
    delta = sum_{k != i,j} a[i,k]*(b[loc_i,loc_k]-b[loc_j,loc_k]) + a[j,k]*(b[loc_j,loc_k]-b[loc_i,loc_k])
          + a[i,j]*(b[loc_i,loc_j]-b[loc_j,loc_i])  [if not symmetric]
          + a[j,i]*(b[loc_j,loc_i]-b[loc_i,loc_j])
    Simplified for symmetric b:
    delta = (a[i,j]+a[j,i])*(b[loc_i,loc_j]-b[loc_j,loc_i])  [zero if b symmetric]
          + sum_{k!=i,j} [a[i,k]-a[j,k]]*(b[loc_i,loc_k]-b[loc_j,loc_k])
    """
    n = len(p)
    p = list(p)
    best_cost = _qap_cost(a, b, p)
    t_start = time.time()
    
    # Precompute for fast delta: we need row-wise differences
    # For swap (i,j): delta = sum_k (a[i,k]-a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])  (for k != i,j, and with cross terms)
    # More precisely for general case:
    # old contribution involving i or j:
    #   sum_k a[i,k]*b[p[i],p[k]] + sum_k a[j,k]*b[p[j],p[k]] + a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
    # new contribution (swap p[i],p[j]):
    #   sum_k a[i,k]*b[p[j],p[k]] + sum_k a[j,k]*b[p[i],p[k]] + a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    # delta = new - old
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() - t_start > time_budget:
            break
        
        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - t_start > time_budget:
                    break
                
                li = p[i]
                lj = p[j]
                
                # Compute delta efficiently
                # delta = sum_k (a[i,k]*b[lj,p[k]] - a[i,k]*b[li,p[k]])
                #       + sum_k (a[j,k]*b[li,p[k]] - a[j,k]*b[lj,p[k]])
                #       + a[i,j]*(b[lj,li] - b[li,lj])
                #       + a[j,i]*(b[li,lj] - b[lj,li])
                # = sum_k (a[i,k]-a[j,k])*(b[lj,p[k]] - b[li,p[k]])
                #   + (a[i,j]-a[j,i])*(b[lj,li] - b[li,lj])
                
                # Vectorized computation
                diff_a = a[i] - a[j]  # shape (n,)
                diff_b = b[lj] - b[li]  # shape (n,)
                # For k=i: p[k]=li, so diff_b[li] = b[lj,li]-b[li,li]
                # For k=j: p[k]=lj, so diff_b[lj] = b[lj,lj]-b[li,lj]
                # The formula sum_k diff_a[k]*diff_b[p[k]] handles these correctly
                delta = np.dot(diff_a, diff_b[np.array(p)])
                # But we need to be careful: the cross terms a[i,j]*b[lj,li] etc. are already included
                # Let me verify: when k=j, term is a[i,j]*(b[lj,lj]-b[li,lj]) + ... 
                # Actually the full expansion is:
                # old_i = sum_k a[i,k]*b[li,p[k]], old_j = sum_k a[j,k]*b[lj,p[k]]
                # new_i = sum_k a[i,k]*b[lj,p[k]], new_j = sum_k a[j,k]*b[li,p[k]]
                # delta = new_i + new_j - old_i - old_j
                # = sum_k (a[i,k]-a[j,k])*(b[lj,p[k]] - b[li,p[k]])
                # This is correct and includes the k=i and k=j terms properly.
                
                new_cost = best_cost + int(delta)
                if new_cost < best_cost:
                    p[i], p[j] = p[j], p[i]
                    best_cost = new_cost
                    improved = True
                # else: no swap needed
    
    return p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    total_time_budget = 140.0  # Safety margin before 150s kill
    
    # Phase 1: Initializations
    # 1. Graph Partitioning (Spectral Bisection)
    # This creates a structured initial guess that respects the hierarchical clustering of flows/distances.
    p_init_gp = _graph_partitioning_init(a, b, time_budget=15.0)
    
    # 2. Random initialization
    rng = np.random.RandomState(42)
    p_init_rand = rng.permutation(n).tolist()
    
    # Evaluate initial costs to decide priority
    cost_gp = _qap_cost(a, b, p_init_gp)
    cost_rand = _qap_cost(a, b, p_init_rand)
    
    best_init = p_init_gp if cost_gp <= cost_rand else p_init_rand
    best_init_cost = min(cost_gp, cost_rand)
    
    # Phase 2: Frank-Wolfe relaxation
    # Use the better initial solution as the primary start.
    # Run FW from the best init.
    fw_time_budget = 60.0
    p_fw1, cost_fw1 = _frank_wolfe_qap(a, b, best_init, max_iters=500, time_budget=fw_time_budget)
    
    # If we have time, run FW from the other init (if it was distinct and good enough)
    elapsed = time.time() - t_start
    remaining = total_time_budget - elapsed
    
    p_best = p_fw1
    cost_best = cost_fw1
    
    # If the random init was significantly better or distinct, maybe run another FW?
    # For now, let's stick to one main FW run to ensure we have time for local search,
    # unless the first FW was very fast.
    
    # Actually, let's try running FW from the *other* init if we have > 20s left.
    if remaining > 25.0:
        other_init = p_init_rand if best_init == p_init_gp else p_init_gp
        fw_time_budget2 = min(remaining - 10.0, 40.0)
        p_fw2, cost_fw2 = _frank_wolfe_qap(a, b, other_init, max_iters=300, time_budget=fw_time_budget2)
        if cost_fw2 < cost_best:
            p_best = p_fw2
            cost_best = cost_fw2
    
    elapsed = time.time() - t_start
    remaining = total_time_budget - elapsed
    
    if remaining < 5.0:
        return p_best
    
    # Phase 3: Local search polish
    # Use a robust 2-opt search.
    # We can afford a longer budget here if FW was efficient.
    ls_time_budget = min(remaining - 2.0, 60.0)
    p_ls, cost_ls = _local_search_2opt(a, b, p_best, time_budget=ls_time_budget)
    
    # Final check
    if cost_ls < cost_best:
        p_best = p_ls
        cost_best = cost_ls
        
    return p_best
# EVOLVE-BLOCK-END
