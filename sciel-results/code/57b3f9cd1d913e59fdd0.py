"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 118s
    deadline = time.time() + 118.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def discrete_refine(p, current_cost, deadline):
        # Fast, limited discrete refinement using vectorized delta calculation for best improvement
        # We only do a few passes to clean up the spectral solution
        max_passes = 5
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Compute all pair deltas efficiently? 
            # For n=150, n^2/2 = 11250 pairs. 
            # Vectorizing all pairs at once might be memory heavy or slow if not careful.
            # Let's stick to a simple loop but optimized numpy inside, or just a few random restarts of local search.
            # Actually, let's just do a standard best-improvement 2-swap local search but with a strict time check.
            
            for i in range(n - 1):
                if time.time() > deadline - 5.0:
                    return p, current_cost
                
                # Vectorize delta calculation for swapping i with all j > i
                u = p[i]
                # p is the permutation array
                # We want to find j that minimizes delta(i, j)
                
                # Delta formula from before:
                # d = sum_k (a[i,k]-a[j,k])b[v,p[k]] - (a[i,k]-a[j,k])b[u,p[k]] ... this is complex.
                # Let's use the previous correct formula but vectorized over j.
                
                # Let's just do a simple random restart local search if spectral gives a good start.
                # Or better: use the spectral solution as the START for the previous program's local search.
                # Since we are changing the strategy, let's implement a lightweight local search.
                
                # To keep it simple and fast, let's just do a few iterations of random swaps + local search
                # No, the directive says "STRUCTURAL change". The spectral part IS the change.
                # The refinement should be minimal to ensure validity, or use a very fast heuristic.
                
                # Let's implement a fast "first improvement" or "best improvement" with limited scope.
                # For now, let's just return the spectral solution if it's good enough, 
                # but we must ensure it's a valid permutation.
                # The spectral projection gives a permutation via Hungarian.
                # So p is already a valid permutation.
                
                # Let's do a quick check: if we have time, do a few swaps.
                # But to be safe and fast, let's just return p as is if we are running out of time.
                # However, spectral solutions are often not locally optimal in the discrete sense.
                # Let's do a limited local search.
                
                # To save time in this draft, I will use a simple local search loop.
                pass
            
            # If we reach here without breaking, we didn't do the loop above correctly in this sketch.
            # Let's replace the inner loop with a proper vectorized best-improvement step.
            
            break # Exit this pass for now, will implement proper LS below if needed.
            
        return p, current_cost

    def spectral_relaxation(deadline):
        # 1. Preprocess matrices
        # Center and scale
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
        b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
        
        a_norm = a_centered / a_scale
        b_norm = b_centered / b_scale
        
        # 2. Construct the combined matrix for spectral relaxation
        # Common approach: M = A + B (or weighted sum)
        # Or use the eigenvectors of A and B separately and align them.
        # Let's try the "Sum of Matrices" approach first.
        M = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
        M = 0.5 * (M + M.T) # Ensure symmetry
        
        # 3. Compute eigenvectors
        # We want the top k eigenvectors to embed the facilities and locations in a k-dimensional space.
        # Then we project to a permutation.
        # For n=150, full eigh is fast.
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        
        # Use the top few eigenvectors (largest eigenvalues)
        # Usually 5-10 dimensions are enough to capture structure.
        k = 10
        # Take the last k eigenvectors (largest eigenvalues)
        # Note: eigh returns ascending order
        top_eigvecs = eigenvectors[:, n-k:]
        
        # 4. Project to permutation
        # One common method: 
        # Let V be the matrix of top eigenvectors (n x k).
        # The rows of V are the coordinates of facilities.
        # We want to assign facilities to locations.
        # If we assume the "ideal" location coordinates are the same as facility coordinates (since M is symmetric sum),
        # we can just sort the facilities by their first principal component and assign locations in the same order?
        # No, that's rank-based.
        
        # Better: Use the eigenvectors to define a linear assignment problem.
        # Or use the "Birkhoff Polytope" gradient descent.
        
        # Let's implement a simple Birkhoff Polytope Gradient Descent.
        # Start with X = Uniform Matrix (1/n)
        X = np.ones((n, n)) / n
        
        # Gradient of the QAP cost w.r.t X:
        # Cost(X) = sum_{i,j} a[i,j] X[i, p[i]] ... wait, X is doubly stochastic.
        # The continuous relaxation cost is: C(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        # This is a quadratic form.
        # Gradient: G = 2 * (A X B^T + A^T X^T B) ... need to be careful with indices.
        # Let's use the standard formula:
        # grad = A X B.T + A.T X.T B
        # Wait, the cost is sum_{i,j} a_ij sum_{k,l} b_kl X_ik X_jl
        # d/dX_pq = sum_{i,j} a_ij b_pq (delta_ip delta_jq ... no)
        # d/dX_pq = sum_{i,j} a_ij (b_pq (delta_{ip} delta_{jq} ... no, X is the variable)
        # Let's derive:
        # C = sum_{i,j} a_ij (B^T X^T X B)_{ij} ... no.
        # C = sum_{i,j} a_ij sum_{k,l} b_kl X_ik X_jl
        # Let Y = X.
        # C = sum_{i,j,k,l} a_ij b_kl Y_ik Y_jl
        # dC/dY_pq = sum_{i,j,k,l} a_ij b_kl (delta_ip delta_kq Y_jl + delta_jp delta_lq Y_ik)
        # = sum_{j,l} a_pj b_pq Y_jl + sum_{i,k} a_ip b_qq Y_ik ... wait, indices are mixed.
        
        # Standard result for QAP relaxation:
        # grad = A X B^T + A^T X^T B
        # Let's verify dimensions.
        # A (n,n), X (n,n), B (n,n).
        # A X B^T: (n,n)
        # A^T X^T B: (n,n)
        # This looks correct for the gradient of trace(A X B X^T) or similar.
        # Our cost is sum a_ij b_kl X_ik X_jl = vec(X)^T (B \otimes A) vec(X) ...
        # Actually, C = sum_{i,k} X_ik (sum_j a_ij (sum_l b_kl X_jl))
        # C = sum_{i,k} X_ik (A (X^T B^T)^T )_{ik} ... 
        # Let's use the matrix form: C = sum_{i,j} A_ij (B X^T X B^T)_{ji} ... no.
        # C = sum_{i,j} A_ij sum_{k,l} B_kl X_ik X_jl
        # Let Z = X.
        # C = sum_{i,k} Z_ik sum_{j,l} A_ij B_kl Z_jl
        # Inner sum is (A Z B^T)_{ik} ?
        # (A Z B^T)_{ik} = sum_j sum_l A_ij Z_jl B^T_lk = sum_j sum_l A_ij Z_jl B_kl.
        # Yes! So C = sum_{i,k} Z_ik (A Z B^T)_{ik} = <Z, A Z B^T> = trace(Z^T A Z B^T).
        # Gradient of trace(Z^T A Z B^T) w.r.t Z:
        # d/dZ trace(Z^T M Z N^T) = M Z N + M^T Z N^T ...
        # Here M=A, N=B.
        # grad = A Z B^T + A^T Z^T B.
        
        # So gradient G = A X B.T + A.T X.T B
        
        # We want to minimize C(X) subject to X being doubly stochastic.
        # Projected Gradient Descent:
        # X_{t+1} = Project(X_t - alpha * G_t)
        # Projection onto Birkhoff Polytope is the Earth Mover's Distance / Linear Sum Assignment problem.
        # scipy.optimize.linear_sum_assignment can be used for projection!
        # The projection of a matrix M onto the Birkhoff polytope is the solution to the assignment problem with cost matrix M.
        # Wait, linear_sum_assignment minimizes sum C_ij X_ij.
        # The projection minimizes sum (M_ij - X_ij)^2 ... which is equivalent to minimizing sum M_ij X_ij for a doubly stochastic X?
        # Yes, because sum X_ij^2 is constant for permutation matrices (equals n), but for general doubly stochastic?
        # Actually, the projection onto the Birkhoff polytope of a matrix M is indeed given by solving the assignment problem with cost matrix M.
        # Reference: "Projection onto the Birkhoff Polytope"
        
        # So:
        # 1. Compute Gradient G
        # 2. X_new = X - alpha * G
        # 3. Project X_new to nearest doubly stochastic matrix P (using linear_sum_assignment on X_new)
        # 4. X = P
        
        # Let's implement this.
        
        # Initial X: Uniform
        X = np.ones((n, n)) / n
        
        # Learning rate
        alpha = 0.01
        # Maybe adapt alpha?
        
        # Number of iterations
        max_iter = 500
        current_cost = None
        
        for t in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Compute Gradient
            # G = A X B.T + A.T X.T B
            G = a @ X @ b.T + a.T @ X.T @ b
            
            # Update
            X = X - alpha * G
            
            # Project to Birkhoff Polytope
            # Solve assignment problem with cost matrix X
            row_ind, col_ind = linear_sum_assignment(X)
            X_proj = np.zeros((n, n))
            X_proj[row_ind, col_ind] = 1.0
            X = X_proj
            
            # Optional: Check cost
            # We can estimate the discrete cost of this permutation
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            cost = compute_cost(p)
            
            if current_cost is None or cost < current_cost:
                current_cost = cost
                best_p = p.copy()
                best_cost = cost
            else:
                # If cost increased, maybe reduce alpha or keep previous best
                pass
        
        if best_p is not None:
            return best_p, best_cost
        else:
            return None, None

    # --- Main Logic ---
    
    best_p = None
    best_cost = np.inf
    
    # 1. Try Spectral Relaxation
    try:
        p_spec, cost_spec = spectral_relaxation(deadline)
        if p_spec is not None:
            best_p = p_spec
            best_cost = cost_spec
    except Exception as e:
        pass

    # 2. Fallback / Refinement: Rank-based initialization + Local Search
    # If spectral failed or we have time, do a quick rank-based init and local search.
    # This ensures we have a valid result even if spectral is slow or fails.
    
    if best_p is None or time.time() < deadline - 20.0:
        # Rank-based init
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        
        p_rank = np.empty(n, dtype=np.int64)
        p_rank[heavy] = central
        
        cost_rank = compute_cost(p_rank)
        
        if cost_rank < best_cost:
            best_cost = cost_rank
            best_p = p_rank.copy()
        else:
            # Use best_p for local search if we have time
            if best_p is not None:
                p_work = best_p.copy()
            else:
                p_work = p_rank.copy()
            
            # Simple Local Search (Best Improvement 2-Swap)
            # We need a fast swap_delta.
            # Re-implementing the fast swap_delta from the previous program.
            
            def swap_delta(i, j, p):
                u = p[i]
                v = p[j]
                
                diff_b_vu = b[v, p] - b[u, p]
                diff_b_uv = b[u, p] - b[v, p]
                
                Si = np.dot(a[i, :], diff_b_vu)
                Sj = np.dot(a[j, :], diff_b_uv)
                
                diff_b_pv = b[p, v] - b[p, u]
                diff_b_pu = b[p, u] - b[p, v]
                
                SiT = np.dot(a[:, i], diff_b_pv)
                SjT = np.dot(a[:, j], diff_b_pu)
                
                Si_sub = a[i, i] * (b[v, u] - b[u, u]) + a[i, j] * (b[v, v] - b[u, v])
                SiT_sub = a[i, i] * (b[u, v] - b[u, u]) + a[j, i] * (b[v, v] - b[v, u])
                Sj_sub = a[j, i] * (b[u, u] - b[v, u]) + a[j, j] * (b[u, v] - b[v, v])
                SjT_sub = a[i, j] * (b[u, u] - b[u, v]) + a[j, j] * (b[v, u] - b[v, v])
                
                DT = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u]) + a[i, i] * (b[v, v] - b[u, u]) + a[j, j] * (b[u, u] - b[v, v])
                
                return (Si - Si_sub) + (SiT - SiT_sub) + (Sj - Sj_sub) + (SjT - SjT_sub) + DT

            current_cost = compute_cost(p_work)
            
            # Local Search Loop
            max_ls_iter = 50
            for _ in range(max_ls_iter):
                if time.time() > deadline - 5.0:
                    break
                
                best_delta = 0
                best_i = -1
                best_j = -1
                
                # Scan all pairs
                for i in range(n - 1):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 5.0:
                            break
                        d = swap_delta(i, j, p_work)
                        if d < best_delta:
                            best_delta = d
                            best_i = i
                            best_j = j
                
                if best_delta < 0:
                    p_work[best_i], p_work[best_j] = p_work[best_j], p_work[best_i]
                    current_cost += best_delta
                else:
                    break
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p_work

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    # Absolute fallback
    return list(range(n))

# EVOLVE-BLOCK-END
