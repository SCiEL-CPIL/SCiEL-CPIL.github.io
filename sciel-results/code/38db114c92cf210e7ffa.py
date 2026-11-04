"""
Deterministic Multi-Start First-Improvement Local Search for QAPLIB wil100.
Strategy:
1. Generate diverse high-quality initial permutations using:
   - Hungarian matching on row-sum centralities.
   - Spectral initialization (eigenvector alignment).
   - Random restarts.
2. Perform strict First-Improvement Local Search (FLS) from each seed.
   - Uses O(n) incremental delta update for 2-swap moves.
   - Scans all n(n-1) moves in a deterministic order.
   - Accepts the first improving move found, repeats until no improving move exists.
3. This approach is deterministic given the seeds, avoids the stochastic variance of SA,
   and is significantly faster per evaluation due to the O(n) delta update and lack of
   population management overhead.
4. Returns the best permutation found across all seeds.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian matching on row-sum centralities.
    Heavy facilities (high row sum in a) go to central locations (low row sum in b).
    """
    n = len(a)
    a_row_sums = a.sum(axis=1).astype(np.float64)
    b_row_sums = b.sum(axis=1).astype(np.float64)
    
    # Cost matrix: we want to minimize sum of a_row_sums[i] * b_row_sums[p[i]]
    cost_matrix = -np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _spectral_initial(a, b):
    """
    Spectral initialization: align principal eigenvectors of a and b.
    """
    n = len(a)
    # Use symmetric parts for stability
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Get top eigenvectors
        # For n=100, full eigendecomposition is fast
        w_a, v_a = np.linalg.eigh(a_sym)
        w_b, v_b = np.linalg.eigh(b_sym)
        
        # Align by matching eigenvectors with largest magnitudes
        # Simple heuristic: sort by eigenvalue magnitude, match signs
        idx_a = np.argsort(-np.abs(w_a))
        idx_b = np.argsort(-np.abs(w_b))
        
        v_a_top = v_a[:, idx_a[:10]]
        v_b_top = v_b[:, idx_b[:10]]
        
        # Project facilities and locations onto these eigenvectors
        proj_a = v_a_top.T @ a_sym # (10, n)
        proj_b = v_b_top.T @ b_sym # (10, n)
        
        # Create a cost matrix based on alignment
        # We want facility i to be at location p[i] such that their projections are similar
        # This is a linear assignment problem on the projected features
        # Cost[i, j] = - sum_k (proj_a[k, i] * proj_b[k, j])
        cost_matrix = - (proj_a @ proj_b.T)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except Exception:
        return np.random.permutation(n)

def _first_improvement_ls(a, b, p, time_limit, start_time):
    """
    First-Improvement Local Search with O(n) delta update.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    # Precompute vectors for fast delta calculation
    # delta(i, j) for swapping p[i] and p[j]
    # Let x = p[i], y = p[j]
    # delta = sum_k (a[i,k]-a[j,k])*(b[y,p[k]]-b[x,p[k]]) + sum_k (a[k,i]-a[k,j])*(b[p[k],y]-b[p[k],x]) - 2*correction
    # We can precompute row/col sums of a and b to speed up, but O(n) per move is already very fast for n=100.
    
    # To make it even faster, we can precompute:
    # S_a_row[i] = sum_k a[i,k]
    # S_a_col[i] = sum_k a[k,i]
    # But the delta depends on the current permutation p.
    
    # Let's use the vectorized delta calculation for a single swap.
    # For FLS, we iterate i from 0 to n-1, j from i+1 to n-1.
    # If delta < 0, swap and restart scan from i=0 (or continue from i+1? 
    # Standard FLS usually continues or restarts. Restarting is safer for finding the true local optimum faster in practice for QAP).
    # Actually, a common efficient FLS variant is:
    # for i in range(n):
    #   for j in range(i+1, n):
    #     if delta(i,j) < 0:
    #       swap(i,j)
    #       break # restart inner loop or outer? 
    #       # If we break inner, we continue to next i. This is "first improvement" in the scan order.
    #       # It is guaranteed to terminate.
    
    # Let's implement the scan.
    # To speed up, we can precompute b_sub = b[p, p] ? No, p changes.
    
    # Efficient delta for swap (i, j) given current p:
    # x = p[i], y = p[j]
    # delta = sum_{k!=i,j} a[i,k]b[y,p[k]] + a[j,k]b[x,p[k]] - a[i,k]b[x,p[k]] - a[j,k]b[y,p[k]]
    #        + sum_{k!=i,j} a[k,i]b[p[k],y] + a[k,j]b[p[k],x] - a[k,i]b[p[k],x] - a[k,j]b[p[k],y]
    #        + terms involving i,j directly.
    
    # A simpler vectorized way for a single swap:
    def _swap_delta_single(p, i, j):
        x = p[i]
        y = p[j]
        # Create a temporary permutation to compute cost? No, too slow.
        # Use the formula:
        # delta = (a[i, :] - a[j, :]) @ (b[y, p] - b[x, p]) + (a[:, i] - a[:, j]) @ (b[p, y] - b[p, x])
        #         - 2 * (a[i, i] - a[j, i]) * (b[y, x] - b[x, x]) ... wait, corrections are tricky.
        
        # Let's use the robust method:
        # cost_new = cost_old - terms_involving_i_j_old + terms_involving_i_j_new
        # This is O(n) if done carefully.
        
        # Terms involving i or j in old cost:
        # sum_k a[i,k]b[x,p[k]] + a[k,i]b[p[k],x]
        # sum_k a[j,k]b[y,p[k]] + a[k,j]b[p[k],y]
        # Note: a[i,j]b[x,y] and a[j,i]b[y,x] are counted in both sums? 
        # In QAP cost sum_{i,j} a[i,j]b[p[i],p[j]], the term (i,j) is a[i,j]b[x,y].
        # The term (j,i) is a[j,i]b[y,x].
        # In the sum for i: a[i,j]b[x,y] is included.
        # In the sum for j: a[j,i]b[y,x] is included.
        # So we subtract these two sums and add the new ones.
        
        # Old terms:
        # S_i_old = sum_{k!=j} a[i,k]b[x,p[k]] + a[i,j]b[x,y] + sum_{k!=j} a[k,i]b[p[k],x] + a[j,i]b[y,x]
        # This is getting messy. Let's use the vectorized difference which is O(n) and clean.
        
        # diff_a_row = a[i, :] - a[j, :]  (n,)
        # diff_a_col = a[:, i] - a[:, j]  (n,)
        # diff_b_row = b[y, p] - b[x, p]  (n,)
        # diff_b_col = b[p, y] - b[p, x]  (n,)
        
        # delta = dot(diff_a_row, diff_b_row) + dot(diff_a_col, diff_b_col)
        #        - (a[i,i]-a[j,i])*(b[y,x]-b[x,x])  # Correction for k=i in first sum?
        #        - (a[i,j]-a[j,j])*(b[y,y]-b[x,y])  # Correction for k=j in first sum?
        #        - (a[i,i]-a[i,j])*(b[x,y]-b[x,x])  # Correction for k=i in second sum?
        #        - (a[j,i]-a[j,j])*(b[y,y]-b[y,x])  # Correction for k=j in second sum?
        
        # Actually, the formula:
        # delta = sum_k (a[i,k]-a[j,k])(b[y,p[k]]-b[x,p[k]]) + sum_k (a[k,i]-a[k,j])(b[p[k],y]-b[p[k],x])
        #        - (a[i,i]-a[j,i])(b[y,x]-b[x,x]) - (a[i,j]-a[j,j])(b[y,y]-b[x,y])
        #        - (a[i,i]-a[i,j])(b[x,y]-b[x,x]) - (a[j,i]-a[j,j])(b[y,y]-b[y,x])
        
        # Let's verify for k=i in first sum:
        # Term is (a[i,i]-a[j,i])(b[y,x]-b[x,x]). We subtract it. Correct.
        # For k=j in first sum:
        # Term is (a[i,j]-a[j,j])(b[y,y]-b[x,y]). We subtract it. Correct.
        
        # This is O(n).
        
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[y, p] - b[x, p]
        diff_b_col = b[p, y] - b[p, x]
        
        delta = np.dot(diff_a_row, diff_b_row) + np.dot(diff_a_col, diff_b_col)
        
        # Corrections
        delta -= (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        delta -= (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        delta -= (a[i, i] - a[i, j]) * (b[x, y] - b[x, x])
        delta -= (a[j, i] - a[j, j]) * (b[y, y] - b[y, x])
        
        return delta

    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_delta_single(p, i, j)
                if delta < 0:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break # Restart inner loop? Or just break inner and continue outer?
                    # Breaking inner and continuing outer is standard FLS. 
                    # It ensures we don't miss moves earlier in the scan.
                    # However, breaking both and restarting from i=0 is often faster in practice for QAP.
                    # Let's try restarting from i=0 for stronger convergence.
                    # But that might be slower. Let's stick to breaking inner and continuing outer.
                    # Wait, if we break inner, we move to i+1. We might miss an improvement at i, i+2 if it was created by the swap at i, j.
                    # So restarting from i=0 is safer for finding the local optimum.
                    # Let's use a flag to restart.
                    break
            if improved:
                # Restart from i=0
                # We can achieve this by setting i = -1 and letting the loop increment? No.
                # Just break out of both loops and let the while loop restart.
                break
                
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin before 150s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Generate a set of diverse initial permutations
    seeds = []
    
    # 1. Hungarian
    p_hung = _hungarian_initial(a, b)
    seeds.append(p_hung)
    
    # 2. Spectral
    p_spec = _spectral_initial(a, b)
    seeds.append(p_spec)
    
    # 3. Random restarts
    num_random = 20
    for _ in range(num_random):
        p_rand = np.random.permutation(n)
        seeds.append(p_rand)
        
    # 4. Perturbed Hungarian
    for _ in range(10):
        p_pert = p_hung.copy()
        for _ in range(5):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        seeds.append(p_pert)

    best_p = None
    best_cost = np.inf
    
    # Allocate time per seed
    # We have ~145s. 31 seeds. ~4.6s per seed.
    # FLS is fast. 100x100 moves = 10k moves. Each move O(n)=100 ops. 1M ops per full scan.
    # Python overhead is high. 1M ops might take 0.1s? 
    # Let's give each seed a time budget.
    
    time_per_seed = (total_deadline - (time.time() - start_time)) / len(seeds)
    
    for idx, p_seed in enumerate(seeds):
        remaining = total_deadline - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        # Give this seed a portion of the remaining time
        # But ensure at least some time for the last few
        t_limit = min(time_per_seed, remaining)
        
        p_sol, cost_sol = _first_improvement_ls(a, b, p_seed.copy(), t_limit, start_time)
        
        if cost_sol < best_cost:
            best_cost = cost_sol
            best_p = p_sol.copy()
            
        # Optional: if we have time left and haven't improved in a while, do a quick polish?
        # The FLS should have converged.
        
    if best_p is None:
        # Fallback
        best_p = np.random.permutation(n)
        
    return best_p.tolist()
