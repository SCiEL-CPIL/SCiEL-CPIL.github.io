"""
Spectral-Embedding Initialization + Incremental 2-Opt Local Search for QAP wil100.
Strategy:
1. Use Spectral Embedding (eigenvectors of A and B) to create a high-quality initial permutation.
   This aligns the topological structure of the flow graph with the distance graph,
   which is a stronger structural prior than simple row-sum correlations.
2. Use a fast, incremental 2-Opt local search. Instead of O(n^2) random swaps (SA),
   we systematically evaluate swaps. By precomputing interaction terms, we can
   evaluate the delta of a swap (i, j) in O(1) or O(n) depending on update strategy.
   For n=100, O(n) per swap is ~100 ops, allowing ~100k swaps/sec in Python/NumPy mix.
   We use a "First Improvement" strategy to quickly descend the cost landscape.
3. If stagnation occurs (no improvement in X steps), we perform a "Kick" (random perturbation)
   to escape local minima, effectively acting as a fast, deterministic SA alternative.
"""

import numpy as np
import time
from scipy.sparse.linalg import eigsh

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Generate initial permutation using spectral embedding.
    We project facilities and locations onto the principal components (eigenvectors)
    of their respective matrices and match them based on cosine similarity or coordinate alignment.
    """
    n = len(a)
    
    # Center matrices to remove bias from global sums
    a_centered = a - np.mean(a, axis=1, keepdims=True) - np.mean(a, axis=0, keepdims=True) + np.mean(a)
    b_centered = b - np.mean(b, axis=1, keepdims=True) - np.mean(b, axis=0, keepdims=True) + np.mean(b)
    
    # Get top k eigenvectors. For QAP, top few capture most structural variance.
    k = min(5, n - 1)
    
    try:
        # eigsh requires symmetric matrices. QAP matrices are usually symmetric or near-symmetric.
        # If not perfectly symmetric, symmetrize: (M + M.T) / 2
        a_sym = (a_centered + a_centered.T) / 2.0
        b_sym = (b_centered + b_centered.T) / 2.0
        
        # Find largest eigenvalues (most structure)
        # sigma='LM' for Largest Magnitude
        evals_a, evecs_a = eigsh(a_sym, k=k, which='LM')
        evals_b, evecs_b = eigsh(b_sym, k=k, which='LM')
        
        # Ensure consistent orientation (signs can flip)
        # We want to match the "shape". 
        # Simple heuristic: align the first principal component to have positive sum for first half
        # Or just project and use Hungarian on the embedding space.
        
        # Projection: E_f = A @ v_a, E_l = B @ v_b
        E_f = evecs_a.T # shape (k, n)
        E_l = evecs_b.T # shape (k, n)
        
        # We want to assign facility i to location j such that E_f[i] ~ E_l[j]
        # Construct cost matrix in embedding space: C[i, j] = || E_f[i] - E_l[j] ||^2
        # This is equivalent to minimizing distance in the spectral subspace.
        
        # Efficient computation of pairwise squared distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        F_sq = np.sum(E_f**2, axis=1) # (n,)
        L_sq = np.sum(E_l**2, axis=1) # (n,)
        
        # Dist matrix (n, n)
        D = F_sq[:, np.newaxis] + L_sq[np.newaxis, :] - 2 * (E_f @ E_l.T)
        
        # Solve assignment problem on this surrogate cost
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(D)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    except Exception:
        # Fallback to greedy row-sum matching if eigsh fails
        a_sums = np.sum(a, axis=1)
        b_sums = np.sum(b, axis=1)
        # Sort both and match
        idx_a = np.argsort(a_sums)
        idx_b = np.argsort(b_sums)
        p = np.empty(n, dtype=np.int64)
        p[idx_a] = idx_b
        return p

def _compute_delta(a, b, p, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    p[i] is facility at location i? No, p[facility] = location.
    Wait, standard QAP: sum a[i,j] b[p[i], p[j]].
    Swap p[i] and p[j] means facility i moves to p[j] and facility j moves to p[i].
    This is a 2-exchange move on the assignment.
    
    Let u = p[i], v = p[j].
    Old terms involving i or j:
    1. a[i,i]b[u,u] + a[j,j]b[v,v]
    2. a[i,j]b[u,v] + a[j,i]b[v,u]
    3. sum_{k!=i,j} a[i,k]b[u,p[k]] + a[k,i]b[p[k],u]
    4. sum_{k!=i,j} a[j,k]b[v,p[k]] + a[k,j]b[p[k],v]
    
    New terms:
    1. a[i,i]b[v,v] + a[j,j]b[u,u]
    2. a[i,j]b[v,u] + a[j,i]b[u,v]
    3. sum_{k!=i,j} a[i,k]b[v,p[k]] + a[k,i]b[p[k],v]
    4. sum_{k!=i,j} a[j,k]b[u,p[k]] + a[k,j]b[p[k],u]
    
    Delta = New - Old
    """
    u = p[i]
    v = p[j]
    
    # Direct pair (i,j)
    delta = (a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v]) + 
             a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]))
             
    # Cross terms with all other k
    # We can vectorize this.
    # Term for facility i moving from u to v:
    # Sum_k (a[i,k] * (b[v, p[k]] - b[u, p[k]]) + a[k,i] * (b[p[k], v] - b[p[k], u]))
    # Exclude k=i and k=j manually or via masking (masking is slower, just subtract)
    
    diff_b_row_v_u = b[v, :] - b[u, :] # (n,)
    diff_b_col_v_u = b[:, v] - b[:, u] # (n,)
    
    # Sum over all k
    sum_i = np.dot(a[i, :], diff_b_row_v_u) + np.dot(a[:, i], diff_b_col_v_u)
    # Subtract k=i: a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[v,u]) -> 0 if b symmetric, else terms
    # Actually, let's just compute the full sum and subtract the specific i and j terms.
    # Full sum for i: sum_{k=0}^{n-1} ...
    # The term for k=i in the sum is: a[i,i]*(b[v, p[i]] - b[u, p[i]]) + a[i,i]*(b[p[i], v] - b[p[i], u])
    # p[i] is u. So: a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[u,u])? No.
    # b[p[i], v] is b[u, v]. b[p[i], u] is b[u, u].
    # So term_k=i = a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[u,u])
    
    term_k_i = a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[u,u])
    
    # The term for k=j in the sum for facility i:
    # a[i,j]*(b[v, p[j]] - b[u, p[j]]) + a[j,i]*(b[p[j], v] - b[p[j], u])
    # p[j] is v.
    term_k_j = a[i,j]*(b[v,v]-b[u,v]) + a[j,i]*(b[v,v]-b[v,u])
    
    sum_i_excl = sum_i - term_k_i - term_k_j
    
    # Similarly for facility j moving from v to u:
    diff_b_row_u_v = b[u, :] - b[v, :]
    diff_b_col_u_v = b[:, u] - b[:, v]
    
    sum_j = np.dot(a[j, :], diff_b_row_u_v) + np.dot(a[:, j], diff_b_col_u_v)
    
    # Term for k=i in sum for j:
    # a[j,i]*(b[u, p[i]] - b[v, p[i]]) + a[i,j]*(b[p[i], u] - b[p[i], v])
    # p[i] is u.
    term_k_i_j = a[j,i]*(b[u,u]-b[v,u]) + a[i,j]*(b[u,u]-b[u,v])
    
    # Term for k=j in sum for j:
    # a[j,j]*(b[u, p[j]] - b[v, p[j]]) + a[j,j]*(b[p[j], u] - b[p[j], v])
    # p[j] is v.
    term_k_j_j = a[j,j]*(b[u,v]-b[v,v]) + a[j,j]*(b[v,u]-b[v,v])
    
    sum_j_excl = sum_j - term_k_i_j - term_k_j_j
    
    delta += sum_i_excl + sum_j_excl
    
    return int(delta)

def _local_search_2opt(a, b, p, time_limit):
    """
    First-improvement 2-opt local search with incremental updates.
    To be fast, we iterate through pairs. If no improvement found in a full pass, we stop.
    To avoid O(n^2) overhead in Python loops for n=100 (4950 pairs), we vectorize the delta calculation
    for a fixed i against all j > i.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 3 # Allow a few passes of no improvement before declaring local min
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Iterate i
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return best_p, best_cost
            
            # Vectorize delta calculation for swapping i with all j > i
            # We can't easily vectorize the complex delta formula for all j at once
            # because it depends on p[j]. 
            # However, for n=100, a Python loop of 100 * 50 = 5000 iterations per pass
            # with ~100 numpy ops each is ~500k ops. That's ~0.1-0.5s per pass.
            # This is too slow if we need many passes.
            
            # Optimization: Precompute parts of the delta.
            # Let's just use the O(n) delta function in a tight loop.
            for j in range(i + 1, n):
                delta = _compute_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    # First improvement: break inner loop to restart scan from beginning
                    # or continue? Restarting is safer for finding local min faster.
                    # But restarting O(n^2) scan is expensive.
                    # Let's just break inner loop and continue outer loop.
                    break
            
            if improved:
                no_improve_count = 0
                break # Restart the entire scan
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        if not improved:
            break

    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 145.0 # Safety margin for 150s limit
    start = time.time()
    
    np.random.seed(42)
    
    # 1. Spectral Initialization
    p_init = _spectral_initial(a, b)
    cost_init = _qap_cost(a, b, p_init)
    
    best_p = p_init.copy()
    best_cost = cost_init
    
    # 2. Local Search Phase 1: Polishing the spectral solution
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        ls_time = min(10.0, time_remaining * 0.2)
        p_ls, cost_ls = _local_search_2opt(a, b, best_p.copy(), ls_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # 3. Diversification & Local Search Loop
    # Since we have a strong local search, we can use it as a "black box"
    # and apply kicks (perturbations) to escape local minima.
    
    kicks = 0
    while time.time() - start < deadline - 5:
        # Check if we are stuck (simple heuristic: if cost hasn't changed in last kick, maybe stuck)
        # For now, just run another kick + LS
        
        # Perturb: Randomly swap 10-20 pairs
        p_kick = best_p.copy()
        n_swaps = 15
        for _ in range(n_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_kick[i], p_kick[j] = p_kick[j], p_kick[i]
        
        # Local search on kicked solution
        time_remaining = deadline - (time.time() - start) - 5 # reserve 5s for final check
        if time_remaining <= 0:
            break
            
        # Allocate time based on remaining budget
        ls_time = min(10.0, time_remaining * 0.5)
        
        p_ls, cost_ls = _local_search_2opt(a, b, p_kick, ls_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            kicks = 0
        else:
            kicks += 1
            # If too many failed kicks, increase perturbation size
            if kicks % 5 == 0:
                n_swaps += 5
                
    # 4. Final Polish
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 2:
        p_final, cost_final = _local_search_2opt(a, b, best_p.copy(), time_remaining - 1)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()
