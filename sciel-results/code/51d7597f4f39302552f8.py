"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _sa_search(a, b, p_init, time_limit, start_time, rng):
    """
    Simulated Annealing with Swap and Or-opt moves.
    """
    n = len(p_init)
    p = p_init.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute for speed (not strictly necessary if we recompute deltas, but good for potential future use)
    # We will compute delta on the fly to save memory if n is large, but n=256 is small.
    
    best_p = p.copy()
    best_cost = current_cost
    
    # Cooling schedule
    T_start = max(1000.0, current_cost * 0.01)
    T_end = 1e-6
    alpha = 0.995  # Cooling rate
    
    T = T_start
    
    while time.time() - start_time < time_limit:
        if T <= T_end:
            T = T_end
            # Optional: restart with higher T if stuck, but for now just keep going at low T (greedy-like)
            # Actually, better to just break or do greedy.
            # Let's implement a simple greedy phase at the end if T is low.
            
        # Choose move type: 70% Swap, 30% Or-opt
        move_type = rng.random()
        
        if move_type < 0.7:
            # Swap move
            i, j = rng.choice(n, size=2, replace=False)
            if i == j: continue
            
            # Compute delta for swapping i and j
            # Cost = sum a[i,k] b[p[i], p[k]] ...
            # Delta = (a[i,i] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[i]]) 
            #       + (a[i,j] - a[j,i]) * (b[p[i], p[j]] - b[p[j], p[i]])
            #       + sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
            
            pi = p[i]
            pj = p[j]
            
            # Vectorized calculation
            # Let u = p[i], v = p[j]
            # Terms involving i and j in the sum
            # Sum over k: a[i,k] b[u, p[k]] + a[j,k] b[v, p[k]]
            # New: a[i,k] b[v, p[k]] + a[j,k] b[u, p[k]]
            
            # We need b[u, :], b[v, :] and a[i, :], a[j, :]
            # Note: D = b[np.ix_(p, p)]
            # D[i, :] is b[pi, p]
            # D[j, :] is b[pj, p]
            
            # Delta = sum_k (a[i,k] - a[j,k]) * (D[j,k] - D[i,k])
            #        + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])  # Diag terms handled in sum?
            # Let's stick to the full formula for correctness.
            
            # diff_a = a[i] - a[j]
            # diff_D = D[j] - D[i]
            # Base sum = sum(diff_a * diff_D)
            
            # Corrections for k=i and k=j
            # When k=i:
            # Original: a[i,i] b[u,u] + a[j,i] b[v,u]
            # New: a[i,i] b[v,u] + a[j,i] b[u,u]
            # Diff: a[i,i](b[v,u]-b[u,u]) + a[j,i](b[u,u]-b[v,u]) = (a[i,i]-a[j,i]) * (b[v,u]-b[u,u])
            # In D terms: D[j, i] is b[pj, pi] = b[v, u]. D[i, i] is b[u, u].
            # So term_i = (a[i,i] - a[j,i]) * (D[j,i] - D[i,i])
            
            # When k=j:
            # Original: a[i,j] b[u,v] + a[j,j] b[v,v]
            # New: a[i,j] b[v,v] + a[j,j] b[u,v]
            # Diff: a[i,j](b[v,v]-b[u,v]) + a[j,j](b[u,v]-b[v,v]) = (a[i,j]-a[j,j]) * (b[v,v]-b[u,v])
            # In D terms: D[j,j] is b[v,v]. D[i,j] is b[u,v].
            # So term_j = (a[i,j] - a[j,j]) * (D[j,j] - D[i,j])
            
            # The base sum includes k=i and k=j.
            # Base sum includes:
            # k=i: (a[i,i]-a[j,i]) * (D[j,i] - D[i,i])
            # k=j: (a[i,j]-a[j,j]) * (D[j,j] - D[i,j])
            
            # Wait, the formula `sum(diff_a * diff_D)` computes:
            # k=i: (a[i,i]-a[j,i]) * (b[v, pi] - b[u, pi]) = (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
            # k=j: (a[i,j]-a[j,j]) * (b[v, pj] - b[u, pj]) = (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
            
            # These match the corrections exactly. So `delta = np.sum((a[i]-a[j]) * (D[j]-D[i]))` is correct?
            # Let's verify.
            # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
            # Swap i,j.
            # Change in cost = 
            # (a[i,i] b[v,v] + a[i,j] b[v,u] + a[j,i] b[u,v] + a[j,j] b[u,u]) 
            # - (a[i,i] b[u,u] + a[i,j] b[u,v] + a[j,i] b[v,u] + a[j,j] b[v,v])
            # = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
            # = (a[i,i]-a[j,j]) * (b[v,v]-b[u,u]) + (a[i,j]-a[j,i]) * (b[v,u]-b[u,v])
            
            # My vectorized formula:
            # sum_k (a[i,k]-a[j,k]) * (b[v, p[k]] - b[u, p[k]])
            # k=i: (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
            # k=j: (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
            # k!=i,j: (a[i,k]-a[j,k]) * (b[v, p[k]] - b[u, p[k]])
            
            # Let's check if these are equivalent.
            # Vector sum = (a[i,i]-a[j,i]) b[v,u] - (a[i,i]-a[j,i]) b[u,u] 
            #            + (a[i,j]-a[j,j]) b[v,v] - (a[i,j]-a[j,j]) b[u,v]
            #            + sum_{k!=i,j} ...
            
            # Exact Delta = (a[i,i]-a[j,j]) b[v,v] - (a[i,i]-a[j,j]) b[u,u] 
            #             + (a[i,j]-a[j,i]) b[v,u] - (a[i,j]-a[j,i]) b[u,v]
            #             + sum_{k!=i,j} (a[i,k]-a[j,k]) b[v, p[k]] - sum_{k!=i,j} (a[i,k]-a[j,k]) b[u, p[k])
            
            # Difference in k=i term:
            # Vector: (a[i,i]-a[j,i]) b[v,u] - (a[i,i]-a[j,i]) b[u,u]
            # Exact: (a[i,j]-a[j,i]) b[v,u] - (a[i,i]-a[j,j]) b[u,u]  <-- Wait, exact has b[u,u] with a[i,i]-a[j,j]?
            # Exact term for b[u,u]: from a[i,i] b[u,u] (old) and a[j,j] b[u,u] (new).
            # So coefficient is a[j,j] - a[i,i].
            # Vector term for b[u,u]: from k=i part: -(a[i,i]-a[j,i]) = a[j,i]-a[i,i].
            # These are different if a[i,i] != a[j,i] or a[j,j] != a[i,i].
            
            # So simple dot product is NOT sufficient. We need to add corrections.
            
            delta = np.sum((a[i] - a[j]) * (D[j] - D[i]))
            
            # Correction for k=i
            # Vector has: (a[i,i]-a[j,i]) * (D[j,i] - D[i,i])
            # Exact needs: (a[i,j]-a[j,i]) * D[j,i]  + (a[j,j]-a[i,i]) * D[i,i]
            # Wait, let's just use the explicit formula for the 4 corner terms and sum the rest.
            
            # It's safer to compute:
            # delta = 0
            # delta += (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            # delta += (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            # delta += np.sum((a[i,2:] - a[j,2:]) * (D[j,2:] - D[i,2:]))  # excluding 0,1? No, indices are i,j.
            
            # Let's do it properly with masks or just accept the O(N) cost for the sum part.
            # N=256, O(N) is fast.
            
            # Sum over k != i, j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            delta += np.sum((a[i, mask] - a[j, mask]) * (D[j, mask] - D[i, mask]))
            
            # Corner terms
            delta += (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
            delta += (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            
            if delta < 0 or (rng.random() < np.exp(-delta / T)):
                # Apply swap
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(delta)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
        else:
            # Or-opt move: move block of size 1, 2, or 3
            size = rng.choice([1, 2, 3])
            if size > n:
                size = n
            start = rng.integers(0, n - size)
            end = start + size
            
            # Find destination
            # Insert before or after a random position
            # Simpler: Pick a random destination index `dest`
            # If dest is in [start, end), skip or adjust.
            # Let's just pick a random `dest` and insert the block there.
            
            # To keep it simple and valid permutation:
            # Remove block, insert at `dest`.
            
            block = p[start:end].copy()
            
            # Determine insertion index
            # We want to insert `block` such that it's a valid permutation.
            # Let's pick a random position `ins_idx` from 0 to n-1.
            # If we insert at `ins_idx`, the elements shift.
            
            # Easier: Pick two indices `u` and `v` (start of block, and destination).
            # If u < v:
            #   p = p[:u] + p[v+size:v+size] ... this is getting complicated with indices.
            
            # Let's use a simpler Or-opt:
            # Pick `i` (start) and `j` (destination index in the *current* array, effectively).
            # Move p[i:i+size] to position j.
            
            i = start
            # Pick j such that j is not in [i, i+size)
            j = rng.integers(0, n)
            if i <= j < i + size:
                j = (j + size) % n
                if i <= j < i + size: # still in range? unlikely but check
                    j = (j + 1) % n

            # Calculate delta for Or-opt is complex.
            # Given time constraints, we will just perform the move and recompute cost if it's cheap enough?
            # No, n=256, recompute is O(N^2) = 65k ops. Very fast.
            # But we have D. Updating D for Or-opt is O(N * size).
            
            # Let's just recompute cost for Or-opt moves to be safe, as delta derivation is prone to error.
            # Or-opt is only 30% of moves.
            
            # Apply move
            new_p = p.copy()
            block = new_p[i:i+size]
            # Remove block
            new_p = np.concatenate([new_p[:i], new_p[i+size:]])
            # Insert at j
            # Note: j index might have shifted if j > i
            if j > i:
                j -= size
            new_p = np.concatenate([new_p[:j], block, new_p[j:]])
            
            # Compute new cost
            new_D = b[np.ix_(new_p, new_p)]
            new_cost = int(np.sum(a * new_D))
            
            if new_cost < current_cost or (rng.random() < np.exp(-(new_cost - current_cost) / T)):
                p = new_p
                D = new_D
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()

        # Cool down
        T *= alpha
        
        # Optional: if T is very low, force greedy
        if T < 1e-4:
             # Run a quick greedy 2-opt pass?
             # For now, just continue with random swaps which will mostly be accepted if improving.
             pass

    return best_p, best_cost

def _eigen_init(a, b, n, rng):
    """
    Generate initial permutation using eigenvalue-based heuristic.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        k = min(16, n)
        idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
        
        score_a = np.zeros(n)
        for idx in idx_a:
            score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
        
        score_b = np.zeros(n)
        for idx in idx_b:
            score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
        
        fac_order = np.argsort(score_a)[::-1]
        loc_order = np.argsort(score_b)
        
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        p_rev = np.zeros(n, dtype=int)
        for i in range(n):
            p_rev[fac_order[i]] = loc_order[n - 1 - i]
        
        cost1 = np.sum(a * b[np.ix_(p, p)])
        cost2 = np.sum(a * b[np.ix_(p_rev, p_rev)])
        
        if cost1 <= cost2:
            return p
        else:
            return p_rev
            
    except:
        return rng.permutation(n)

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

# Removed VNS perturbations as we are switching to SA

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for safety
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse initial solutions
    # 1. Hungarian initialization (strong structural start)
    p_hung = _hungarian_init(a, b)
    
    # 2. Eigenvalue-based initialization
    p_eigen = _eigen_init(a, b, n, rng)
    
    # 3. Random initialization
    p_random = rng.permutation(n)
    
    initial_solutions = [p_hung, p_eigen, p_random]
    
    # Run SA on each initial solution
    # Allocate time: 40% Hungarian, 30% Eigen, 30% Random
    time_allocs = [0.4, 0.3, 0.3]
    
    for idx, p_init in enumerate(initial_solutions):
        elapsed = time.time() - start_time
        remaining = time_budget - 5 - elapsed
        if remaining < 1:
            break
        
        # Proportional time allocation based on remaining time
        # Simple approach: divide remaining time by number of remaining runs, weighted
        # Or just fixed proportions of total budget, capped by remaining.
        
        # Let's use a simple strategy:
        # First run gets 40% of total budget.
        # Second gets 30%.
        # Third gets 30%.
        # Adjust if time is running out.
        
        target_time = time_budget * time_allocs[idx]
        ls_time = min(target_time, remaining)
        
        if ls_time < 1:
            continue

        p_ls, cost_ls = _sa_search(a, b, p_init, ls_time, start_time, rng)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # If we have time left, run SA again from the best solution found
    elapsed = time.time() - start_time
    remaining = time_budget - 5 - elapsed
    if remaining > 5 and best_p is not None:
        p_ls, cost_ls = _sa_search(a, b, best_p, remaining, start_time, rng)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
