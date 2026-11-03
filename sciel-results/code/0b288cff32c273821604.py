"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # 148s safety margin for 150s limit
    time_limit = 148.0 

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # Precompute necessary matrices for speed
    # For QAP, cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # We assume symmetric matrices as is standard for QAPLIB instances like tho150
    # If not symmetric, the cost definition still holds, but local search delta logic 
    # might need care. Standard QAP is symmetric.
    A = a
    B = b

    def get_initial_solution_greedy(alpha=0.5):
        """
        Construct a solution using Greedy Randomized Adaptive Search Procedure (GRASP).
        alpha controls the size of the Restricted Candidate List (RCL).
        alpha=0 is pure greedy, alpha=1 is random.
        """
        p = np.full(n, -1, dtype=np.int64)
        unassigned_facilities = list(range(n))
        assigned_locations = np.full(n, False, dtype=bool)
        
        # current_cost is the partial cost of the assignment so far
        # We track the cost contribution of already assigned facilities.
        # When we assign facility i to location j, the new cost added is:
        # sum_{k assigned} (a[i,k] * b[j, p[k]] + a[k,i] * b[p[k], j])
        # Since A and B are symmetric, this is 2 * sum_{k assigned} a[i,k] * b[j, p[k]]
        # Note: The term a[i,i]*b[j,j] is not included in the sum over k if k!=i, 
        # but since i is not assigned yet, it's not in the sum.
        # However, the total cost includes the diagonal term a[i,i]*b[p[i],p[i]].
        # It is easier to compute the delta relative to the final cost structure.
        
        # Let's use a simpler metric for the greedy step:
        # Cost of assigning i to j given current assignment P:
        # delta = 2 * sum_{k in P} a[i,k] b[j, p[k]]
        # (Ignoring diagonal for now, or adding it if we want exact partial cost)
        
        # To speed up, we can maintain a matrix of "attractiveness" or just compute on the fly.
        # n=150 is small. 150 iterations * 150 candidates * 150 assigned = 3.3M ops per construction.
        # This is acceptable for a few constructions.
        
        # We will maintain a vector `assigned_locs` of length n, where assigned_locs[k] = p[k] if k assigned, else -1.
        # Actually, we just need the list of assigned (facility, location) pairs.
        
        assigned_pairs = [] # list of (fac, loc)
        
        # For efficient delta calculation, let's precompute nothing, just use vectorized ops on the subset.
        # As we assign more, the subset grows.
        
        for _ in range(n):
            if not unassigned_facilities:
                break
            
            # Calculate cost delta for each unassigned facility i to each free location j
            # We want to pick a facility i and location j to add.
            # Standard GRASP for QAP usually picks one facility at a time.
            
            # Let's pick the facility with the highest "urgency" or just iterate through unassigned?
            # A common GRASP approach for QAP:
            # 1. Calculate the cost contribution of assigning every unassigned facility to every free location.
            # 2. Pick the best (i, j) pair? Or pick i greedily then j from RCL?
            # Let's pick the facility i that has the largest minimum possible cost improvement?
            # Simpler: Iterate through unassigned facilities, pick one at random? No, greedy.
            
            # Let's compute the cost delta for all unassigned i and all free j.
            # This is a matrix of size |U| x |F|.
            
            # To make it fast, we can vectorize the sum over assigned k.
            # Let K be the list of assigned facilities.
            # Let L be the list of their locations.
            # delta[i, j] = 2 * sum_{k in K} a[i, k] * b[j, L[k]]
            
            # This can be computed as:
            # M = a[np.ix_(unassigned, K)]  # shape |U| x |K|
            # N = b[np.ix_(free_locs, L)]   # shape |F| x |K|
            # delta_matrix = 2 * (M @ N.T)  # shape |U| x |F|
            
            # This is O(|U|*|K|*|F|) roughly via matmul.
            
            if len(assigned_pairs) == 0:
                # First assignment, cost is 0 (or diagonal)
                # Just pick a random facility and location to break symmetry, 
                # or pick the one with highest flow sum?
                # Let's pick facility with max row sum A and location with min row sum B (central)
                # Or just random to ensure diversity in GRASP?
                # Let's use a heuristic for the first one to anchor the solution.
                i = np.argmax(a.sum(axis=1))
                j = np.argmin(b.sum(axis=1))
                p[i] = j
                assigned_pairs.append((i, j))
                unassigned_facilities.remove(i)
                assigned_locations[j] = True
                continue
            
            K = np.array([fac for fac, _ in assigned_pairs])
            L = np.array([loc for _, loc in assigned_pairs])
            U = np.array(unassigned_facilities)
            F = np.array([l for l in range(n) if not assigned_locations[l]])
            
            # Compute delta matrix
            # M[i_idx, k_idx] = a[U[i_idx], K[k_idx]]
            # N[j_idx, k_idx] = b[F[j_idx], L[k_idx]]
            # delta[i_idx, j_idx] = 2 * sum_k M[i_idx, k] * N[j_idx, k]
            
            M = A[np.ix_(U, K)]
            N = B[np.ix_(F, L)]
            
            # delta = 2 * (M @ N.T)
            delta_mat = 2 * (M @ N.T)
            
            # Also consider the diagonal term?
            # The total cost is sum_{i,j} a[i,j]b[p[i],p[j]].
            # The delta formula above accounts for interactions with existing assigned nodes.
            # The term a[i,i]b[j,j] is added when i is assigned.
            # Let's add it to be precise.
            diag_term = A[U, U] * B[F, F]
            # diag_term is shape |U| x |U|? No, we need a[i,i] for each i in U and b[j,j] for each j in F.
            # a_ii = A[U, U].diagonal()
            # b_jj = B[F, F].diagonal()
            # delta_diag = a_ii[:, None] * b_jj[None, :]
            
            a_ii = A[U, U].diagonal()
            b_jj = B[F, F].diagonal()
            delta_diag = np.outer(a_ii, b_jj)
            
            total_delta = delta_mat + delta_diag
            
            # Now we have a matrix of costs to add for each (i, j) pair.
            # We want to pick one (i, j) to assign.
            # GRASP: 
            # 1. Find min and max of total_delta.
            # 2. RCL = { (i,j) | total_delta[i,j] <= min_val + alpha * (max_val - min_val) }
            # 3. Pick one (i,j) from RCL uniformly at random.
            
            min_d = np.min(total_delta)
            max_d = np.max(total_delta)
            
            if alpha <= 0:
                # Pure greedy
                idx = np.unravel_index(np.argmin(total_delta), total_delta.shape)
            elif alpha >= 1:
                # Random
                idx = (np.random.randint(0, len(U)), np.random.randint(0, len(F)))
            else:
                threshold = min_d + alpha * (max_d - min_d)
                rcl_mask = total_delta <= threshold
                rcl_indices = np.argwhere(rcl_mask)
                if len(rcl_indices) == 0:
                    idx = np.unravel_index(np.argmin(total_delta), total_delta.shape)
                else:
                    choice = np.random.randint(0, len(rcl_indices))
                    idx = tuple(rcl_indices[choice])
            
            i = U[idx[0]]
            j = F[idx[1]]
            
            p[i] = j
            assigned_pairs.append((i, j))
            unassigned_facilities.remove(i)
            assigned_locations[j] = True
            
        return p

    def compute_cost(p):
        B_p = B[np.ix_(p, p)]
        return int(np.sum(A * B_p))

    # Local Search: 2-swap
    # We will implement a fast local search using incremental cost updates.
    
    def local_search(p, time_limit, max_iters=10000):
        p = p.copy()
        cost = compute_cost(p)
        
        # Precompute c[i] = sum_j a[i,j] b[p[i], p[j]]
        # Total cost = sum_i c[i] ? No, cost = sum_{i,j} a[i,j] b[p[i], p[j]].
        # c[i] = sum_j a[i,j] b[p[i], p[j]].
        # Sum_i c[i] = Cost.
        
        # Efficient update for swap (i, j) with locations u=p[i], v=p[j]:
        # Delta = 2 * (a[i,j] * (b[u,v] - b[u,u] - b[v,v] + b[v,u])) + 
        #         sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Wait, let's derive carefully.
        # Old Cost = sum_{x,y} a[x,y] b[p[x], p[y]]
        # New Cost = sum_{x,y} a[x,y] b[p'[x], p'[y]]
        # Only terms involving i or j change.
        # Terms:
        # 1. a[i,i] b[u,u] -> a[i,i] b[v,v]
        # 2. a[j,j] b[v,v] -> a[j,j] b[u,u]
        # 3. a[i,j] b[u,v] -> a[i,j] b[v,u] (same if symmetric)
        # 4. a[j,i] b[v,u] -> a[j,i] b[u,v] (same if symmetric)
        # 5. sum_{k!=i,j} a[i,k] b[u, p[k]] -> sum_{k!=i,j} a[i,k] b[v, p[k]]
        # 6. sum_{k!=i,j} a[k,i] b[p[k], u] -> sum_{k!=i,j} a[k,i] b[p[k], v]
        # 7. sum_{k!=i,j} a[j,k] b[v, p[k]] -> sum_{k!=i,j} a[j,k] b[u, p[k]]
        # 8. sum_{k!=i,j} a[k,j] b[p[k], v] -> sum_{k!=i,j} a[k,j] b[p[k], u]
        
        # Assuming symmetry A[i,k]=A[k,i] and B[u,v]=B[v,u]:
        # Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + 
        #         sum_{k!=i,j} (a[i,k] + a[k,i]) (b[v, p[k]] - b[u, p[k]]) / 2 ? No.
        # Let's group by k.
        # For a specific k != i, j:
        # Old terms: a[i,k]b[u, p[k]] + a[k,i]b[p[k], u] + a[j,k]b[v, p[k]] + a[k,j]b[p[k], v]
        #           = 2 a[i,k] b[u, p[k]] + 2 a[j,k] b[v, p[k]]  (assuming symmetry)
        # New terms: a[i,k]b[v, p[k]] + a[k,i]b[p[k], v] + a[j,k]b[u, p[k]] + a[k,j]b[p[k], u]
        #           = 2 a[i,k] b[v, p[k]] + 2 a[j,k] b[u, p[k]]
        # Delta_k = 2 a[i,k] (b[v, p[k]] - b[u, p[k]]) + 2 a[j,k] (b[u, p[k]] - b[v, p[k]])
        #         = 2 (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
        
        # So Total Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + 
        #                  2 * sum_{k!=i,j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
        
        # We can vectorize the sum over k.
        # Let diff_a = A[:, i] - A[:, j]  (shape n)
        # Let diff_b = B[:, v][p] - B[:, u][p] (shape n)
        # Sum = dot(diff_a, diff_b)
        # But this includes k=i and k=j.
        # k=i: diff_a[i] = a[i,i] - a[i,j]. diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u].
        # k=j: diff_a[j] = a[j,i] - a[j,j]. diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v].
        
        # We need to exclude i and j from the sum.
        # Sum_excl = dot(diff_a, diff_b) - diff_a[i]*diff_b[i] - diff_a[j]*diff_b[j]
        
        # Then Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + 2 * Sum_excl
        
        # This is O(n) per swap check.
        # With n=150, 11175 pairs. 11175 * 150 = 1.7M ops per full scan.
        # In Python loop, this is slow.
        # We can batch the checks?
        # Or use a "best improvement" strategy with random sampling if full scan is too slow.
        
        # Let's try to optimize the inner loop.
        # We can precompute B_p = B[np.ix_(p, p)]? No, p changes.
        # But B[:, v] and B[:, u] are just columns of B.
        
        best_cost = cost
        best_p = p.copy()
        
        # Precompute A_col_diffs? No, A is static.
        
        # Strategy: First do a few full scans of best improvement.
        # If too slow, switch to random sampling.
        
        # Let's estimate time for one full scan in Python.
        # 11175 iterations. Each involves a few numpy ops on size-150 arrays.
        # Numpy overhead is ~1-5us. 11175 * 5us = 50ms.
        # 11175 * 10us = 100ms.
        # So 1 full scan takes ~0.1-0.2s.
        # We have 148s. We can do ~1000 full scans.
        # This is plenty for local search to converge.
        
        # However, we are doing this for MULTIPLE GRASP starts.
        # We need to balance time between construction and local search.
        
        # Let's allocate time:
        # 1. Generate 1 GRASP solution, do full local search until no improvement or time budget for this start is reached.
        # 2. If time remains, generate another GRASP solution (different alpha) and do local search.
        
        # For this function, we assume we are called with a specific p.
        # We will run local search until no improvement or time_limit is reached.
        
        no_improve_count = 0
        max_no_improve = 5 # Stop if 5 full scans yield no improvement
        
        while time.time() - start_time < time_limit:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Full scan of all pairs
            # To speed up, we can iterate i from 0 to n-1, j from i+1 to n-1
            # We can vectorize the delta calculation for all j for a fixed i?
            # Delta(i, j) depends on p[i], p[j].
            # For fixed i, u = p[i] is fixed.
            # v = p[j] varies.
            # diff_a = A[:, i] - A[:, j] varies with j.
            # diff_b = B[:, v] - B[:, u] varies with j.
            
            # This is hard to fully vectorize across j because A[:, j] and B[:, v] change.
            # We will stick to the loop but optimize the inner part.
            
            for i in range(n):
                u = p[i]
                # Precompute parts that don't depend on j?
                # a_i = A[i, :]
                # b_u = B[u, :]
                
                for j in range(i + 1, n):
                    v = p[j]
                    
                    # diff_a = A[:, i] - A[:, j]
                    # diff_b = B[:, v][p] - B[:, u][p]
                    
                    # Optimization: 
                    # Sum = dot(A[:, i] - A[:, j], B[v, p] - B[u, p])
                    # = dot(A[:, i], B[v, p]) - dot(A[:, i], B[u, p]) - dot(A[:, j], B[v, p]) + dot(A[:, j], B[u, p])
                    
                    # Let's compute these 4 dots.
                    # Note: B[v, p] is just B[v, p].
                    
                    # To reduce numpy calls, we can do:
                    # av = A[:, i]
                    # aj = A[:, j]
                    # bv = B[v, p]
                    # bu = B[u, p]
                    
                    # dot_av_bv = np.dot(av, bv)
                    # dot_av_bu = np.dot(av, bu)
                    # dot_aj_bv = np.dot(aj, bv)
                    # dot_aj_bu = np.dot(aj, bu)
                    
                    # Sum = dot_av_bv - dot_av_bu - dot_aj_bv + dot_aj_bu
                    
                    # This is 4 dots of size n.
                    # 4 * 150 = 600 ops.
                    # 11175 * 600 = 6.7M ops.
                    # In numpy, this is very fast.
                    
                    av = A[:, i]
                    aj = A[:, j]
                    bv = B[v, p]
                    bu = B[u, p]
                    
                    sum_val = np.dot(av, bv) - np.dot(av, bu) - np.dot(aj, bv) + np.dot(aj, bu)
                    
                    # Exclude i and j
                    # term_i = (A[i, i] - A[i, j]) * (B[v, u] - B[u, u])
                    # term_j = (A[j, i] - A[j, j]) * (B[v, v] - B[u, v])
                    
                    term_i = (A[i, i] - A[i, j]) * (B[v, u] - B[u, u])
                    term_j = (A[j, i] - A[j, j]) * (B[v, v] - B[u, v])
                    
                    sum_excl = sum_val - term_i - term_j
                    
                    delta = A[i, i] * (B[v, v] - B[u, u]) + A[j, j] * (B[u, u] - B[v, v]) + 2 * sum_excl
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                # Apply swap
                i, j = best_i, best_j
                u, v = p[i], p[j]
                p[i], p[j] = v, u
                cost += int(best_delta)
                improved = True
                no_improve_count = 0
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
        return best_p, best_cost

    # Main GRASP Loop
    best_p_overall = None
    best_cost_overall = np.inf
    
    # Try a few different alpha values for diversity
    # alpha=0.1 (mostly greedy), alpha=0.3, alpha=0.5
    alphas = [0.1, 0.3, 0.5]
    
    # Allocate time: e.g., 40% for first start, 30% for second, 30% for third?
    # Or just run until time limit, prioritizing the first one which is likely best.
    # Let's give the first one a generous budget.
    
    # Time management
    # Total 148s.
    # Construction is fast (<1s).
    # Local search is the main consumer.
    
    for alpha in alphas:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Construct
        p_init = get_initial_solution_greedy(alpha)
        
        # Local Search
        # Give it a time limit relative to remaining time
        # If it's the first one, give it more.
        remaining = time_limit - (time.time() - start_time)
        # Reserve 5s for overhead/next starts
        ls_time_limit = remaining - 5 if len(alphas) > 1 else remaining - 2
        
        if ls_time_limit < 5:
            ls_time_limit = 5
            
        p_ls, cost_ls = local_search(p_init, ls_time_limit)
        
        if cost_ls < best_cost_overall:
            best_cost_overall = cost_ls
            best_p_overall = p_ls

    if best_p_overall is None:
        # Fallback
        p_init = get_initial_solution_greedy(0.1)
        best_p_overall, best_cost_overall = local_search(p_init, time_limit - (time.time() - start_time))

    return best_p_overall.tolist()

# EVOLVE-BLOCK-END
