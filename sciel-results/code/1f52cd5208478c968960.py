"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def local_search(p, current_cost, deadline):
        """
        Fast local search with O(n^2) incremental update of interaction vectors.
        Maintains dot_row[i] = sum_k a[i,k]*b[p[i],p[k]] and dot_col[i] = sum_k a[k,i]*b[p[k],p[i]]
        """
        # Precompute interaction vectors for O(n^2) delta evaluation
        # dot_row[i] = sum_k a[i,k] * b[p[i], p[k]]
        # dot_col[i] = sum_k a[k,i] * b[p[k], p[i]]
        
        dot_row = np.zeros(n, dtype=np.float64)
        dot_col = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            dot_row[i] = np.dot(a[i, :], b[p[i], p])
            dot_col[i] = np.dot(a[:, i], b[p, p[i]])
        
        improved = True
        max_iter = 5000
        iter_count = 0
        time_check_interval = 100
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if iter_count % time_check_interval == 0 and time.time() >= deadline - 2.0:
                    return p, current_cost
                
                ai = a[i, :]
                ai_col = a[:, i]
                pi = p[i]
                
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    
                    # Delta for swapping i and j:
                    # = (a[i,j]-a[j,i])*(b[v,u]-b[u,v]) 
                    #   + sum_{k!=i,j} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
                    #   + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
                    
                    # Using precomputed dot vectors:
                    # sum_k (a[i,k]-a[j,k])*b[v,p[k]] = dot_row_i_at_v - dot_row_j_at_v
                    # where dot_row_i_at_v = sum_k a[i,k]*b[v,p[k]]
                    
                    # We can compute these incrementally. But for now, let's use the
                    # fact that we can express the delta in terms of the current dot vectors.
                    
                    # Actually, let's use the standard formula with precomputed contributions:
                    # The change in cost when swapping locations u and v for facilities i and j is:
                    # delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
                    #       + sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
                    #       + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
                    
                    # Let's precompute for each pair (i,j) the needed sums using the dot vectors.
                    # 
                    # Note: dot_row[i] = sum_k a[i,k]*b[p[i],p[k]]
                    # If we swap p[i] and p[j], the new dot_row[i] would be sum_k a[i,k]*b[v,p[k]]
                    # (with p[k] unchanged for k!=i,j, but p[i] becomes v and p[j] becomes u)
                    
                    # This is getting complex. Let me use a simpler O(n) per pair approach
                    # that leverages the structure better.
                    
                    # Alternative: maintain for each location l, the contribution vector.
                    # Let's just use the vectorized computation but cache b[v,p] and b[u,p].
                    
                    # For efficiency, precompute b[v,p] and b[u,p] once per j iteration? No, v depends on j.
                    
                    # Let's use the formula:
                    # s1 = sum_k (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
                    # s2 = sum_k (a[k,i] - a[k,j]) * (b[p[k],v] - b[p[k],u])
                    
                    # We can rewrite:
                    # s1 = sum_k a[i,k]*b[v,p[k]] - sum_k a[i,k]*b[u,p[k]] - sum_k a[j,k]*b[v,p[k]] + sum_k a[j,k]*b[u,p[k]]
                    #    = (dot_row_i_if_at_v) - (dot_row_i_if_at_u) - (dot_row_j_if_at_v) + (dot_row_j_if_at_u)
                    
                    # Where dot_row_i_if_at_v = sum_k a[i,k]*b[v,p[k]]
                    
                    # This is still O(n) per term. Let me just do the direct computation
                    # but with better vectorization.
                    
                    diff_a_rows = ai - a[j, :]
                    diff_b_rows = b[v, p] - b[u, p]
                    s1 = np.dot(diff_a_rows, diff_b_rows)
                    
                    diff_a_cols = ai_col - a[:, j]
                    diff_b_cols = b[p, v] - b[p, u]
                    s2 = np.dot(diff_a_cols, diff_b_cols)
                    
                    # Corrections for i,j terms
                    ai_i = a[i, i]
                    aj_i = a[j, i]
                    ai_j = a[i, j]
                    aj_j = a[j, j]
                    bu_u = b[u, u]
                    bu_v = b[u, v]
                    bv_u = b[v, u]
                    bv_v = b[v, v]
                    
                    term_ii_s1 = (ai_i - aj_i) * (bv_u - bu_u)
                    term_ii_s2 = (ai_i - aj_i) * (bu_v - bu_u)
                    term_jj_s1 = (ai_j - aj_j) * (bv_v - bu_v)
                    term_jj_s2 = (ai_j - aj_j) * (bu_v - bv_v)
                    interaction_delta = (ai_j - aj_i) * (bv_u - bu_v)
                    
                    d = s1 + s2 - term_ii_s1 - term_ii_s2 - term_jj_s1 - term_jj_s2 + interaction_delta
                    
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                # Apply the swap
                bi = best_i
                bj = best_j
                u = p[bi]
                v = p[bj]
                
                p[bi], p[bj] = v, u
                current_cost += best_delta
                improved = True
                iter_count += 1
                
                # Update dot_row and dot_col incrementally
                # For all k != bi, bj:
                # dot_row[k] changes because p[bi] and p[bj] changed
                # dot_row[k] = sum_l a[k,l]*b[p[k],p[l]]
                # The terms involving l=bi and l=bj change.
                
                # New dot_row[k] = old_dot_row[k] 
                #   - a[k,bi]*b[p[k],u] - a[k,bj]*b[p[k],v]  (remove old contributions)
                #   + a[k,bi]*b[p[k],v] + a[k,bj]*b[p[k],u]  (add new contributions)
                
                # For k = bi:
                # dot_row[bi] = sum_l a[bi,l]*b[v,p[l]] (since p[bi] is now v)
                # We need to recompute this.
                
                # For k = bj:
                # dot_row[bj] = sum_l a[bj,l]*b[u,p[l]] (since p[bj] is now u)
                
                # Similarly for dot_col.
                
                # This incremental update is O(n) per swap, which is much faster than O(n^2) recompute.
                
                # Let's implement the O(n) update.
                
                # First, update dot_row for all k
                # For k not in {bi, bj}:
                # dot_row[k] += a[k,bi]*(b[p[k],v]-b[p[k],u]) + a[k,bj]*(b[p[k],u]-b[p[k],v])
                #             = (a[k,bi]-a[k,bj])*(b[p[k],v]-b[p[k],u])
                
                mask = np.ones(n, dtype=bool)
                mask[bi] = False
                mask[bj] = False
                ks = np.where(mask)[0]
                
                if len(ks) > 0:
                    b_p_k_v = b[p[ks], v]
                    b_p_k_u = b[p[ks], u]
                    diff_a = a[ks, bi] - a[ks, bj]
                    dot_row[ks] += diff_a * (b_p_k_v - b_p_k_u)
                
                # For k = bi: p[bi] is now v
                # dot_row[bi] = sum_l a[bi,l]*b[v,p[l]]
                # p[l] for l=bi is v, for l=bj is u, for others unchanged
                dot_row[bi] = np.dot(a[bi, :], b[v, p])
                
                # For k = bj: p[bj] is now u
                dot_row[bj] = np.dot(a[bj, :], b[u, p])
                
                # Similarly for dot_col
                # dot_col[k] = sum_l a[l,k]*b[p[l],p[k]]
                
                # For k not in {bi, bj}:
                # dot_col[k] += (a[bi,k]-a[bj,k])*(b[v,p[k]]-b[u,p[k]])
                
                if len(ks) > 0:
                    b_v_pk = b[v, p[ks]]
                    b_u_pk = b[u, p[ks]]
                    diff_a = a[bi, ks] - a[bj, ks]
                    dot_col[ks] += diff_a * (b_v_pk - b_u_pk)
                
                # For k = bi: p[bi] is now v
                dot_col[bi] = np.dot(a[:, bi], b[p, v])
                
                # For k = bj: p[bj] is now u
                dot_col[bj] = np.dot(a[:, bj], b[p, u])
                
            else:
                break
                
        return p, current_cost

    def greedy_init():
        """
        Greedy construction: assign facilities in order of flow sum 
        to the location that minimizes the incremental cost.
        """
        p = np.full(n, -1, dtype=np.int64)
        flow_sums = a.sum(axis=1)
        order = np.argsort(-flow_sums, kind='stable')
        
        assigned = np.zeros(n, dtype=bool)
        assigned_count = 0
        
        for idx in order:
            best_loc = -1
            best_inc = np.inf
            
            inter_vec = a[idx, :] + a[:, idx]
            
            if assigned_count == 0:
                p[idx] = 0  # Arbitrary first placement
                assigned[0] = True
                assigned_count += 1
                continue
            
            assigned_indices = np.where(assigned)[0]
            b_loc_p_assigned = b[:, p[assigned_indices]]  # shape (n, assigned_count)
            inter_vec_assigned = inter_vec[assigned_indices]  # shape (assigned_count,)
            
            # For each unassigned location, compute inc = sum_j inter_vec[j] * b[loc, p[j]]
            # = inter_vec_assigned @ b_loc_p_assigned[loc, :]
            
            unassigned_locs = np.where(~assigned)[0]
            if len(unassigned_locs) == 0:
                break
            
            # Vectorized: compute all incremental costs at once
            # b_loc_p_assigned has shape (n, assigned_count)
            # For each unassigned loc, inc = np.dot(b_loc_p_assigned[loc, :], inter_vec_assigned)
            incs = b_loc_p_assigned[unassigned_locs] @ inter_vec_assigned
            
            best_idx_in_unassigned = np.argmin(incs)
            best_loc = unassigned_locs[best_idx_in_unassigned]
            best_inc = incs[best_idx_in_unassigned]
            
            p[idx] = best_loc
            assigned[best_loc] = True
            assigned_count += 1
            
        return p

    # 1. Initialization
    best_cost = np.inf
    best_p = None
    
    # Generate diverse initial solutions
    inits = []
    
    # Greedy Init
    inits.append(greedy_init())
    
    # Rank Init (Forward)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    inits.append(p_rank)
    
    # Rank Init (Reversed)
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    inits.append(p_rank_rev)
    
    # Random Init
    rng = np.random.RandomState(42)
    inits.append(rng.permutation(n))
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 10.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)
    
    # 2. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    no_improve_count = 0
    rng = np.random.RandomState(123)
    
    while time.time() < deadline - 5.0:
        # Perturb the best solution
        p = best_p.copy()
        
        # Adaptive perturbation
        base_swaps = 3
        extra_swaps = min(no_improve_count, 10)
        num_swaps = base_swaps + extra_swaps
        num_swaps = min(num_swaps, 15)
        
        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 20:
            no_improve_count = 0

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
