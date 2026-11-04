"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    n = len(p)
    # Efficient: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = trace(a @ B) where B[i,j] = b[p[i], p[j]]
    # But we can do: for each i, contribution = sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[:, i] @ b[p[i], :])  -- wait, let me think more carefully.
    # Actually: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's use: for each pair (i,j), a[i,j] * b[p[i], p[j]]
    # We can vectorize: create B where B[i,j] = b[p[i], p[j]]
    # Then cost = np.sum(a * B)
    # But building B explicitly is O(n^2) memory which is fine for n=150.
    B = b[np.ix_(p, p)]
    return int(np.sum(a * B))


def _compute_cost_incremental(a, b, p, i, j):
    """Compute cost change when swapping positions i and j in permutation p.
    p is a numpy array. Returns the delta in cost.
    """
    # Before: facility at position i is p[i], facility at position j is p[j]
    # After swap: facility at position i is p[j], facility at position j is p[i]
    # 
    # Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # 
    # The terms that change involve u=i or v=i or u=j or v=j.
    # Let's compute the old contribution of all terms involving i or j,
    # and the new contribution, then take the difference.
    
    pi = p[i]
    pj = p[j]
    
    # Old cost terms involving i or j:
    # Terms where u=i: sum_v a[i,v] * b[pi, p[v]]
    # Terms where v=i: sum_u a[u,i] * b[p[u], pi]
    # Terms where u=j: sum_v a[j,v] * b[pj, p[v]]
    # Terms where v=j: sum_u a[u,j] * b[p[u], pj]
    # But term (i,i), (i,j), (j,i), (j,j) are counted in multiple.
    
    # Let's just compute the full cost difference by considering all affected terms.
    # Affected terms are those where u in {i,j} or v in {i,j}.
    
    # Old contributions:
    # For u=i: sum_v a[i,v] * b[pi, p[v]]
    old_i = np.dot(a[i, :], b[pi, p])
    # For v=i: sum_u a[u,i] * b[p[u], pi]
    old_i_col = np.dot(a[:, i], b[p, pi])
    # For u=j: sum_v a[j,v] * b[pj, p[v]]
    old_j = np.dot(a[j, :], b[pj, p])
    # For v=j: sum_u a[u,j] * b[p[u], pj]
    old_j_col = np.dot(a[:, j], b[p, pj])
    
    # The term (i,i) is counted in old_i (v=i) and old_i_col (u=i), so it's double counted.
    # Similarly (i,j) in old_i (v=j) and old_j_col (u=i)
    # (j,i) in old_j (v=i) and old_i_col (u=j)
    # (j,j) in old_j (v=j) and old_j_col (u=j)
    
    # Actually, let me just compute the total old and new cost for affected terms more carefully.
    # Total cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # When we swap p[i] and p[j], only terms where u or v is i or j change.
    
    # Let's compute:
    # old_affected = sum_{u,v: u in {i,j} or v in {i,j}} a[u,v] * b[p[u], p[v]]
    # new_affected = same but with p[i]<->p[j]
    
    # To avoid double counting, let's list all unique (u,v) pairs where u in {i,j} or v in {i,j}:
    # These are: (i,v) for all v, (j,v) for all v, (u,i) for u != i,j, (u,j) for u != i,j
    # But (i,i), (i,j), (j,i), (j,j) are in both (i,v) and (j,v) sets.
    
    # Simpler: just compute the difference directly.
    # The change in cost = new_cost - old_cost
    # = sum_{u,v} a[u,v] * (b[p_new[u], p_new[v]] - b[p[u], p[v]])
    # where p_new differs from p only at positions i and j.
    
    # So the difference is non-zero only when u or v is i or j.
    
    # Let's compute:
    # For u=i: sum_v a[i,v] * (b[pj, p_new[v]] - b[pi, p[v]])
    #   For v not i,j: p_new[v] = p[v], so term = a[i,v] * (b[pj, p[v]] - b[pi, p[v]])
    #   For v=i: p_new[i]=pj, so term = a[i,i] * (b[pj, pj] - b[pi, pi])
    #   For v=j: p_new[j]=pi, so term = a[i,j] * (b[pj, pi] - b[pi, pj])
    
    # For u=j: sum_v a[j,v] * (b[pi, p_new[v]] - b[pj, p[v]])
    #   For v not i,j: term = a[j,v] * (b[pi, p[v]] - b[pj, p[v]])
    #   For v=i: term = a[j,i] * (b[pi, pj] - b[pj, pi])
    #   For v=j: term = a[j,j] * (b[pi, pi] - b[pj, pj])
    
    # For u not i,j, v=i: a[u,i] * (b[p_new[u], pj] - b[p[u], pi]) = a[u,i] * (b[p[u], pj] - b[p[u], pi])
    # For u not i,j, v=j: a[u,j] * (b[p_new[u], pi] - b[p[u], pj]) = a[u,j] * (b[p[u], pi] - b[p[u], pj])
    
    # This is getting complex. Let's just do it with arrays for n=150, it's fast enough.
    
    # Create a copy of p with swap
    p_new = p.copy()
    p_new[i], p_new[j] = p_new[j], p_new[i]
    
    # Compute cost difference
    # old_cost_terms = sum over all (u,v) of a[u,v]*b[p[u],p[v]]
    # new_cost_terms = sum over all (u,v) of a[u,v]*b[p_new[u],p_new[v]]
    # But we only need the difference, and only terms where p[u] != p_new[u] or p[v] != p_new[v] contribute.
    # That's when u in {i,j} or v in {i,j}.
    
    # Let's compute using the fact that for u,v not in {i,j}, p[u]=p_new[u] and p[v]=p_new[v].
    
    # delta = sum_{u,v} a[u,v] * (b[p_new[u], p_new[v]] - b[p[u], p[v]])
    # = sum_{u in {i,j}} sum_v a[u,v] * (b[p_new[u], p_new[v]] - b[p[u], p[v]])
    #   + sum_{u not in {i,j}} sum_{v in {i,j}} a[u,v] * (b[p_new[u], p_new[v]] - b[p[u], p[v]])
    
    # First part: u in {i,j}
    part1 = 0
    for u in [i, j]:
        pu = p[u]
        pun = p_new[u]
        for v in range(len(p)):
            pv = p[v]
            pvn = p_new[v]
            part1 += a[u, v] * (b[pun, pvn] - b[pu, pv])
    
    # Second part: u not in {i,j}, v in {i,j}
    part2 = 0
    for v in [i, j]:
        pv = p[v]
        pvn = p_new[v]
        for u in range(len(p)):
            if u == i or u == j:
                continue
            pu = p[u]
            pun = p_new[u]  # same as pu since u not i,j
            part2 += a[u, v] * (b[pun, pvn] - b[pu, pv])
    
    return part1 + part2


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140  # 10s safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Tabu search with 2-opt swaps
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    tabu_tenure = 10
    iteration = 0
    
    # Precompute for faster swap evaluation
    # For a swap of positions i and j, we need:
    # The change in cost depends on a[i,:], a[:,i], a[j,:], a[:,j] and b rows/cols
    
    while time.time() < deadline:
        # Find best non-tabu swap
        best_delta = 0
        best_i, best_j = -1, -1
        
        # To speed up, we can sample a subset of swaps
        # For n=150, total swaps = 150*149/2 = 11175, which is manageable
        # Let's evaluate all swaps but use vectorized computation
        
        # Actually, let's use a smarter approach:
        # For each position i, compute the "attraction" to each other position j
        # This allows us to find the best swap more efficiently.
        
        # The cost change for swapping i and j:
        # Let's derive it more carefully.
        # 
        # Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
        # 
        # When we swap p[i] and p[j]:
        # Let x = p[i], y = p[j]
        # 
        # The terms that change are those where u or v is i or j.
        # 
        # Let's define:
        # S_i = sum_v a[i,v] * b[x, p[v]]  (row i contribution)
        # S_j = sum_v a[j,v] * b[y, p[v]]  (row j contribution)
        # T_i = sum_u a[u,i] * b[p[u], x]  (col i contribution)
        # T_j = sum_u a[u,j] * b[p[u], y]  (col j contribution)
        # 
        # But note that S_i includes v=i and v=j terms, and T_i includes u=i and u=j terms.
        # The term (i,i) appears in both S_i (v=i) and T_i (u=i).
        # The term (i,j) appears in S_i (v=j) and T_j (u=i).
        # Etc.
        
        # This is getting complicated. Let's just use the direct method but optimize it.
        # For n=150, evaluating all ~11175 swaps per iteration with O(n) each = ~1.7M ops per iteration.
        # That should be fast enough in numpy if we vectorize.
        
        # Vectorized swap evaluation:
        # For swap (i,j), the cost change is:
        # delta = sum_{v} a[i,v]*(b[pj,p_new[v]] - b[pi,p[v]]) + sum_{v} a[j,v]*(b[pi,p_new[v]] - b[pj,p[v]])
        #       + sum_{u != i,j} a[u,i]*(b[p[u],pj] - b[p[u],pi]) + sum_{u != i,j} a[u,j]*(b[p[u],pi] - b[p[u],pj])
        
        # Let's precompute some things:
        # For each position i, and each facility f, we have b[f, p[i]] and b[p[i], f]
        
        # Actually, let's just do a simple loop but break early if time is tight.
        
        # For efficiency, let's sample a random subset of swaps if full evaluation is too slow
        sample_size = min(2000, n * (n - 1) // 2)
        
        if sample_size < n * (n - 1) // 2:
            # Sample random pairs
            pairs = []
            for _ in range(sample_size):
                i, j = np.random.choice(n, 2, replace=False)
                if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                    continue
                if (j, i) in tabu_list and tabu_list[(j, i)] > iteration:
                    continue
                pairs.append((i, j))
        else:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            pairs = [(i, j) for (i, j) in pairs if not ((i, j) in tabu_list and tabu_list[(i, j)] > iteration) and not ((j, i) in tabu_list and tabu_list[(j, i)] > iteration)]
        
        for i, j in pairs:
            if time.time() > deadline - 5:
                break
            
            # Compute cost change for swap (i, j)
            pi = p[i]
            pj = p[j]
            
            # Vectorized computation of delta
            # Terms involving row i: sum_v a[i,v] * (b[pj, p_new[v]] - b[pi, p[v]])
            # For v != i,j: p_new[v] = p[v]
            # For v = i: p_new[i] = pj
            # For v = j: p_new[j] = pi
            
            # Let's compute using numpy
            # b[pj, p] - b[pi, p] gives the change for each v (if p[v] doesn't change)
            # But for v=i and v=j, p[v] changes.
            
            # Simpler: just compute the full affected cost
            # old_affected = a[i, :] @ b[pi, p] + a[j, :] @ b[pj, p] + a[:, i] @ b[p, pi] + a[:, j] @ b[p, pj]
            # But this double counts (i,i), (i,j), (j,i), (j,j)
            
            # Let me just compute it directly with the swap
            p_new = p.copy()
            p_new[i], p_new[j] = p_new[j], p_new[i]
            
            # Compute the difference in cost
            # Only terms where u or v is i or j change
            # delta = sum_{u,v: u in {i,j} or v in {i,j}} a[u,v] * (b[p_new[u], p_new[v]] - b[p[u], p[v]])
            
            # For u in {i,j}, all v:
            delta = 0
            for u in (i, j):
                pu = p[u]
                pun = p_new[u]
                # sum_v a[u,v] * (b[pun, p_new[v]] - b[pu, p[v]])
                # p_new[v] = p[v] for v not i,j; p_new[i]=pj, p_new[j]=pi
                # So b[pun, p_new[v]] = b[pun, p[v]] for v not i,j
                # and b[pun, p_new[i]] = b[pun, pj] for v=i
                # and b[pun, p_new[j]] = b[pun, pi] for v=j
                
                # Let's compute:
                # sum_{v not i,j} a[u,v] * (b[pun, p[v]] - b[pu, p[v]])
                # + a[u,i] * (b[pun, p_new[i]] - b[pu, p[i]])
                # + a[u,j] * (b[pun, p_new[j]] - b[pu, p[j]])
                
                # For v not i,j:
                # b[pun, p[v]] - b[pu, p[v]] = b[pun, p[v]] - b[pu, p[v]]
                # We can compute this as: for each v not i,j, a[u,v] * (b[pun, p[v]] - b[pu, p[v]])
                
                # This is still O(n). Let's just do it.
                for v in range(n):
                    if v == i:
                        pvn = pj  # p_new[i]
                    elif v == j:
                        pvn = pi  # p_new[j]
                    else:
                        pvn = p[v]
                    pv = p[v]
                    delta += a[u, v] * (b[pun, pvn] - b[pu, pv])
            
            # For u not in {i,j}, v in {i,j}:
            for v in (i, j):
                pv = p[v]
                pvn = p_new[v]
                for u in range(n):
                    if u == i or u == j:
                        continue
                    pu = p[u]
                    delta += a[u, v] * (b[pu, pvn] - b[pu, pv])
            
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j
        
        # Apply best swap if it improves
        if best_i >= 0 and best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += best_delta
            if best_cost < _compute_cost(a, b, best_p):
                # Update best (this check is expensive, so do it less often)
                pass
            
            # Add to tabu
            tabu_list[(best_i, best_j)] = iteration + tabu_tenure
            tabu_list[(best_j, best_i)] = iteration + tabu_tenure
        
        iteration += 1
        
        # Periodically update best
        if iteration % 50 == 0:
            current_cost = _compute_cost(a, b, p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Clean up tabu list
        if iteration % 100 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
    
    # Final check
    final_cost = _compute_cost(a, b, p)
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
