import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin
    
    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback[:]

    # Precompute edge lookups for various n
    # We will try n from 12 up to 26 (or as high as time allows)
    # The known bound is 26, so we aim for 12-20 first, then higher if fast.
    
    def get_edge_lookup(n):
        lookup = []
        for i in range(n):
            for j in range(i + 1, n):
                lookup.append((i, j))
        return lookup

    def solve_n(n, time_limit):
        """
        Attempt to find a valid 5-coloring of K_n.
        Returns (colors, success)
        """
        num_edges = n * (n - 1) // 2
        edge_lookup = get_edge_lookup(n)
        
        # Data structures for O(n) updates
        # adj[c] is a list of sets of neighbors for each vertex? 
        # No, for speed, use numpy arrays for adjacency but maintain explicit common-neighbor counts.
        
        # Let's use a flat array for colors
        colors = np.random.randint(0, 5, size=num_edges, dtype=np.int8)
        
        # Build adjacency matrices A_c (n x n, symmetric, 0 diag)
        # And common neighbor counts M_c[i,j] = number of common neighbors of i,j in color c
        # M_c = A_c @ A_c
        
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        # Compute initial M and cost
        # Cost = sum over all c, sum over all i<j, (M_c[i,j] choose 2)
        # Note: M_c[i,j] includes i=j (degree), but we only care about i != j for C4s.
        # Actually, a C4 is i-k-j-l-i. This corresponds to common neighbors of i and j.
        # If i and j have >= 2 common neighbors in color c, that's a C4.
        
        common_mats = [m @ m for m in mats]
        
        def calc_cost(common_mats):
            total = 0.0
            for M in common_mats:
                # We want sum_{i < j} binom(M[i,j], 2)
                # M is symmetric.
                # binom(x, 2) = x(x-1)/2
                # We can compute sum_{i,j} M[i,j]^2 - sum_{i,j} M[i,j]
                # But we need to exclude diagonal?
                # Diagonal M[i,i] is degree. Common neighbors of i and i is degree.
                # Does a "C4" involve a vertex connected to itself? No.
                # So we only consider i != j.
                # Sum_{i != j} binom(M[i,j], 2)
                # Let S = sum_{i,j} binom(M[i,j], 2)
                # Diag part = sum_i binom(M[i,i], 2)
                # Off-diag part = S - Diag
                # Since M is symmetric, off-diag entries appear twice in S (i,j and j,i).
                # So Sum_{i < j} binom(M[i,j], 2) = (S - Diag) / 2
                # Where S = sum_{i,j} M[i,j](M[i,j]-1)/2
                # Let's just use numpy for the whole thing.
                M_sq = M * M
                M_sum = M.sum()
                M_sq_sum = M_sq.sum()
                # Sum_{i,j} M[i,j]^2 - Sum_{i,j} M[i,j] = 2 * Sum_{i,j} binom(M[i,j], 2)
                # So Sum_{i,j} binom(M[i,j], 2) = (M_sq_sum - M_sum) / 2
                # We need to subtract diagonal contributions.
                diag_M = np.diag(M)
                diag_M_sq = diag_M * diag_M
                diag_sum = diag_M.sum()
                diag_binom_sum = (diag_M_sq - diag_sum) / 2.0
                total_binom_sum = (M_sq_sum - M_sum) / 2.0
                off_diag_binom_sum = total_binom_sum - diag_binom_sum
                # This off_diag_binom_sum is sum_{i != j} binom(M[i,j], 2)
                # Since M is symmetric, this is 2 * sum_{i < j} binom(M[i,j], 2)
                c4_count = off_diag_binom_sum / 2.0
                total += c4_count
            return total

        current_cost = calc_cost(common_mats)
        
        if current_cost == 0:
            return colors.tolist(), True

        # Simulated Annealing
        # Temperature schedule
        # Start high, cool down.
        # We need to accept bad moves to escape local minima.
        
        # Precompute edge index to (u, v) for fast access
        # We already have edge_lookup.
        
        # To make updates fast, we maintain:
        # 1. colors array
        # 2. mats (adjacency)
        # 3. common_mats (M = A @ A)
        
        # Update logic for changing edge (u, v) from color c_old to c_new:
        # 1. Update mats[c_old][u,v] = 0, mats[c_old][v,u] = 0
        # 2. Update mats[c_new][u,v] = 1, mats[c_new][v,u] = 1
        # 3. Update common_mats[c_old] and common_mats[c_new]
        #    Delta M_c[i,j] when adding edge (u,v) to A_c:
        #    M_c[i,j] += A_c[i,u] * A_c[u,j] + A_c[i,v] * A_c[v,j]  <-- Wait, this is for M = A@A.
        #    Actually, M[i,j] = sum_k A[i,k]A[k,j].
        #    If we add A[u,v]=1 and A[v,u]=1:
        #    New M[i,j] = Old M[i,j] + A[i,u]A[u,j] + A[i,v]A[v,j]
        #    Note: A[u,j] is 1 only if j==v (since we just added u-v, and u is only connected to v in this delta? No, u might be connected to others).
        #    Wait, A[u,j] is the adjacency matrix value. It doesn't change for j != v.
        #    So:
        #    Delta M[i,j] = A[i,u] * A[u,j] + A[i,v] * A[v,j]
        #    This is not just a rank 1 update. It's a sum of two rank 1 updates?
        #    Let's check:
        #    A_new = A_old + E_uv where E_uv has 1 at (u,v) and (v,u).
        #    M_new = A_new^2 = (A_old + E)^2 = A_old^2 + A_old E + E A_old + E^2
        #    E^2 has 1 at (u,u) and (v,v) if u-v edge exists? No, E is symmetric 0-1.
        #    E^2[u,u] = E[u,v]E[v,u] = 1.
        #    E^2[v,v] = 1.
        #    E^2[u,v] = 0.
        #    So M_new = M_old + A_old E + E A_old + E^2.
        #    This requires O(n^2) to update M if we do it via matrix mult.
        #    But we can do it via vector operations.
        #    (A_old E)[i,j] = sum_k A_old[i,k] E[k,j].
        #    E[k,j] is non-zero only if (k,j) is (u,v) or (v,u).
        #    So (A_old E)[i,v] = A_old[i,u].
        #    (A_old E)[i,u] = A_old[i,v].
        #    (A_old E)[i,j] = 0 otherwise.
        #    Similarly for E A_old.
        #    So Delta M[i,j] = A_old[i,u] (if j==v) + A_old[i,v] (if j==u) + A_old[j,u] (if i==v) + A_old[j,v] (if i==u) + (E^2 terms)
        #    Wait, E A_old is the transpose of A_old E since A is symmetric.
        #    So Delta M is symmetric.
        #    Delta M[i,j] = 
        #      if i==u, j==v: A[u,v] (old, which is 0) + A[v,u] (old 0) + ... 
        #      Let's just use the property:
        #      M_new[i,j] = M_old[i,j] + A_old[i,u] * (j==v) + A_old[i,v] * (j==u) + A_old[j,u] * (i==v) + A_old[j,v] * (i==u)
        #      Plus the E^2 terms: +1 if i==u and j==u, +1 if i==v and j==v.
        #      Let's verify for i=u, j=v:
        #      M_new[u,v] = M_old[u,v] + A[u,u](0) + A[u,v](0) + A[v,u](0) + A[v,v](0) + 0 (E^2[u,v]=0)
        #      Wait, M[u,v] is the number of common neighbors.
        #      If we add edge (u,v), do u and v have a new common neighbor?
        #      A common neighbor k must be connected to both u and v.
        #      Previously, k was connected to u and v.
        #      Adding the edge (u,v) itself doesn't make u a neighbor of u.
        #      So M[u,v] doesn't change?
        #      Let's check formula:
        #      M_new[u,v] = sum_k A_new[u,k] A_new[k,v]
        #      A_new[u,v] = 1. Term k=v: A_new[u,v]A_new[v,v] = 1*0 = 0.
        #      Term k=u: A_new[u,u]A_new[u,v] = 0*1 = 0.
        #      Other k: A_old[u,k]A_old[k,v].
        #      So M_new[u,v] = M_old[u,v]. Correct.
        
        #      What about M[u,u]? (Degree)
        #      M_new[u,u] = sum_k A_new[u,k]^2 = sum_k A_new[u,k]
        #      = deg_old(u) + 1 (since we added edge to v).
        #      Formula:
        #      M_new[u,u] = M_old[u,u] + A[u,u](0) + A[u,v](0) + A[u,u](0) + A[u,v](0) + 1 (E^2[u,u])
        #      = M_old[u,u] + 1. Correct.
        
        #      What about M[x,u] for x not u,v?
        #      M_new[x,u] = sum_k A_new[x,k] A_new[k,u]
        #      = sum_k A_old[x,k] A_old[k,u] + A_new[x,v]A_new[v,u] + A_new[x,u]A_new[u,u]
        #      = M_old[x,u] + A_old[x,v] * 1 + A_old[x,u] * 0
        #      = M_old[x,u] + A_old[x,v]
        #      Formula:
        #      M_new[x,u] = M_old[x,u] + A[x,u](0 if j!=v) + A[x,v](1 if j==u? No, j=u, so second term A[x,v]*(j==u) is active) 
        #      My formula: Delta M[x,u] = A[x,u]*(u==v? No) + A[x,v]*(u==u? Yes) + A[u,u]*(x==v? No) + A[u,v]*(x==u? No)
        #      = A[x,v]. Correct.
        
        # So the update is:
        # For color c, changing edge (u,v) from 0 to 1 (add):
        # 1. Save cols u, v of A_c.
        # 2. M_c += outer(col_u, e_v) + outer(e_v, col_u) + ... wait.
        #    Let's just do:
        #    M_c[:, v] += col_u  (where col_u is A_c[:, u])
        #    M_c[v, :] += col_u
        #    M_c[:, u] += col_v
        #    M_c[u, :] += col_v
        #    M_c[u, u] += 1
        #    M_c[v, v] += 1
        #    M_c[u, v] += 0 (as derived)
        #    M_c[v, u] += 0
        #    Wait, does M_c[u,v] change?
        #    M_c[u,v] = sum_k A[u,k]A[k,v].
        #    Added terms: k=v -> A[u,v]A[v,v]=0. k=u -> A[u,u]A[u,v]=0.
        #    So M[u,v] unchanged.
        #    My update:
        #    M_c[:, v] += col_u -> M_c[u,v] += col_u[u] = A[u,u] = 0. Correct.
        #    M_c[v, :] += col_u -> M_c[v,u] += col_u[v] = A[v,u] = 0 (before add). Correct.
        #    M_c[:, u] += col_v -> M_c[u,u] += col_v[u] = A[v,u] = 0 (before add). 
        #    Wait, M_c[u,u] should increase by 1.
        #    Where does the +1 come from?
        #    M_c[u,u] = sum_k A[u,k]^2.
        #    New edge (u,v). A[u,v] becomes 1.
        #    So sum increases by 1.
        #    In my update:
        #    M_c[u,u] is updated by:
        #    1. M_c[:, v] += col_u -> index (u,v) is updated, not (u,u).
        #    2. M_c[v, :] += col_u -> index (v,u) is updated.
        #    3. M_c[:, u] += col_v -> index (u,u) is updated by col_v[u] = A[v,u] = 0.
        #    4. M_c[u, :] += col_v -> index (u,u) is updated by col_v[u] = A[v,u] = 0.
        #    So M_c[u,u] gets 0 from these.
        #    I need to explicitly add 1 to M_c[u,u] and M_c[v,v].
        #    And what about M_c[u,v]?
        #    It gets 0 from col_u[u] and col_u[v].
        #    So M_c[u,v] is unchanged. Correct.
        
        # So the update for adding (u,v) to color c is:
        # M_c[:, v] += A_c[:, u]
        # M_c[v, :] += A_c[:, u]
        # M_c[:, u] += A_c[:, v]
        # M_c[u, :] += A_c[:, v]
        # M_c[u, u] += 1
        # M_c[v, v] += 1
        # A_c[u, v] = 1
        # A_c[v, u] = 1
        
        # And for removing, subtract the same (using old values).
        
        # Cost update:
        # We need delta cost.
        # Cost = sum_{i<j} binom(M[i,j], 2)
        # We only need to update entries where M changed.
        # Changed entries in M_c:
        # - Column v: all rows i. M[i,v] changed.
        # - Row v: all cols i. M[v,i] changed. (Symmetric, so same set of pairs {i,v})
        # - Column u: all rows i. M[i,u] changed.
        # - Row u: all cols i. M[u,i] changed. (Pairs {i,u})
        # - Diagonals (u,u) and (v,v). (Do not contribute to C4 cost as i!=j)
        # - (u,v) and (v,u) did NOT change.
        
        # So the pairs (i,j) with i!=j that changed are:
        # {i, v} for all i != v
        # {i, u} for all i != u
        # Note: {u, v} is in both sets?
        # For i=u in first set: {u, v}. M[u,v] didn't change. So we should skip it or ensure delta is 0.
        # For i=v in second set: {v, u}. M[v,u] didn't change.
        
        # So we iterate over i in 0..n-1:
        # 1. If i != v: update cost for pair (i, v).
        # 2. If i != u: update cost for pair (i, u).
        # 3. If i == u and i == v (impossible).
        
        # Let's implement a helper to get current cost contribution of a pair (i,j) in color c.
        # contrib = binom(M_c[i,j], 2)
        
        # In the loop, for each move:
        # 1. Calculate delta_cost for old_color (removal) and new_color (addition).
        # 2. If new_cost < current_cost + acceptance_prob * (new_cost - current_cost):
        #    Apply move.
        
        # To be efficient, we can just recompute the cost for the affected pairs.
        
        t = time.time()
        # Simulated Annealing Parameters
        # We want to explore.
        # Let's try multiple restarts with different seeds if stuck.
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - t > time_limit:
                break
            
            # Re-initialize if not first restart? Or just continue from current state?
            # Usually SA continues. But if stuck at high cost, random restart helps.
            if restart > 0:
                # Randomize a subset of edges to escape
                for _ in range(num_edges // 5):
                    idx = random.randint(0, num_edges - 1)
                    colors[idx] = random.randint(0, 4)
                # Rebuild mats and common_mats
                for m in mats:
                    m[:] = 0
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = colors[idx]
                        mats[c][i, j] = 1
                        mats[c][j, i] = 1
                        idx += 1
                common_mats = [m @ m for m in mats]
                current_cost = calc_cost(common_mats)
            
            # SA Loop
            # Temperature
            T_start = 50.0
            T_end = 0.1
            steps = 5000
            for step in range(steps):
                if time.time() - t > time_limit:
                    return colors.tolist(), (current_cost == 0)
                
                if current_cost == 0:
                    return colors.tolist(), True
                
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                
                u, v = edge_lookup[e_idx]
                
                # Calculate delta cost
                delta_cost = 0.0
                
                # For old_c (removal)
                A_old = mats[old_c]
                M_old = common_mats[old_c]
                
                # Save cols
                col_u_old = A_old[:, u].copy()
                col_v_old = A_old[:, v].copy()
                
                # Update cost for affected pairs
                # Pairs: (i, v) for i != v, (i, u) for i != u
                # Note: M_old[i,v] changes by -col_u_old[i]
                # M_old[i,u] changes by -col_v_old[i]
                
                # We need old M values to compute old binom, and new M values for new binom.
                # Old M[i,v] = M_old[i,v]
                # New M[i,v] = M_old[i,v] - col_u_old[i]
                
                # Iterate i
                for i in range(n):
                    if i == v: continue
                    # Pair (i, v)
                    m_old_val = M_old[i, v]
                    m_new_val = m_old_val - col_u_old[i]
                    if m_old_val >= 2:
                        delta_cost -= 0.5 * m_old_val * (m_old_val - 1)
                    if m_new_val >= 2:
                        delta_cost += 0.5 * m_new_val * (m_new_val - 1)
                    
                    if i == u: continue
                    # Pair (i, u)
                    m_old_val = M_old[i, u]
                    m_new_val = m_old_val - col_v_old[i]
                    if m_old_val >= 2:
                        delta_cost -= 0.5 * m_old_val * (m_old_val - 1)
                    if m_new_val >= 2:
                        delta_cost += 0.5 * m_new_val * (m_new_val - 1)
                
                # For new_c (addition)
                A_new = mats[new_c]
                M_new = common_mats[new_c]
                
                col_u_new = A_new[:, u].copy()
                col_v_new = A_new[:, v].copy()
                
                for i in range(n):
                    if i == v: continue
                    # Pair (i, v)
                    m_old_val = M_new[i, v]
                    m_new_val = m_old_val + col_u_new[i]
                    if m_old_val >= 2:
                        delta_cost -= 0.5 * m_old_val * (m_old_val - 1)
                    if m_new_val >= 2:
                        delta_cost += 0.5 * m_new_val * (m_new_val - 1)
                    
                    if i == u: continue
                    # Pair (i, u)
                    m_old_val = M_new[i, u]
                    m_new_val = m_old_val + col_v_new[i]
                    if m_old_val >= 2:
                        delta_cost -= 0.5 * m_old_val * (m_old_val - 1)
                    if m_new_val >= 2:
                        delta_cost += 0.5 * m_new_val * (m_new_val - 1)
                
                new_cost = current_cost + delta_cost
                
                # Acceptance
                if new_cost < current_cost:
                    accept = True
                else:
                    # T decreases over steps
                    T = T_start * (T_end / T_start) ** (step / steps)
                    if random.random() < np.exp(-delta_cost / T):
                        accept = True
                    else:
                        accept = False
                
                if accept:
                    # Apply changes
                    # Update M_old
                    M_old[:, v] -= col_u_old
                    M_old[v, :] -= col_u_old
                    M_old[:, u] -= col_v_old
                    M_old[u, :] -= col_v_old
                    M_old[u, u] -= 1
                    M_old[v, v] -= 1
                    A_old[u, v] = 0
                    A_old[v, u] = 0
                    
                    # Update M_new
                    M_new[:, v] += col_u_new
                    M_new[v, :] += col_u_new
                    M_new[:, u] += col_v_new
                    M_new[u, :] += col_v_new
                    M_new[u, u] += 1
                    M_new[v, v] += 1
                    A_new[u, v] = 1
                    A_new[v, u] = 1
                    
                    colors[e_idx] = new_c
                    current_cost = new_cost
                else:
                    # Revert is not needed since we didn't modify anything yet
                    pass
        
        return colors.tolist(), (current_cost == 0)

    # Main loop: try increasing n
    # We have ~110s.
    # Let's try n=12, 13, ... 
    # If we find a solution for n, we update best_n and best_colors.
    # We should spend more time on higher n if lower n are easy.
    # But n=12 is likely easy. n=26 is hard.
    # Let's allocate time based on n.
    
    # Strategy: Try n from 12 up to 26.
    # For each n, give it a time budget.
    # If it fails, move on.
    # If it succeeds, we have a valid coloring for that n.
    # We want to MAXIMIZE n.
    
    # Let's try a few high n's first with short time, then lower n's with longer time?
    # No, if n=26 is not found, n=25 might be.
    # Let's just iterate 12 to 26.
    
    # Time allocation:
    # Total 110s.
    # Let's give 5s to each n initially? 15 * 5 = 75s.
    # If we find a solution, we can stop? No, we want MAX n.
    # So we should try the highest n first?
    # If we find n=20, we should still try n=21..26.
    
    # Let's try n from 26 down to 12.
    # If we find a valid one, we record it and continue to lower n?
    # No, if we find n=26, that's the max possible (known bound).
    # If we find n=25, we should still try n=26?
    # Yes.
    
    # But if n=26 is very hard, we might waste all time.
    # Let's give a small budget to high n, larger to lower n?
    # Or just fixed budget per n.
    
    # Let's try:
    # n=26: 5s
    # n=25: 5s
    # ...
    # n=12: 5s
    # Total 15 * 5 = 75s.
    # Remaining 35s for the best n found so far to refine?
    # Or just iterate.
    
    # Let's do a simple loop.
    
    for n in range(26, 11, -1):
        if time.time() > deadline - 5:
            break
        
        # Time budget for this n
        # If we already have a solution for n-1, we have less to gain from n?
        # No, n is better.
        # Let's give 5 seconds per n.
        t_start = time.time()
        limit = 5.0
        
        colors_n, success = solve_n(n, limit)
        
        if success:
            best_n = n
            best_colors = colors_n
            # If we found n, we can break?
            # No, we want MAX n.
            # If we are iterating downwards, and we found n, we should continue to n-1?
            # No, n is higher than n-1.
            # If we found n=26, we are done.
            # If we found n=25, we should have tried n=26 first.
            # Since we iterate 26 -> 12, if we find n=25, it means n=26 failed.
            # So we can stop?
            # Not necessarily, maybe n=26 was just unlucky.
            # But with 5s each, it's a gamble.
            # Let's just record the best and continue.
            
    return {'n': best_n, 'colors': best_colors}

