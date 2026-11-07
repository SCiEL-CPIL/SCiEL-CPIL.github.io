import sys
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # is_ind[S] is True if S is an independent set in G
    is_ind = [False] * (1 << n)
    is_ind[0] = True
    # is_clique[S] is True if S is a clique in G (independent in complement)
    is_clique = [False] * (1 << n)
    is_clique[0] = True
    
    # Precompute is_ind and is_clique
    # A set S is independent if for all v in S, no neighbor of v is in S.
    # Or simply, for all v in S, (adj[v] & S) == 0? No, that checks if v has NO neighbors in S.
    # Wait, if S={u,v}, and (u,v) is an edge, it's not independent.
    # Condition: for all v in S, (adj[v] & S) must only contain v? No, adj[v] doesn't contain v.
    # So for all v in S, (adj[v] & S) == 0 is incorrect if S has multiple vertices.
    # Correct: S is independent iff for all v in S, (adj[v] & S) == 0 is wrong.
    # It should be: for all v in S, (adj[v] & S) is a subset of {v}? No, v is not in adj[v].
    # So (adj[v] & S) should be empty?
    # If S={0, 2} and edge (0,2) exists. adj[0] has bit 2. adj[0] & S has bit 2. So it's not 0.
    # If S={0, 1} and no edge. adj[0] & S = 0.
    # So yes, S is independent iff for all v in S, (adj[v] & S) == 0.
    # This is O(n * 2^n).
    
    # Faster way: 
    # is_ind[S] = is_ind[S without lsb] and (adj[lsb] & (S without lsb)) == 0
    for s in range(1, 1 << n):
        lsb = s & -s
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        is_ind[s] = is_ind[rest] and ((adj[v] & rest) == 0)
        
        # For clique: S is clique iff for all v in S, (adj[v] & rest) == rest
        # i.e., all other vertices in S are neighbors of v.
        is_clique[s] = is_clique[rest] and ((adj[v] & rest) == rest)

    # dp_G[S] = chromatic number of G[S]
    # dp_comp[S] = chromatic number of complement of G[S] (which is clique cover number of G[S])
    
    dp_G = [n + 1] * (1 << n)
    dp_G[0] = 0
    dp_comp = [n + 1] * (1 << n)
    dp_comp[0] = 0
    
    # To reconstruct, we need to store the choice of the first independent set/clique
    # choice_G[S] = the independent set I used in the optimal partition of S
    # choice_comp[S] = the clique I used in the optimal partition of S
    choice_G = [0] * (1 << n)
    choice_comp = [0] * (1 << n)
    
    # Calculate dp_G
    # dp_G[S] = 1 + min { dp_G[S \ I] : I is independent subset of S, I != empty }
    # We can iterate over all independent subsets I of S?
    # That is expensive.
    # Instead, iterate over S, and for each S, iterate over all subsets I of S that are independent.
    # This is 3^n.
    
    # Optimization: 
    # We can just iterate I (independent sets) and update supersets?
    # No, standard DP is:
    # for S in 1..2^n:
    #   best = n
    #   for I subset of S:
    #      if is_ind[I]:
    #         if dp_G[S^I] + 1 < best:
    #             best = dp_G[S^I] + 1
    #             choice_G[S] = I
    #   dp_G[S] = best
    
    # This is 3^n. 3^14 = 4.7M. This is fine.
    
    # Let's optimize the inner loop by iterating submasks.
    for s in range(1, 1 << n):
        best_g = n
        best_c = n
        best_ig = 0
        best_ic = 0
        # Iterate submasks of s
        sub = s
        while sub:
            if is_ind[sub]:
                val = dp_G[s ^ sub] + 1
                if val < best_g:
                    best_g = val
                    best_ig = sub
            if is_clique[sub]:
                val = dp_comp[s ^ sub] + 1
                if val < best_c:
                    best_c = val
                    best_ic = sub
            sub = (sub - 1) & s
            
        dp_G[s] = best_g
        choice_G[s] = best_ig
        dp_comp[s] = best_c
        choice_comp[s] = best_ic

    # Search for minimum M
    # M = k_cliques + k_ind_sets
    # We want min M such that exists S:
    # dp_comp[S] <= k_cliques
    # dp_G[full_mask ^ S] <= k_ind_sets
    # where k_cliques + k_ind_sets = M
    
    best_partition = None
    best_M = n + 1
    
    # We can iterate M from 1 to n
    for M in range(1, n + 1):
        found = False
        # k_cliques can range from 0 to M
        for k_c in range(0, M + 1):
            k_i = M - k_c
            # Check if there is a split S
            # dp_comp[S] <= k_c and dp_G[full_mask ^ S] <= k_i
            # We can iterate S? 2^n iterations.
            # Total 2^n * n^2 = 16384 * 196 ~ 3M. This is fine.
            for s in range(1 << n):
                if dp_comp[s] <= k_c and dp_G[full_mask ^ s] <= k_i:
                    # Found a valid split
                    # Reconstruct
                    parts = []
                    
                    # Extract cliques from S
                    cur_s = s
                    while cur_s:
                        clique = choice_comp[cur_s]
                        # clique is a mask
                        # Convert mask to list of vertices
                        verts = []
                        m = clique
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            verts.append(v)
                            m ^= lsb
                        if verts:
                            parts.append(verts)
                        cur_s ^= clique
                        if cur_s == 0:
                            break
                        # Safety break if no progress
                        if choice_comp[cur_s] == 0 and cur_s != 0:
                             # This shouldn't happen if DP is correct
                             break
                             
                    # Extract independent sets from C = full_mask ^ s
                    cur_c = full_mask ^ s
                    while cur_c:
                        ind_set = choice_G[cur_c]
                        verts = []
                        m = ind_set
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            verts.append(v)
                            m ^= lsb
                        if verts:
                            parts.append(verts)
                        cur_c ^= ind_set
                        if cur_c == 0:
                            break
                        if choice_G[cur_c] == 0 and cur_c != 0:
                             break

                    # Verify partition
                    # Check if all vertices are present exactly once
                    # Check if each part is clique or independent
                    # If valid, return
                    # We assume DP correctness, but let's add a quick check for safety
                    # Since we need to return the list of lists
                    best_partition = parts
                    found = True
                    break
            if found:
                break
        if found:
            break
            
    if best_partition:
        return best_partition
    else:
        # Fallback: trivial partition (each vertex its own part)
        # This is always valid (cliques of size 1)
        return [[i] for i in range(n)]
