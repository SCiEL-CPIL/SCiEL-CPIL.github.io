import time
import sys
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    all_mask = (1 << n) - 1
    # Non-adjacency mask (vertices not adjacent to v, excluding v itself)
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = all_mask & ~(adj[v] | (1 << v))

    deadline = time.time() + 55
    
    # Precompute all maximal cliques and independent sets as bitmasks
    # For n=14, this is feasible
    def get_maximal_cliques():
        cliques = set()
        # Bron-Kerbosch variant to find all maximal cliques
        # But for n=14, we can just enumerate all subsets and check
        # Actually, let's use a simpler approach: enumerate all subsets
        # 2^14 = 16384 subsets, check each if it's a clique
        for mask in range(1, 1 << n):
            # Check if mask is a clique
            is_clique = True
            m = mask
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                # Check if all other vertices in mask are adjacent to u
                others = mask & ~lb
                if (adj[u] & others) != others:
                    is_clique = False
                    break
                m ^= lb
            if is_clique:
                # Check if maximal: no superset is also a clique
                # For simplicity, just add all cliques, we'll filter later
                cliques.add(mask)
        
        # Filter to maximal cliques
        maximal = []
        cliques_list = list(cliques)
        for i, c in enumerate(cliques_list):
            is_maximal = True
            for j, d in enumerate(cliques_list):
                if i != j and (c | d) == d:
                    is_maximal = False
                    break
            if is_maximal:
                maximal.append(c)
        return maximal
    
    def get_maximal_independent_sets():
        # Independent set in G = Clique in complement G
        # Complement adjacency
        comp_adj = [0] * n
        for v in range(n):
            comp_adj[v] = all_mask & ~(adj[v] | (1 << v))
        
        cliques = set()
        for mask in range(1, 1 << n):
            # Check if mask is a clique in complement (i.e., independent set in G)
            is_clique = True
            m = mask
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                others = mask & ~lb
                if (comp_adj[u] & others) != others:
                    is_clique = False
                    break
                m ^= lb
            if is_clique:
                cliques.add(mask)
        
        # Filter to maximal
        maximal = []
        cliques_list = list(cliques)
        for i, c in enumerate(cliques_list):
            is_maximal = True
            for j, d in enumerate(cliques_list):
                if i != j and (c | d) == d:
                    is_maximal = False
                    break
            if is_maximal:
                maximal.append(c)
        return maximal
    
    max_cliques = get_maximal_cliques()
    max_indep = get_maximal_independent_sets()
    
    # For each maximal set, precompute the set of vertices it contains
    # We'll use these for subset enumeration in the search
    
    def solve_k(k):
        """
        Try to partition vertices into k parts.
        Parts are dynamically typed (Clique or Independent Set).
        """
        # State:
        # masks[i]: bitmask of vertices in part i
        # types[i]: 0=Unknown (size 0 or 1), 1=Clique, 2=Independent
        # allows[i]: bitmask of vertices that can be added to part i
        masks = [0] * k
        types = [0] * k
        allows = [all_mask] * k
        
        # Memoization
        failed = set()
        
        def get_unassigned():
            assigned = 0
            for m in masks:
                assigned |= m
            return all_mask ^ assigned

        def backtrack():
            if time.time() > deadline:
                return False
            
            unassigned = get_unassigned()
            if unassigned == 0:
                return True
            
            # Create a canonical state key for memoization
            # Sort parts by (type, allows) to break symmetry
            part_states = []
            for i in range(k):
                if masks[i] != 0:
                    part_states.append((types[i], allows[i]))
            part_states.sort()
            state_key = (unassigned, tuple(part_states))
            
            if state_key in failed:
                return False
            
            # Find the first empty part index to enforce symmetry breaking
            first_empty = -1
            for i in range(k):
                if masks[i] == 0:
                    first_empty = i
                    break
            
            # Dynamic variable ordering: pick the unassigned vertex with fewest options
            unassigned_list = []
            u = unassigned
            while u:
                lb = u & (-u)
                v = lb.bit_length() - 1
                unassigned_list.append(v)
                u ^= lb
            
            # Count options for each unassigned vertex
            min_options = float('inf')
            best_v = -1
            for v in unassigned_list:
                v_bit = 1 << v
                options = 0
                for i in range(k):
                    if masks[i] == 0:
                        if i == first_empty:
                            options += 1
                    elif (allows[i] & v_bit):
                        options += 1
                if options < min_options:
                    min_options = options
                    best_v = v
                if options == 0:
                    # Dead end - forward checking
                    failed.add(state_key)
                    return False
            
            v = best_v
            v_bit = 1 << v
            
            tried_any = False
            
            for i in range(k):
                # Symmetry breaking:
                # If this part is empty, and there is an earlier empty part, skip this one.
                if masks[i] == 0:
                    if i != first_empty:
                        continue
                
                # Check if v is allowed in part i
                if not (allows[i] & v_bit):
                    continue
                
                tried_any = True
                
                # Save state
                old_mask = masks[i]
                old_type = types[i]
                old_allow = allows[i]
                
                # Update state
                masks[i] |= v_bit
                
                if old_mask == 0:
                    # Part was empty, now has size 1
                    types[i] = 0
                    allows[i] = all_mask
                else:
                    # Part had at least 1 vertex
                    if types[i] == 0:
                        # Part had size 1
                        u_bit = old_mask
                        u = u_bit.bit_length() - 1
                        if adj[v] & u_bit:
                            types[i] = 1 # Clique
                            allows[i] = adj[v] & adj[u]
                        else:
                            types[i] = 2 # Independent
                            allows[i] = non_adj[v] & non_adj[u]
                    elif types[i] == 1:
                        # Part is Clique
                        types[i] = 1
                        allows[i] = allows[i] & adj[v]
                    else:
                        # Part is Independent
                        types[i] = 2
                        allows[i] = allows[i] & non_adj[v]
                
                if backtrack():
                    return True
                
                # Restore state
                masks[i] = old_mask
                types[i] = old_type
                allows[i] = old_allow
            
            if not tried_any:
                failed.add(state_key)
            
            return False
        
        if backtrack():
            # Reconstruct parts
            parts = []
            for i in range(k):
                if masks[i]:
                    part_verts = []
                    m = masks[i]
                    for v in range(n):
                        if m & (1 << v):
                            part_verts.append(v)
                    parts.append(part_verts)
            return parts
        return None

    # Try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        res = solve_k(k)
        if res:
            # Ensure return type is list of lists of ints
            return res
        
    # Fallback
    return [[v] for v in range(n)]
