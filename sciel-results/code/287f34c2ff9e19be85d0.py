import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Complement adjacency for independent sets (cliques in complement)
    comp_adj = [0] * n
    for u in range(n):
        comp_adj[u] = ((1 << n) - 1) ^ (1 << u) ^ adj[u]
        
    def find_maximal_sets(adj_list):
        max_sets = []
        
        def expand(clique, candidates, excluded):
            if not candidates and not excluded:
                if clique:
                    max_sets.append(clique)
                return
            
            if not candidates:
                if not excluded:
                    if clique:
                        max_sets.append(clique)
                return
            
            pivot_candidates = candidates | excluded
            if not pivot_candidates:
                if clique:
                    max_sets.append(clique)
                return
            
            pivot = (pivot_candidates & -pivot_candidates).bit_length() - 1
            pivot_mask = 1 << pivot
            
            cand_not_adj = candidates & ~adj_list[pivot]
            
            while cand_not_adj:
                v_mask = cand_not_adj & -cand_not_adj
                v = v_mask.bit_length() - 1
                cand_not_adj ^= v_mask
                
                new_clique = clique | v_mask
                new_cand = candidates & ~adj_list[v]
                new_excl = excluded & ~adj_list[v]
                
                expand(new_clique, new_cand, new_excl)
                
                candidates ^= v_mask
                excluded |= v_mask
                
        expand(0, (1 << n) - 1, 0)
        return max_sets

    max_cliques = find_maximal_sets(adj)
    max_independents = find_maximal_sets(comp_adj)
    
    all_sets = list(set(max_cliques + max_independents))
    all_sets.sort(key=lambda x: -bin(x).count('1'))
    
    memo = {}
    
    def can_cover(remaining, k):
        if remaining == 0:
            return True
        if k == 0:
            return False
        key = (remaining, k)
        if key in memo:
            return memo[key]
            
        v_mask = remaining & -remaining
        
        for s in all_sets:
            if (s & remaining) != s:
                continue
            if (s & v_mask) != v_mask:
                continue
            
            new_remaining = remaining ^ s
            if can_cover(new_remaining, k - 1):
                memo[key] = True
                return True
                
        memo[key] = False
        return False

    for k in range(1, n + 1):
        if can_cover((1 << n) - 1, k):
            memo_recon = {}
            
            def get_partition(remaining, k):
                if remaining == 0:
                    return []
                if k == 0:
                    return None
                key = (remaining, k)
                if key in memo_recon:
                    return memo_recon[key]
                
                v_mask = remaining & -remaining
                
                for s in all_sets:
                    if (s & remaining) != s:
                        continue
                    if (s & v_mask) != v_mask:
                        continue
                    
                    new_remaining = remaining ^ s
                    sub_part = get_partition(new_remaining, k - 1)
                    if sub_part is not None:
                        part = [s] + sub_part
                        memo_recon[key] = part
                        return part
                        
                memo_recon[key] = None
                return None

            part_bitmasks = get_partition((1 << n) - 1, k)
            if part_bitmasks:
                partition = []
                for mask in part_bitmasks:
                    verts = [i for i in range(n) if (mask & (1 << i))]
                    partition.append(verts)
                return partition

    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
