import itertools
import random
import time

def construct():
    n = 16
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute 8-vertex subsets for each pair (u, v)
    # pairs_to_sets[(u, v)] = list of 8-tuples containing u and v
    pairs_to_sets = {}
    all_vertices = range(n)
    all_8_sets = list(itertools.combinations(all_vertices, 8))
    
    for s in all_8_sets:
        s_set = set(s)
        for u, v in itertools.combinations(s, 2):
            key = (min(u, v), max(u, v))
            if key not in pairs_to_sets:
                pairs_to_sets[key] = []
            pairs_to_sets[key].append(s)
    
    # Precompute bipartitions for each 8-set
    # bipartitions_cache[s] = list of (mask_a, mask_b) where mask_a, mask_b are bitmasks
    bipartitions_cache = {}
    for s in all_8_sets:
        v0 = s[0]
        rest = s[1:]
        parts = []
        for comb in itertools.combinations(rest, 3):
            a = (v0,) + comb
            b = tuple(x for x in s if x not in a)
            # Create bitmasks
            mask_a = 0
            for x in a:
                mask_a |= (1 << x)
            mask_b = 0
            for x in b:
                mask_b |= (1 << x)
            parts.append((mask_a, mask_b, a, b))
        bipartitions_cache[s] = parts

    def is_valid_addition(neighbors, u, v):
        """
        neighbors: list of 16 ints, neighbors[i] is bitmask of neighbors of i
        u, v: vertices to add edge between
        Returns True if adding (u,v) does NOT create a Q3
        """
        key = (min(u, v), max(u, v))
        if key not in pairs_to_sets:
            return True
        
        sets_to_check = pairs_to_sets[key]
        
        for s in sets_to_check:
            parts = bipartitions_cache[s]
            for mask_a, mask_b, a, b in parts:
                # u and v must be in different parts
                u_in_a = (mask_a >> u) & 1
                v_in_a = (mask_a >> v) & 1
                
                if u_in_a == v_in_a:
                    continue
                
                if u_in_a:
                    part_u_mask = mask_a
                    part_v_mask = mask_b
                    part_u_list = a
                    part_v_list = b
                else:
                    part_u_mask = mask_b
                    part_v_mask = mask_a
                    part_u_list = b
                    part_v_list = a
                
                # Check missing degrees
                # For x in part_u:
                #   missing_deg = popcount(part_v_mask & ~neighbors[x])
                #   If x == u, we are adding (u,v), so v is now a neighbor.
                #   So effective neighbors[x] = neighbors[x] | (1 << v)
                #   But we only care about neighbors in part_v.
                #   So missing_deg = popcount(part_v_mask & ~(neighbors[x] | (1 << v)))
                #   Wait, if x != u, neighbors[x] doesn't change.
                #   If x == u, neighbors[u] gains v.
                
                ok = True
                for x in part_u_list:
                    # Simulate adding edge (u,v)
                    # If x == u, neighbor v is added.
                    # If x != u, no change.
                    # But v is in part_v.
                    # So for x == u, v is now a neighbor.
                    # For x != u, v is NOT a neighbor (unless it was already, which is false since we are adding it).
                    # Wait, if x != u, does adding (u,v) affect x's neighbors in part_v?
                    # No. x's neighbors in part_v are determined by edges (x, y) where y in part_v.
                    # Adding (u,v) only adds edge (u,v).
                    # So only u's neighbor set changes (gains v).
                    # Only v's neighbor set changes (gains u).
                    
                    if x == u:
                        # v is in part_v. So v is a neighbor of u now.
                        # missing neighbors in part_v = part_v_mask & ~(neighbors[u] | (1 << v))
                        # But v is in part_v_mask.
                        # So we can just check:
                        # popcount(part_v_mask & ~neighbors[u]) - 1 (since v is now present)
                        # Or simply: popcount(part_v_mask & ~(neighbors[u] | (1 << v)))
                        miss_mask = part_v_mask & ~(neighbors[u] | (1 << v))
                    else:
                        # x is not u. Edge (u,v) doesn't affect x's neighbors in part_v.
                        # Unless x == v? No, x is in part_u, v is in part_v.
                        miss_mask = part_v_mask & ~neighbors[x]
                    
                    if bin(miss_mask).count('1') > 1:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # Check part_v
                for y in part_v_list:
                    if y == v:
                        # u is in part_u. So u is a neighbor of v now.
                        miss_mask = part_u_mask & ~(neighbors[v] | (1 << u))
                    else:
                        miss_mask = part_u_mask & ~neighbors[y]
                    
                    if bin(miss_mask).count('1') > 1:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # If we reach here, all missing degrees <= 1.
                # This means missing edges form a matching.
                # So Q3 is formed.
                return False
        
        return True

    # Greedy construction with local search
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    best_edges = 0
    best_graph = None
    
    trials = 0
    while time.time() - start_time < time_limit:
        trials += 1
        random.seed(42 + trials)
        random.shuffle(all_edges)
        
        neighbors = [0] * n
        edge_set = set()
        
        for u, v in all_edges:
            if is_valid_addition(neighbors, u, v):
                # Add edge
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edge_set.add((u, v))
        
        if len(edge_set) > best_edges:
            best_edges = len(edge_set)
            best_graph = neighbors.copy()
            best_edge_set = edge_set.copy()
            
            # If we reached a high number, we might break early?
            # ex(16, Q3) is likely around 50-60?
            # Let's continue to see if we can improve.
            
    # Return the best found
    if best_graph is None:
        return []
        
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_graph[u] >> v) & 1:
                result.append([u, v])
    return result
