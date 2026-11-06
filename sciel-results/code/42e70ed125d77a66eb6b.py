import itertools
import random

def construct():
    n = 16
    best_edges = 0
    best_edge_set = set()
    
    # Precompute all 8-vertex subsets
    all_8_sets = list(itertools.combinations(range(n), 8))
    
    # Precompute bipartitions for each 8-set?
    # There are 35 bipartitions.
    # We can generate them on the fly or precompute.
    # Precomputing might be faster.
    bipartitions_cache = {}
    for s in all_8_sets:
        # s is a tuple of 8 vertices
        # Generate all ways to split into two sets of 4
        # To avoid duplicates, fix the smallest vertex in the first set
        v0 = s[0]
        rest = s[1:]
        parts = []
        for comb in itertools.combinations(rest, 3):
            a = (v0,) + comb
            b = tuple(x for x in s if x not in a)
            parts.append((a, b))
        bipartitions_cache[s] = parts

    def is_q3_free(edge_set):
        # Check if the graph defined by edge_set is Q3-free
        # A graph contains Q3 if there exists an 8-set S and bipartition A,B
        # such that missing edges between A and B form a matching.
        for s in all_8_sets:
            for a, b in bipartitions_cache[s]:
                # Check missing edges between a and b
                # A vertex x in a has missing degree = 4 - (number of neighbors in b present in edge_set)
                # If all missing degrees <= 1, then missing edges form a matching.
                # Wait, if missing edges form a matching, it means max missing degree <= 1.
                # This implies the present edges contain K_{4,4} minus a matching, which is Q3.
                # So if this condition holds, Q3 exists.
                # We want to return False if Q3 exists.
                valid = False
                for x in a:
                    miss_deg = 0
                    for y in b:
                        if (x, y) not in edge_set and (y, x) not in edge_set:
                            miss_deg += 1
                    if miss_deg > 1:
                        valid = True
                        break
                if not valid:
                    # All miss_deg <= 1, so missing edges form a matching
                    return False
        return True

    def try_add_edge(edge_set, u, v):
        # Check if adding (u,v) creates a Q3
        # Only need to check 8-sets containing u and v
        for s in all_8_sets:
            if u not in s or v not in s:
                continue
            for a, b in bipartitions_cache[s]:
                # Determine if u and v are in different parts
                if u in a:
                    if v not in b:
                        continue
                    part_u = a
                    part_v = b
                else:
                    if v not in a:
                        continue
                    part_u = b
                    part_v = a
                
                # Check missing degrees
                # We simulate adding (u,v)
                # For each x in part_u, miss_deg = 4 - (neighbors in part_v in edge_set) - (1 if x==u else 0)
                # Wait, u is in part_u. Its neighbors in part_v are checked.
                # If we add (u,v), u gains a neighbor v in part_v.
                # So u's miss_deg decreases by 1.
                # v is in part_v. Its neighbors in part_u are checked.
                # If we add (u,v), v gains a neighbor u in part_u.
                # So v's miss_deg decreases by 1.
                
                ok = True
                for x in part_u:
                    miss_deg = 0
                    for y in part_v:
                        if (x, y) not in edge_set and (y, x) not in edge_set:
                            miss_deg += 1
                    if x == u:
                        miss_deg -= 1
                    if miss_deg > 1:
                        ok = False
                        break
                if not ok:
                    continue
                    
                for y in part_v:
                    miss_deg = 0
                    for x in part_u:
                        if (x, y) not in edge_set and (y, x) not in edge_set:
                            miss_deg += 1
                    if y == v:
                        miss_deg -= 1
                    if miss_deg > 1:
                        ok = False
                        break
                if not ok:
                    continue
                    
                # If we reach here, all miss_deg <= 1 for this bipartition
                # So adding (u,v) creates a Q3
                return False
                
        return True

    # Run multiple random trials
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(50):
        random.seed(42 + trial)
        random.shuffle(all_edges)
        edge_set = set()
        
        for u, v in all_edges:
            if try_add_edge(edge_set, u, v):
                edge_set.add((u, v))
        
        if len(edge_set) > best_edges:
            best_edges = len(edge_set)
            best_edge_set = edge_set.copy()
            
    # Convert to list of lists
    result = [[u, v] for u, v in best_edge_set]
    return result
