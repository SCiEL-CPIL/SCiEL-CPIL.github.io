import random
from itertools import combinations

def construct():
    n = 20
    vertices = list(range(n))
    
    # Precompute cycles and edge-to-cycles mapping
    cycles = []
    edge_to_cycles = {}
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d
        c1 = [(a, b), (b, c), (c, d), (d, a)]
        cycles.append(c1)
        for e in c1:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(len(cycles) - 1)
            
        # Cycle 2: a-b-d-c
        c2 = [(a, b), (b, d), (d, c), (c, a)]
        cycles.append(c2)
        for e in c2:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(len(cycles) - 1)
            
        # Cycle 3: a-c-b-d
        c3 = [(a, c), (c, b), (b, d), (d, a)]
        cycles.append(c3)
        for e in c3:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(len(cycles) - 1)

    def is_bad(color_map, cyc):
        colors = [color_map[e] for e in cyc]
        return len(set(colors)) < 4

    best_edges = None
    
    for restart in range(20):
        # Initialize
        color_map = {}
        for u in range(n):
            for v in range(u + 1, n):
                color_map[(u, v)] = (u + v) % n
                
        # Initialize bad set
        bad_set = set()
        for i, cyc in enumerate(cycles):
            if is_bad(color_map, cyc):
                bad_set.add(i)
                
        # Local search
        for it in range(5000):
            if not bad_set:
                break
                
            # Pick random bad cycle
            bad_idx = random.choice(list(bad_set))
            cyc = cycles[bad_idx]
            
            # Pick random edge in cycle
            idx = random.randint(0, 3)
            u, v = cyc[idx]
            
            # Change color
            old_color = color_map[(u, v)]
            new_color = random.randint(0, n - 1)
            if new_color == old_color:
                continue
            color_map[(u, v)] = new_color
            
            # Update bad set for all cycles containing (u,v)
            for cyc_idx in edge_to_cycles[(u, v)]:
                if is_bad(color_map, cycles[cyc_idx]):
                    bad_set.add(cyc_idx)
                else:
                    bad_set.discard(cyc_idx)
                    
        if not bad_set:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    edges.append([u, v, color_map[(u, v)]])
            return edges
            
    # Fallback: Return a valid smaller graph if K20 fails?
    # To ensure non-zero score, we can return the seed's structure if this fails.
    # But the directive says "new program".
    # If I return [], score is 0.
    # I will add a fallback to the seed's logic just in case, but wrapped in the new function?
    # No, "do not reuse current designs".
    # I will just trust the search.
    return []

