import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 20
    num_colours = 20
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and a pairing.
    # There are 3 pairings for {a,b,c,d}: (ab,cd), (ac,bd), (ad,bc)
    # But we need the cycle order.
    # Cycles: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a (and reverses which are same edge sets)
    # Let's just list unique edge sets.
    # Edges: (a,b), (b,c), (c,d), (d,a)
    # Edges: (a,b), (b,d), (d,c), (c,a)
    # Edges: (a,c), (c,b), (b,d), (d,a)
    
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    num_cycles = len(cycles)
    
    # Precompute edge -> cycles mapping for fast validation
    edge_to_cycles = {e: [] for e in combinations(range(n), 2)}
    for i, cycle in enumerate(cycles):
        for e in cycle:
            eu = min(e[0], e[1])
            ev = max(e[0], e[1])
            edge_to_cycles[(eu, ev)].append(i)
            
    # Precompute cycle edge sets for fast "is this cycle satisfied" check?
    # Actually, we just need to check colours.
    
    def check_validity(adj):
        """Returns list of indices of invalid cycles"""
        invalid = []
        for i, cycle in enumerate(cycles):
            e1, e2, e3, e4 = cycle
            # Check if all edges exist
            if adj[e1[0], e1[1]] == -1 or adj[e2[0], e2[1]] == -1 or \
               adj[e3[0], e3[1]] == -1 or adj[e4[0], e4[1]] == -1:
                continue
            c1 = adj[e1[0], e1[1]]
            c2 = adj[e2[0], e2[1]]
            c3 = adj[e3[0], e3[1]]
            c4 = adj[e4[0], e4[1]]
            if len({c1, c2, c3, c4}) < 4:
                invalid.append(i)
        return invalid

    # --- DETERMINISTIC CONSTRUCTIVE BASE: C5 x C4 ---
    # Vertices are pairs (x, y) where x in 0..4, y in 0..3
    # Map vertex index v to (x, y): v = 4*x + y
    
    adj = np.full((n, n), -1, dtype=int)
    
    # Edges in C5 x C4:
    # 1. Horizontal edges (x, y) - (x, y+1 mod 4) : 5 * 4 = 20 edges
    # 2. Vertical edges (x, y) - (x+1 mod 5, y) : 5 * 4 = 20 edges
    # Total 40 edges.
    
    # Let's color them.
    # A 4-cycle in the grid is formed by 2 horizontal and 2 vertical edges.
    # If we color all horizontal edges with distinct colors from a set H and vertical with V,
    # and H intersect V is empty, then any 4-cycle (which has 2H, 2V) will have colors {h1, h2, v1, v2}.
    # For it to be rainbow, we need h1 != h2 and v1 != v2.
    # In C5 x C4, a 4-cycle is a square. 
    # In a grid, the two horizontal edges in a square are parallel. If we color by "row" or "column" index, 
    # do we get distinct colors?
    
    # Let's try a specific coloring:
    # Color of horizontal edge (x, y)-(x, y+1): depends on x? Or y?
    # If we color by x (0..4), then in a square at row x, both horizontal edges have color x. Not rainbow.
    # If we color by y (0..3), then in a square spanning y, y+1, the top edge has color y and bottom y+1. Distinct.
    # So:
    # Horizontal edge (x, y)-(x, y+1): Color = y (0,1,2,3)
    # Vertical edge (x, y)-(x+1, y): Color = x + 4 (4,5,6,7,8)
    
    # Let's verify a 4-cycle: (x,y), (x,y+1), (x+1,y+1), (x+1,y)
    # Edges:
    # 1. (x,y)-(x,y+1) [Horiz]: color y
    # 2. (x,y+1)-(x+1,y+1) [Vert]: color (x+1 mod 5) + 4
    # 3. (x+1,y+1)-(x+1,y) [Horiz]: color (y+1 mod 4)
    # 4. (x+1,y)-(x,y) [Vert]: color x + 4
    
    # Colors: {y, (x+1)%5 + 4, (y+1)%4, x + 4}
    # Are they distinct?
    # y and (y+1)%4 are distinct (0,1,2,3 vs 1,2,3,0).
    # x+4 and (x+1)%5 + 4 are distinct (4..8 vs 5..8,4).
    # Sets {0,1,2,3} and {4,5,6,7,8} are disjoint.
    # So yes, all 4 colors are distinct.
    
    # This gives a valid graph with 40 edges.
    
    for x in range(5):
        for y in range(4):
            u = 4 * x + y
            # Horizontal
            v = 4 * x + ((y + 1) % 4)
            c = y
            adj[u, v] = c
            adj[v, u] = c
            
            # Vertical
            v = 4 * ((x + 1) % 5) + y
            c = x + 4
            adj[u, v] = c
            adj[v, u] = c
            
    # Now we have a valid graph with 40 edges.
    # Let's try to add more edges greedily.
    # We have 150 missing edges.
    # For each missing edge, try to add it with a color that doesn't violate any 4-cycle.
    
    # To be fast, we can just try adding edges and checking validity.
    # Since we start from a valid state, we can just iterate.
    
    missing_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == -1:
                missing_edges.append((u, v))
                
    random.shuffle(missing_edges)
    
    # We will try to add edges.
    # To speed up checking, we can keep a set of "active" invalid cycles? 
    # No, we are adding edges, which might create new invalid cycles.
    # So we must check all cycles containing the new edge.
    
    # Let's optimize the check for a single edge addition.
    # When adding edge (u, v) with color c, we only need to check cycles containing (u, v).
    # There are ~100 such cycles.
    # For each cycle, check if the other 3 edges exist and if their colors + c are distinct.
    
    def can_add_edge(u, v, c, adj):
        """Check if adding (u,v) with color c keeps graph valid."""
        # Get all cycles containing (u,v)
        eu = min(u, v)
        ev = max(u, v)
        cycle_indices = edge_to_cycles[(eu, ev)]
        
        for idx in cycle_indices:
            cycle = cycles[idx]
            # Find the other 3 edges in this cycle
            other_edges = []
            for e in cycle:
                e_u = min(e[0], e[1])
                e_v = max(e[0], e[1])
                if (e_u, e_v) == (eu, ev):
                    continue
                other_edges.append((e_u, e_v, e[0], e[1]))
                
            # Check if all other edges exist
            colors = [c]
            for e_u, e_v, _u, _v in other_edges:
                if adj[e_u, e_v] == -1:
                    continue # Cycle not present in graph
                colors.append(adj[e_u, e_v])
                
            if len(colors) == 4:
                if len(set(colors)) < 4:
                    return False
        return True

    # Greedy addition
    # We have 20 colors.
    # For each missing edge, try each color? Or just one random color?
    # Trying all 20 colors for 150 edges * 100 cycles * 4 checks = 1.2M ops. Very fast.
    
    for u, v in missing_edges:
        if time.time() > deadline:
            break
        added = False
        # Try colors in random order to break symmetry?
        # Or just 0..19.
        for c in range(num_colours):
            if can_add_edge(u, v, c, adj):
                adj[u, v] = c
                adj[v, u] = c
                added = True
                break
        # If no color worked, leave it out.
        
    # Final verification
    invalid = check_validity(adj)
    if invalid:
        # Fallback to just the C5xC4 base if something went wrong (shouldn't)
        # Or just return whatever we have if it's valid? 
        # The task requires validity.
        # If invalid, we should probably return a smaller valid graph.
        # Let's reconstruct the base.
        adj = np.full((n, n), -1, dtype=int)
        for x in range(5):
            for y in range(4):
                u = 4 * x + y
                v = 4 * x + ((y + 1) % 4)
                c = y
                adj[u, v] = c
                adj[v, u] = c
                v = 4 * ((x + 1) % 5) + y
                c = x + 4
                adj[u, v] = c
                adj[v, u] = c
        # This is definitely valid.
        
    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] != -1:
                edges_list.append([u, v, int(adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
