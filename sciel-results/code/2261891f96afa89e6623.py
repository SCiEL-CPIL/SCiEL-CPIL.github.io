import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    num_colors = 20
    
    # Initialize complete graph
    edges = []
    edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            c = random.randint(0, num_colors - 1)
            edges.append([u, v, c])
            edge_set.add((u, v))
            edge_set.add((v, u))
            
    # Store colours in a dict for quick access
    # Key: (min(u,v), max(u,v))
    edge_colour = {}
    for u, v, c in edges:
        key = (min(u, v), max(u, v))
        edge_colour[key] = c
        
    # Generate all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices.
    # For each set of 4 vertices, there are 3 cycles.
    cycles = []
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Cycle 1: u-v-w-x-u
        cycles.append([(u,v), (v,w), (w,x), (x,u)])
        # Cycle 2: u-v-x-w-u
        cycles.append([(u,v), (v,x), (x,w), (w,u)])
        # Cycle 3: u-w-v-x-u
        cycles.append([(u,w), (w,v), (v,x), (x,u)])
        
    # Greedy deletion
    # We want to maximize edges, so we remove as few as possible.
    # Strategy: Find a bad cycle, remove an edge.
    # To be efficient, we can just iterate through cycles repeatedly.
    
    while time.time() < deadline:
        bad_found = False
        for cycle in cycles:
            # Check if all edges exist
            valid = True
            colours = []
            for e in cycle:
                u, v = e
                key = (min(u, v), max(u, v))
                if key not in edge_colour:
                    valid = False
                    break
                colours.append(edge_colour[key])
            
            if not valid:
                continue
                
            # Check if rainbow
            if len(set(colours)) < 4:
                # Bad cycle found
                bad_found = True
                # Remove an edge
                # Heuristic: remove the edge that appears in the most other cycles?
                # Too expensive. Just remove the first one?
                # Or random?
                # Let's remove the first one to be deterministic-ish but fast.
                # Actually, removing an edge that is shared by many bad cycles is better.
                # But we don't know which are bad without checking.
                # Let's just remove the first edge in the cycle.
                e_to_remove = cycle[0]
                u, v = e_to_remove
                key = (min(u, v), max(u, v))
                del edge_colour[key]
                edge_set.discard((u, v))
                edge_set.discard((v, u))
                break # Restart cycle check from beginning to ensure consistency
                
        if not bad_found:
            break

    # Reconstruct list of edges from edge_colour
    result = []
    for (u, v), c in edge_colour.items():
        result.append([u, v, c])
        
    return result
