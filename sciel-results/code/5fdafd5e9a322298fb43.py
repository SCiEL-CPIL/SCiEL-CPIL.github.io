import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 20
    # Fallback: Star
    best_edges = [[0, v, 0] for v in range(1, 20)]
    best_count = 19
    
    # Try K_{10,10}
    left = list(range(10))
    right = list(range(10, 20))
    
    # Precompute all 4-cycles in terms of indices
    # Cycle defined by (l1, l2, r1, r2)
    # Edges: (l1, r1), (l2, r1), (l2, r2), (l1, r2)
    cycles = []
    for l1, l2 in combinations(left, 2):
        for r1, r2 in combinations(right, 2):
            cycles.append((l1, l2, r1, r2))
            
    def is_valid_colouring(colours):
        # colours is a dict (l, r) -> c
        for l1, l2, r1, r2 in cycles:
            c1 = colours[(l1, r1)]
            c2 = colours[(l2, r1)]
            c3 = colours[(l2, r2)]
            c4 = colours[(l1, r2)]
            if len({c1, c2, c3, c4}) != 4:
                return False
        return True

    # Randomized search for valid K_{10,10}
    # We have 20 colors.
    # Start with a proper coloring (10 colors) and perturb?
    # Or just random assignment and fix?
    
    # Let's try to construct a valid one using a specific pattern
    # c[l][r] = (l * A + r * B) % 20?
    # We need to avoid collisions.
    
    # Try random permutations of a base structure
    # Base: c[l][r] = (l + r) % 10
    # Map 10 base colors to 20 actual colors randomly?
    # If base has collision, mapping doesn't fix it unless we split colors.
    
    # Better: Random assignment of 20 colors to 100 edges, then repair.
    # Given 140s, we can do many iterations.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        colours = {}
        # Random initialization
        for l in left:
            for r in right:
                colours[(l, r)] = random.randint(0, 19)
                
        # Repair loop
        valid = False
        for _ in range(100):
            if time.time() > deadline:
                break
            valid = True
            # Pick a random violated cycle
            # Or check all and fix one
            for l1, l2, r1, r2 in cycles:
                c1 = colours[(l1, r1)]
                c2 = colours[(l2, r1)]
                c3 = colours[(l2, r2)]
                c4 = colours[(l1, r2)]
                if len({c1, c2, c3, c4}) < 4:
                    valid = False
                    # Change one of the colors
                    # Randomly pick one of the 4 edges
                    edge_idx = random.randint(0, 3)
                    if edge_idx == 0:
                        colours[(l1, r1)] = random.randint(0, 19)
                    elif edge_idx == 1:
                        colours[(l2, r1)] = random.randint(0, 19)
                    elif edge_idx == 2:
                        colours[(l2, r2)] = random.randint(0, 19)
                    else:
                        colours[(l1, r2)] = random.randint(0, 19)
                    break # Restart check after fix
                    
            if valid:
                break
                
        if valid:
            best_count = 100
            best_edges = []
            for l in left:
                for r in right:
                    best_edges.append([l, r, colours[(l, r)]])
            # Check if we can add edges?
            # For now, return 100 if found.
            
            # Try to add edges?
            # If we have time, try adding 1 edge.
            # But 100 is a solid improvement over star (19).
            # Seed was 1.0 (100 edges).
            # So finding a valid K_{10,10} gives 1.0.
            # To beat 1.0, we need > 100.
            
            # Let's try to add an edge (l1, l2)
            # This requires checking new 4-cycles.
            # We'll skip for now to ensure we return a valid 100-edge graph.
            
    # If no valid K_{10,10} found, fallback to seed's logic or star
    # But the repair loop should find one quickly.
    
    return best_edges
