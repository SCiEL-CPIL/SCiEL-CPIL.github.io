"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with a small valid coloring
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to extend to larger n using random local search
    max_n = 34  # Known published construction size
    
    for target_n in range(10, max_n + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Initialize colors for n vertices
            colors = [random.randint(0, 1) for _ in combinations(range(target_n), 3)]
            color_map = {}
            for i, (a, b, c) in enumerate(combinations(range(target_n), 3)):
                color_map[(a, b, c)] = colors[i]
            
            # Local search to fix violations
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Find a random red K4^(3)
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Check for red K4^(3)
                    found_red_k4 = False
                    for quad in combinations(range(target_n), 4):
                        a, b, c, d = quad
                        triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                        if all(color_map[t] == 0 for t in triples):
                            # Found red K4, flip one triple
                            flip_idx = list(triples).index(random.choice(triples))
                            # Find index in colors list
                            for i, (x,y,z) in enumerate(combinations(range(target_n), 3)):
                                if (x,y,z) == flip_idx:
                                    colors[i] = 1 - colors[i]
                                    color_map[flip_idx] = colors[i]
                                    improved = True
                                    break
                            found_red_k4 = True
                            break
                    
                    if found_red_k4:
                        continue
                    
                    # Check for blue K5^(3)
                    found_blue_k5 = False
                    for quint in combinations(range(target_n), 5):
                        a, b, c, d, e = quint
                        triples = list(combinations(quint, 3))
                        if all(color_map[t] == 1 for t in triples):
                            # Found blue K5, flip one triple
                            flip_triple = random.choice(triples)
                            for i, (x,y,z) in enumerate(combinations(range(target_n), 3)):
                                if (x,y,z) == flip_triple:
                                    colors[i] = 1 - colors[i]
                                    color_map[flip_triple] = colors[i]
                                    improved = True
                                    break
                            found_blue_k5 = True
                            break
                    
                    if found_blue_k5:
                        continue
                    
                    # No violations found
                    improved = False
                    break
            
            # Verify the coloring is valid
            if _is_valid(target_n, color_map):
                best_n = target_n
                best_colors = colors
                break
    
    return {'n': best_n, 'colors': best_colors}

def _is_valid(n, color_map):
    """Check if coloring has no red K4^(3) and no blue K5^(3)"""
    # Check for red K4^(3)
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        if all(color_map[t] == 0 for t in triples):
            return False
    
    # Check for blue K5^(3)
    for quint in combinations(range(n), 5):
        triples = list(combinations(quint, 3))
        if all(color_map[t] == 1 for t in triples):
            return False
    
    return True
# EVOLVE-BLOCK-END
