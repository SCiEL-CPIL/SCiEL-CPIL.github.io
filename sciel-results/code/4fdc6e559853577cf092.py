"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Precompute all triples for various n values
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    def build_adjacency(n):
        """For each vertex, precompute which triples it participates in."""
        triples = get_triples(n)
        tri_list = [(i, t) for i, t in enumerate(triples)]
        vertex_to_triples = {v: [] for v in range(n)}
        for idx, (a, b, c) in enumerate(triples):
            vertex_to_triples[a].append(idx)
            vertex_to_triples[b].append(idx)
            vertex_to_triples[c].append(idx)
        return triples, vertex_to_triples
    
    def count_violations(n, colors, triples, vertex_to_triples):
        """Count red K4^(3) and blue K5^(3) violations."""
        red_violations = 0
        blue_violations = 0
        
        # Check for red K4: any 4 vertices where all 4 triples are red
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # The 4 triples in this quad
            tris = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            all_red = True
            for (x, y, z) in tris:
                # Find index of this triple
                idx = None
                for t_idx, (a2, b2, c2) in enumerate(triples):
                    if (a2, b2, c2) == (x, y, z):
                        idx = t_idx
                        break
                if idx is not None and colors[idx] == 0:
                    continue
                else:
                    all_red = False
                    break
            if all_red:
                red_violations += 1
        
        # Check for blue K5: any 5 vertices where all 10 triples are blue
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                x, y, z = tri
                idx = None
                for t_idx, (a2, b2, c2) in enumerate(triples):
                    if (a2, b2, c2) == (x, y, z):
                        idx = t_idx
                        break
                if idx is not None and colors[idx] == 1:
                    continue
                else:
                    all_blue = False
                    break
            if all_blue:
                blue_violations += 1
        
        return red_violations, blue_violations
    
    # Use a faster violation counting approach
    def count_violations_fast(n, colors, triples, tri_index):
        """Count violations using precomputed triple indices."""
        red_violations = 0
        blue_violations = 0
        
        # For red K4: check all 4-sets
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            tri_indices = [
                tri_index[(a,b,c)], tri_index[(a,b,d)],
                tri_index[(a,c,d)], tri_index[(b,c,d)]
            ]
            if all(colors[i] == 0 for i in tri_indices):
                red_violations += 1
        
        # For blue K5: check all 5-sets
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                idx = tri_index[tri]
                if colors[idx] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_violations += 1
        
        return red_violations, blue_violations
    
    def build_triple_index(n):
        """Map triple tuple to its index in the triples list."""
        triples = get_triples(n)
        return {t: i for i, t in enumerate(triples)}, triples
    
    def local_search(n, max_iterations=5000):
        """Perform local search to find a valid coloring."""
        triples, vertex_to_triples = build_adjacency(n)
        tri_index, _ = build_triple_index(n)
        num_triples = len(triples)
        
        # Start with random coloring
        colors = np.random.randint(0, 2, size=num_triples).tolist()
        
        best_colors = colors[:]
        best_score = float('inf')
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            red_v, blue_v = count_violations_fast(n, colors, triples, tri_index)
            score = red_v * 10 + blue_v  # Weight blue violations more heavily
            
            if score < best_score:
                best_score = score
                best_colors = colors[:]
            
            if score == 0:
                break
            
            # Find a triple to flip that reduces violations
            # Try flipping random triples
            improved = False
            for _ in range(100):
                idx = random.randint(0, num_triples - 1)
                old_color = colors[idx]
                colors[idx] = 1 - old_color
                
                new_red_v, new_blue_v = count_violations_fast(n, colors, triples, tri_index)
                new_score = new_red_v * 10 + new_blue_v
                
                if new_score < score:
                    score = new_score
                    improved = True
                    break
                else:
                    colors[idx] = old_color
            
            if not improved:
                # Accept some worse moves to escape local minima
                if random.random() < 0.1:
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
                else:
                    break
        
        return best_colors, best_score
    
    # Try to find valid colorings for increasing n
    best_n = 9
    best_colors = None
    
    # Start with the known valid n=9 coloring
    triples9 = get_triples(9)
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples9]
    best_n = 9
    
    # Try to extend to larger n using local search
    for target_n in range(10, 25):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        found_valid = False
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            colors, score = local_search(target_n, max_iterations=3000)
            
            if score == 0:
                found_valid = True
                best_n = target_n
                best_colors = colors
                break
        
        if found_valid:
            # Continue trying to go higher
            continue
        else:
            # If we couldn't find valid coloring, stop
            break
    
    # Return the best result found
    triples_final = get_triples(best_n)
    # Ensure we have the right number of colors
    if len(best_colors) != len(triples_final):
        # Regenerate
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_final]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
