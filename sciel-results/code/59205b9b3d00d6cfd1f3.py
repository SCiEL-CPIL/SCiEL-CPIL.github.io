"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    target_n = 34
    
    # Precompute all triples in lexicographic order
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # Map triple -> index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
    
    # For each 4-set, the 4 triples that form K4^(3)
    k4_triple_indices = []
    for quad in combinations(range(target_n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_triple_indices.append(idxs)
    
    # For each 5-set, the 10 triples that form K5^(3)
    k5_triple_indices = []
    for quint in combinations(range(target_n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_triple_indices.append(idxs)
    
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    # Build reverse maps: triple -> which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
    
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)
    
    def compute_penalty(colors):
        """Compute total penalty: count of red K4 violations + blue K5 violations."""
        penalty = 0
        for k4_i, idxs in enumerate(k4_triple_indices):
            red_count = sum(1 for idx in idxs if colors[idx] == 0)
            if red_count == 4:
                penalty += 1
        for k5_i, idxs in enumerate(k5_triple_indices):
            blue_count = sum(1 for idx in idxs if colors[idx] == 1)
            if blue_count == 10:
                penalty += 1
        return penalty
    
    def local_search(colors, max_steps=50000):
        """Local search with single-flip moves, accepting improving or neutral moves."""
        current_penalty = compute_penalty(colors)
        best_penalty = current_penalty
        best_colors = colors[:]
        
        no_improve = 0
        max_no_improve = 5000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick a random triple to flip
            t = random.randrange(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            # Compute delta penalty
            delta = 0
            if old_color == 0:  # red -> blue
                for k4_i in triple_to_k4s[t]:
                    # This triple was red, now becomes blue
                    # Check if this K4 was a red violation (all 4 red)
                    red_count = 0
                    for idx in k4_triple_indices[k4_i]:
                        if idx == t:
                            pass  # this one is changing
                        elif colors[idx] == 0:
                            red_count += 1
                    # After flip, red_count stays the same for other triples
                    # Before: if all 4 were red (including t), red_count=4 -> violation
                    # After: t is blue, so red_count <= 3 -> no violation
                    if red_count == 3:  # means all other 3 are red, t was red => was violation
                        delta -= 1
                for k5_i in triple_to_k5s[t]:
                    # This triple was red, now becomes blue
                    blue_count = 0
                    for idx in k5_triple_indices[k5_i]:
                        if idx == t:
                            pass
                        elif colors[idx] == 1:
                            blue_count += 1
                    # After flip, blue_count + 1
                    if blue_count == 9:  # means other 9 are blue, t becomes blue => violation
                        delta += 1
            else:  # blue -> red
                for k4_i in triple_to_k4s[t]:
                    red_count = 0
                    for idx in k4_triple_indices[k4_i]:
                        if idx == t:
                            pass
                        elif colors[idx] == 0:
                            red_count += 1
                    # After flip, red_count + 1
                    if red_count == 3:  # means other 3 are red, t becomes red => violation
                        delta += 1
                for k5_i in triple_to_k5s[t]:
                    blue_count = 0
                    for idx in k5_triple_indices[k5_i]:
                        if idx == t:
                            pass
                        elif colors[idx] == 1:
                            blue_count += 1
                    # Before: if all 10 were blue (including t), blue_count=10 -> violation
                    # After: t is red, so blue_count <= 9 -> no violation
                    if blue_count == 9:  # means all other 9 are blue, t was blue => was violation
                        delta -= 1
            
            new_penalty = current_penalty + delta
            
            # Accept if improving or neutral (with some randomness for exploration)
            if delta <= 0 or (delta == 0 and random.random() < 0.3):
                colors[t] = new_color
                current_penalty = new_penalty
                no_improve += 1
            else:
                no_improve += 1
            
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colors = colors[:]
                no_improve = 0
            
            # Early exit if valid
            if best_penalty == 0:
                break
        
        return best_penalty, best_colors
    
    # Strategy: multiple random initializations + local search
    best_overall_penalty = float('inf')
    best_overall_colors = None
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create random initial coloring with slight bias
        p_red = 0.5
        colors = [1 if random.random() > p_red else 0 for _ in range(num_triples)]
        
        # Run local search
        penalty, colors = local_search(colors, max_steps=30000)
        
        if penalty < best_overall_penalty:
            best_overall_penalty = penalty
            best_overall_colors = colors[:]
        
        if best_overall_penalty == 0:
            break
    
    if best_overall_penalty == 0 and best_overall_colors is not None:
        return {'n': target_n, 'colors': best_overall_colors}
    
    # If local search didn't find valid, try a more aggressive approach:
    # Start from best found and do more iterations
    if best_overall_colors is not None:
        penalty, colors = local_search(best_overall_colors, max_steps=100000)
        if penalty < best_overall_penalty:
            best_overall_penalty = penalty
            best_overall_colors = colors[:]
    
    if best_overall_penalty == 0 and best_overall_colors is not None:
        return {'n': target_n, 'colors': best_overall_colors}
    
    # Fallback: try smaller n
    for n in range(33, 8, -1):
        if time.time() > deadline:
            break
        triples_n = list(combinations(range(n), 3))
        num_triples_n = len(triples_n)
        triple_index_n = {}
        for i, t in enumerate(triples_n):
            triple_index_n[t] = i
        
        k4_triple_indices_n = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            idxs = [
                triple_index_n[(a, b, c)],
                triple_index_n[(a, b, d)],
                triple_index_n[(a, c, d)],
                triple_index_n[(b, c, d)]
            ]
            k4_triple_indices_n.append(idxs)
        
        k5_triple_indices_n = []
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            idxs = [
                triple_index_n[(a, b, c)],
                triple_index_n[(a, b, d)],
                triple_index_n[(a, b, e)],
                triple_index_n[(a, c, d)],
                triple_index_n[(a, c, e)],
                triple_index_n[(a, d, e)],
                triple_index_n[(b, c, d)],
                triple_index_n[(b, c, e)],
                triple_index_n[(b, d, e)],
                triple_index_n[(c, d, e)]
            ]
            k5_triple_indices_n.append(idxs)
        
        # Simple greedy for smaller n
        colors_n = [0] * num_triples_n
        valid = True
        for tri_idx in range(num_triples_n):
            # Try both colors
            for test_color in [0, 1]:
                colors_n[tri_idx] = test_color
                # Check if this creates a violation
                violation = False
                if test_color == 0:  # red
                    for k4_i, idxs in enumerate(k4_triple_indices_n):
                        if tri_idx in idxs:
                            red_count = sum(1 for idx in idxs if colors_n[idx] == 0)
                            if red_count == 4:
                                violation = True
                                break
                else:  # blue
                    for k5_i, idxs in enumerate(k5_triple_indices_n):
                        if tri_idx in idxs:
                            blue_count = sum(1 for idx in idxs if colors_n[idx] == 1)
                            if blue_count == 10:
                                violation = True
                                break
                if not violation:
                    break
                else:
                    # Try other color
                    colors_n[tri_idx] = 1 - test_color
                    violation2 = False
                    if colors_n[tri_idx] == 0:
                        for k4_i, idxs in enumerate(k4_triple_indices_n):
                            if tri_idx in idxs:
                                red_count = sum(1 for idx in idxs if colors_n[idx] == 0)
                                if red_count == 4:
                                    violation2 = True
                                    break
                    else:
                        for k5_i, idxs in enumerate(k5_triple_indices_n):
                            if tri_idx in idxs:
                                blue_count = sum(1 for idx in idxs if colors_n[idx] == 1)
                                if blue_count == 10:
                                    violation2 = True
                                    break
                    if violation2:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            # Verify
            is_valid = True
            for k4_i, idxs in enumerate(k4_triple_indices_n):
                red_count = sum(1 for idx in idxs if colors_n[idx] == 0)
                if red_count == 4:
                    is_valid = False
                    break
            if is_valid:
                for k5_i, idxs in enumerate(k5_triple_indices_n):
                    blue_count = sum(1 for idx in idxs if colors_n[idx] == 1)
                    if blue_count == 10:
                        is_valid = False
                        break
            if is_valid:
                return {'n': n, 'colors': colors_n}
    
    # Final fallback: very small construction
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
