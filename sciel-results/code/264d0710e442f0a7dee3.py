"""
Hypergraph Ramsey R(4,5;3) construction via structured initialization + targeted repair.

Strategy:
1. Use algebraic/structured initialization based on modular patterns.
2. Use targeted local search to fix violations (focus on most violated K4s/K5s).
3. Try multiple structured initializations and pick the best.
4. Return the largest n with a valid coloring found within time budget.
"""

import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 140.0

def construct():
    global _DEADLINE
    start_time = time.time()
    
    best_n = 0
    best_colors = None
    
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    def build_index_map(n):
        triples = get_triples(n)
        idx_map = {}
        for i, (a, b, c) in enumerate(triples):
            idx_map[(a, b, c)] = i
        return triples, idx_map
    
    def structured_coloring(n, pattern='mod3'):
        """Generate a structured initial coloring."""
        triples = get_triples(n)
        colors = np.zeros(len(triples), dtype=np.int8)
        
        for i, (a, b, c) in enumerate(triples):
            if pattern == 'mod3':
                # Color based on (a+b+c) mod 3
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1
            elif pattern == 'mod5':
                # Color based on (a+2*b+3*c) mod 5
                colors[i] = 0 if (a + 2*b + 3*c) % 5 < 2 else 1
            elif pattern == 'xor':
                # Color based on XOR of bits
                x = a ^ b ^ c
                colors[i] = 0 if x % 2 == 0 else 1
            elif pattern == 'product':
                # Color based on product modulo
                colors[i] = 0 if (a * b * c) % 7 < 3 else 1
            elif pattern == 'random':
                colors[i] = random.randint(0, 1)
            elif pattern == 'greedy':
                # Greedy: try to avoid red K4 and blue K5
                # Start with all red, flip to blue if it creates a red K4
                colors[i] = 0  # default red
        
        return colors
    
    def count_violations(colors, n, triples, idx_map):
        """Count red K4s and blue K5s."""
        red_k4_count = 0
        blue_k5_count = 0
        
        # Check all K4s for red
        for quad in combinations(range(n), 4):
            all_red = True
            for tri in combinations(quad, 3):
                if colors[idx_map[tri]] != 0:
                    all_red = False
                    break
            if all_red:
                red_k4_count += 1
        
        # Check all K5s for blue
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                if colors[idx_map[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5_count += 1
        
        return red_k4_count, blue_k5_count
    
    def is_valid(colors, n, triples, idx_map):
        red_k4, blue_k5 = count_violations(colors, n, triples, idx_map)
        return red_k4 == 0 and blue_k5 == 0
    
    def targeted_repair(colors, n, triples, idx_map, max_iter=50000):
        """
        Targeted repair: focus on violating K4s and K5s.
        For each violating K4 (all red), flip one triple to blue.
        For each violating K5 (all blue), flip one triple to red.
        Choose flips that minimize new violations.
        """
        if time.time() > _DEADLINE:
            return None
        
        triples_list, idx_map_local = build_index_map(n)
        num_triples = len(triples_list)
        
        # Precompute which K4s and K5s contain each triple
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for k4_idx, quad in enumerate(k4_list):
            for tri in combinations(quad, 3):
                triple_to_k4s[idx_map_local[tri]].append(k4_idx)
        
        for k5_idx, quint in enumerate(k5_list):
            for tri in combinations(quint, 3):
                triple_to_k5s[idx_map_local[tri]].append(k5_idx)
        
        # Track violation counts
        k4_red_counts = np.zeros(len(k4_list), dtype=np.int32)
        k5_blue_counts = np.zeros(len(k5_list), dtype=np.int32)
        
        def recompute_counts():
            nonlocal k4_red_counts, k5_blue_counts
            k4_red_counts[:] = 0
            k5_blue_counts[:] = 0
            for i, (a, b, c) in enumerate(triples_list):
                if colors[i] == 0:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] += 1
                else:
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] += 1
        
        recompute_counts()
        
        def current_violations():
            return int(np.sum(k4_red_counts == 4)) + int(np.sum(k5_blue_counts == 10))
        
        best_colors = colors.copy()
        best_violations = current_violations()
        
        for it in range(max_iter):
            if time.time() > _DEADLINE:
                break
            
            # Find all violating K4s and K5s
            violating_k4s = np.where(k4_red_counts == 4)[0]
            violating_k5s = np.where(k5_blue_counts == 10)[0]
            
            if len(violating_k4s) == 0 and len(violating_k5s) == 0:
                return best_colors
            
            # Choose a random violating structure
            if len(violating_k4s) > 0 and (len(violating_k5s) == 0 or random.random() < 0.5):
                # Pick a violating K4
                k4_idx = violating_k4s[random.randint(0, len(violating_k4s) - 1)]
                quad = k4_list[k4_idx]
                # Pick a triple in this K4 to flip (red -> blue)
                tri_options = list(combinations(quad, 3))
                tri = tri_options[random.randint(0, len(tri_options) - 1)]
                tri_idx = idx_map_local[tri]
                old_color = colors[tri_idx]
                if old_color != 0:
                    continue  # Not red, skip
                new_color = 1
            else:
                # Pick a violating K5
                k5_idx = violating_k5s[random.randint(0, len(violating_k5s) - 1)]
                quint = k5_list[k5_idx]
                # Pick a triple in this K5 to flip (blue -> red)
                tri_options = list(combinations(quint, 3))
                tri = tri_options[random.randint(0, len(tri_options) - 1)]
                tri_idx = idx_map_local[tri]
                old_color = colors[tri_idx]
                if old_color != 1:
                    continue  # Not blue, skip
                new_color = 0
            
            # Compute delta in violations if we flip
            delta = 0
            
            # Effect on K4s containing this triple
            for k4_i in triple_to_k4s[tri_idx]:
                if old_color == 0:  # red -> blue
                    if k4_red_counts[k4_i] == 4:
                        delta -= 1
                else:  # blue -> red
                    if k4_red_counts[k4_i] == 3:
                        delta += 1
            
            # Effect on K5s containing this triple
            for k5_i in triple_to_k5s[tri_idx]:
                if old_color == 1:  # blue -> red
                    if k5_blue_counts[k5_i] == 10:
                        delta -= 1
                else:  # red -> blue
                    if k5_blue_counts[k5_i] == 9:
                        delta += 1
            
            new_violations = best_violations + delta
            
            # Accept if it doesn't make things worse, or with some probability
            if new_violations <= best_violations or random.random() < np.exp(-max(0, delta) / 0.5):
                # Apply flip
                colors[tri_idx] = new_color
                for k4_i in triple_to_k4s[tri_idx]:
                    if old_color == 0:
                        k4_red_counts[k4_i] -= 1
                    else:
                        k4_red_counts[k4_i] += 1
                for k5_i in triple_to_k5s[tri_idx]:
                    if old_color == 1:
                        k5_blue_counts[k5_i] -= 1
                    else:
                        k5_blue_counts[k5_i] += 1
                
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
        
        return best_colors
    
    # Try different n values with different initialization patterns
    patterns = ['mod3', 'mod5', 'xor', 'product', 'random']
    
    for n in range(20, 36):
        if time.time() > _DEADLINE:
            break
        
        found = False
        for pattern in patterns:
            if time.time() > _DEADLINE:
                break
            
            colors = structured_coloring(n, pattern=pattern)
            triples, idx_map = build_index_map(n)
            
            # Check if already valid
            if is_valid(colors, n, triples, idx_map):
                if n > best_n:
                    best_n = n
                    best_colors = colors.tolist()
                found = True
                break
            
            # Try targeted repair
            repaired = targeted_repair(colors.copy(), n, triples, idx_map, max_iter=20000)
            if repaired is not None:
                if is_valid(repaired, n, triples, idx_map):
                    if n > best_n:
                        best_n = n
                        best_colors = repaired.tolist()
                    found = True
                    break
        
        if found:
            continue
        else:
            # Try a few more n values in case of bad luck
            continue
    
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return a small valid coloring
    n = 9
    triples = get_triples(n)
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
