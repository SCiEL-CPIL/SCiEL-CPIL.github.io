"""Simulated annealing for 3-uniform hypergraph Ramsey R(4,5;3)."""
import time
import random
import numpy as np
from itertools import combinations

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 130.0  # Leave 20s safety margin before 150s kill

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Precompute all triples and their indices
    def get_triples(n):
        triples = list(combinations(range(n), 3))
        return triples
    
    def triple_index(a, b, c):
        # Lexicographic index of triple (a,b,c) where a<b<c
        # For n vertices, number of triples starting with vertex i:
        # triples with first element < a: sum_{i=0}^{a-1} C(n-1-i, 2)
        # triples with first element = a and second < b: (b - a - 1)
        # triples with first=a, second=b, third<c: (c - b - 1)
        # We'll use a lookup for speed
        pass
    
    # Build index lookup for triples
    def build_triple_index(n):
        triples = list(combinations(range(n), 3))
        idx = {}
        for i, (a, b, c) in enumerate(triples):
            idx[(a, b, c)] = i
        return idx
    
    def count_violations(colors, n, triples, idx):
        """Count red K4s (4 vertices all 4 triples red) and blue K5s (5 vertices all 10 triples blue)."""
        red_k4 = 0
        blue_k5 = 0
        
        # Check all 4-sets for red K4
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[idx[(a,b,c)]] == 0 and 
                colors[idx[(a,b,d)]] == 0 and 
                colors[idx[(a,c,d)]] == 0 and 
                colors[idx[(b,c,d)]] == 0):
                red_k4 += 1
        
        # Check all 5-sets for blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for combo in combinations(quint, 3):
                if colors[idx[combo]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5 += 1
        
        return red_k4, blue_k5, red_k4 + blue_k5
    
    def incremental_violation_change(colors, n, triples, idx, flip_triple_idx):
        """
        When we flip triple t from color c to 1-c, we need to check:
        - For red K4s: any 4-set containing all vertices of t that was previously red K4 
          but now has t flipped (so no longer red K4), or was not red K4 but now is.
        - For blue K5s: similar for 5-sets.
        
        Actually, let's think differently. When we flip triple (a,b,c):
        - Red K4s affected: 4-sets {a,b,c,d} for each d not in {a,b,c}.
          Before flip: if all 4 triples in {a,b,c,d} were red, it was a red K4.
          After flip: triple (a,b,c) changes color.
          So if (a,b,c) was red and becomes blue: red K4 count decreases by 1 for each d
             where (a,b,d),(a,c,d),(b,c,d) were all red.
          If (a,b,c) was blue and becomes red: red K4 count increases by 1 for each d
             where (a,b,d),(a,c,d),(b,c,d) were all red.
        
        - Blue K5s affected: 5-sets {a,b,c,d,e} for each pair {d,e} not in {a,b,c}.
          Before flip: if all 10 triples in the 5-set were blue, it was a blue K5.
          After flip: triple (a,b,c) changes.
          If (a,b,c) was blue and becomes red: blue K5 count decreases by 1 for each {d,e}
             where all other 9 triples in {a,b,c,d,e} were blue.
          If (a,b,c) was red and becomes blue: blue K5 count increases by 1 for each {d,e}
             where all other 9 triples in {a,b,c,d,e} were blue.
        """
        a, b, c = triples[flip_triple_idx]
        old_color = colors[flip_triple_idx]
        new_color = 1 - old_color
        
        delta_red_k4 = 0
        delta_blue_k5 = 0
        
        # Red K4s: check 4-sets {a,b,c,d} for d not in {a,b,c}
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            # Check if (a,b,d), (a,c,d), (b,c,d) are all red
            if (colors[idx[(a,b,d)]] == 0 and 
                colors[idx[(a,c,d)]] == 0 and 
                colors[idx[(b,c,d)]] == 0):
                if old_color == 0:
                    # Was red, now blue -> red K4 broken
                    delta_red_k4 -= 1
                else:
                    # Was blue, now red -> red K4 formed
                    delta_red_k4 += 1
        
        # Blue K5s: check 5-sets {a,b,c,d,e} for pairs {d,e} not in {a,b,c}
        others = [v for v in range(n) if v != a and v != b and v != c]
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                d, e = others[i], others[j]
                # Check if all other 9 triples in {a,b,c,d,e} are blue
                # Triples: (a,b,c) [this one], (a,b,d), (a,b,e), (a,c,d), (a,c,e),
                #          (b,c,d), (b,c,e), (d,e,a), (d,e,b), (d,e,c)
                # Wait, need all 10 triples from 5 vertices
                quint = (a, b, c, d, e)
                all_others_blue = True
                for combo in combinations(quint, 3):
                    if combo == (a,b,c):
                        continue
                    if colors[idx[combo]] != 1:
                        all_others_blue = False
                        break
                if all_others_blue:
                    if old_color == 1:
                        # Was blue, now red -> blue K5 broken
                        delta_blue_k5 -= 1
                    else:
                        # Was red, now blue -> blue K5 formed
                        delta_blue_k5 += 1
        
        return delta_red_k4, delta_blue_k5
    
    # Start with a reasonable n and random initialization
    best_n = 0
    best_colors = None
    
    # Try increasing n values
    for n in range(9, 35):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        triples = get_triples(n)
        idx = build_triple_index(n)
        num_triples = len(triples)
        
        # Initialize with random colors
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Count initial violations
        red_k4, blue_k5, total_viol = count_violations(colors, n, triples, idx)
        
        # Simulated annealing
        T = 2.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        max_steps = 500000
        
        while T > min_T and steps < max_steps:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Randomly pick a triple to flip
            t_idx = np.random.randint(0, num_triples)
            
            # Compute change in violations
            delta_red, delta_blue = incremental_violation_change(colors, n, triples, idx, t_idx)
            delta_total = delta_red + delta_blue
            
            if delta_total <= 0:
                # Accept (improves or equal)
                colors[t_idx] = 1 - colors[t_idx]
                red_k4 += delta_red
                blue_k5 += delta_blue
                total_viol = red_k4 + blue_k5
            else:
                # Accept with probability exp(-delta/T)
                if np.random.random() < np.exp(-delta_total / T):
                    colors[t_idx] = 1 - colors[t_idx]
                    red_k4 += delta_red
                    blue_k5 += delta_blue
                    total_viol = red_k4 + blue_k5
            
            steps += 1
            T *= cooling
        
        if total_viol == 0:
            if n > best_n:
                best_n = n
                best_colors = colors.tolist()
        
        # Try to extend: if we found valid coloring on n, try n+1
        if total_viol == 0:
            # We have a valid coloring on n vertices
            # Try to add vertex n by randomly coloring new triples
            for attempt in range(50):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                
                n_new = n + 1
                if n_new > 40:
                    break
                
                triples_new = get_triples(n_new)
                idx_new = build_triple_index(n_new)
                num_triples_new = len(triples_new)
                
                # Start from existing coloring, add random colors for new triples
                colors_new = np.zeros(num_triples_new, dtype=int)
                for i, (a, b, c) in enumerate(triples_new):
                    if c < n:  # Old triple
                        colors_new[i] = colors[idx[(a,b,c)]]
                    else:
                        colors_new[i] = np.random.randint(0, 2)
                
                # SA on new coloring
                T2 = 1.0
                cooling2 = 0.999
                min_T2 = 0.01
                steps2 = 0
                max_steps2 = 200000
                
                red_k4_2, blue_k5_2, total_viol_2 = count_violations(colors_new, n_new, triples_new, idx_new)
                
                while T2 > min_T2 and steps2 < max_steps2:
                    if time.time() - START_TIME > TIME_LIMIT:
                        break
                    
                    t_idx2 = np.random.randint(0, num_triples_new)
                    delta_red2, delta_blue2 = incremental_violation_change(colors_new, n_new, triples_new, idx_new, t_idx2)
                    delta_total2 = delta_red2 + delta_blue2
                    
                    if delta_total2 <= 0:
                        colors_new[t_idx2] = 1 - colors_new[t_idx2]
                        red_k4_2 += delta_red2
                        blue_k5_2 += delta_blue2
                        total_viol_2 = red_k4_2 + blue_k5_2
                    else:
                        if np.random.random() < np.exp(-delta_total2 / T2):
                            colors_new[t_idx2] = 1 - colors_new[t_idx2]
                            red_k4_2 += delta_red2
                            blue_k5_2 += delta_blue2
                            total_viol_2 = red_k4_2 + blue_k5_2
                    
                    steps2 += 1
                    T2 *= cooling2
                
                if total_viol_2 == 0:
                    if n_new > best_n:
                        best_n = n_new
                        best_colors = colors_new.tolist()
                    n = n_new
                    break
                else:
                    break
        
        if time.time() - START_TIME > TIME_LIMIT:
            break
    
    # Fallback: return the best we found, or a small valid one
    if best_n >= 9 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    else:
        # Return a small valid coloring
        n_small = 9
        triples_small = get_triples(n_small)
        colors_small = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_small]
        # Verify
        idx_small = build_triple_index(n_small)
        r, b, t = count_violations(np.array(colors_small), n_small, triples_small, idx_small)
        if t == 0:
            return {'n': n_small, 'colors': colors_small}
        else:
            # Very small fallback
            n_tiny = 5
            triples_tiny = get_triples(n_tiny)
            colors_tiny = [1] * len(triples_tiny)  # All blue, no red K4 possible, check blue K5
            # Actually all blue means the one 5-set is a blue K5, so invalid
            colors_tiny = [0] * len(triples_tiny)  # All red, check red K4: no 4-set can have all 4 red if n=5? 
            # For n=5, there are C(5,4)=5 four-sets. Each has 4 triples. All red -> each is a red K4. Invalid.
            # Let's just use the mod 3 one for n=5
            colors_tiny = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_tiny]
            return {'n': n_tiny, 'colors': colors_tiny}

