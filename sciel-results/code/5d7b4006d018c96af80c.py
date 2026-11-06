"""Hypergraph Ramsey R(4,5;3) - colouring triples without red K4^(3) or blue K5^(3)."""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def solve_n(n, time_limit):
        """Try to find a valid coloring for n vertices using local search with incremental updates."""
        # Build triples in lex order
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # For each triple, find all 4-sets that contain it
        triple_to_k4 = [[] for _ in range(num_triples)]
        for s in combinations(range(n), 4):
            for a in range(4):
                for b in range(a+1, 4):
                    for c in range(b+1, 4):
                        t = (s[a], s[b], s[c])
                        triple_to_k4[triple_idx[t]].append(s)
        
        # For each triple, find all 5-sets that contain it
        triple_to_k5 = [[] for _ in range(num_triples)]
        for s in combinations(range(n), 5):
            for a in range(5):
                for b in range(a+1, 5):
                    for c in range(b+1, 5):
                        t = (s[a], s[b], s[c])
                        triple_to_k5[triple_idx[t]].append(s)
        
        # For each 4-set, count how many red triples it has
        # For each 5-set, count how many blue triples it has
        k4_red_count = {}
        for s in combinations(range(n), 4):
            k4_red_count[s] = 0
        
        k5_blue_count = {}
        for s in combinations(range(n), 5):
            k5_blue_count[s] = 0
        
        # Initialize colors randomly
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        # Initialize counts
        for i, t in enumerate(triples):
            if colors[i] == 0:  # red
                for s in triple_to_k4[i]:
                    k4_red_count[s] += 1
            else:  # blue
                for s in triple_to_k5[i]:
                    k5_blue_count[s] += 1
        
        # Count violations
        def count_violations():
            red_k4 = sum(1 for s, c in k4_red_count.items() if c == 4)
            blue_k5 = sum(1 for s, c in k5_blue_count.items() if c == 10)
            return red_k4 + blue_k5
        
        best_colors = colors.copy()
        best_violations = count_violations()
        
        # Local search
        while time.time() < time_limit and best_violations > 0:
            # Try flipping a random triple
            t_idx = random.randint(0, num_triples - 1)
            old_val = colors[t_idx]
            new_val = 1 - old_val
            t = triples[t_idx]
            
            # Compute delta in violations
            delta = 0
            
            if new_val == 0:  # changing to red
                # Decrease blue count for 5-sets, increase red count for 4-sets
                for s in triple_to_k5[t_idx]:
                    old_c = k5_blue_count[s]
                    if old_c == 10:
                        delta += 1  # was a violation
                    elif old_c == 9:
                        delta -= 1  # was not, becomes... no, decreasing blue count
                    # Actually, we're decreasing blue count, so:
                    # if old_c was 10, we remove a violation: delta -= 1
                    # if old_c was 10, after decrement it's 9, no violation
                    # This is getting complicated. Let me just recompute for now.
                
                # Let's just flip and recompute violations for simplicity
                # This is O(num_k4 + num_k5) per flip, which is too slow.
                pass
            else:  # changing to blue
                pass
            
            # For now, just flip and recompute
            colors[t_idx] = new_val
            
            # Update counts
            if new_val == 0:  # now red
                for s in triple_to_k4[t_idx]:
                    k4_red_count[s] += 1
                for s in triple_to_k5[t_idx]:
                    k5_blue_count[s] -= 1
            else:  # now blue
                for s in triple_to_k4[t_idx]:
                    k4_red_count[s] -= 1
                for s in triple_to_k5[t_idx]:
                    k5_blue_count[s] += 1
            
            new_violations = count_violations()
            
            if new_violations < best_violations:
                best_colors = colors.copy()
                best_violations = new_violations
                if best_violations == 0:
                    break
            else:
                # Revert
                colors[t_idx] = old_val
                if old_val == 0:  # was red
                    for s in triple_to_k4[t_idx]:
                        k4_red_count[s] -= 1
                    for s in triple_to_k5[t_idx]:
                        k5_blue_count[s] += 1
                else:  # was blue
                    for s in triple_to_k4[t_idx]:
                        k4_red_count[s] += 1
                    for s in triple_to_k5[t_idx]:
                        k5_blue_count[s] -= 1
        
        return best_colors, best_violations == 0
    
    # Try n=34 first
    if time.time() < deadline - 5:
        colors_34, valid_34 = solve_n(34, deadline - 3)
        if valid_34:
            return {'n': 34, 'colors': colors_34.tolist()}
    
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-END
