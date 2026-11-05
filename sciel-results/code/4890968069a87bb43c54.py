import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def verify(colors, n):
        """
        Verify if a coloring is valid.
        colors: list of 0/1 for triples in lex order.
        Returns True if valid (no red K4, no blue K5).
        """
        triple_index = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_index[(a,b,c)] = idx
                    idx += 1
        
        # Check Red K4
        for quad in combinations(range(n), 4):
            all_red = True
            for triple in combinations(quad, 3):
                if colors[triple_index[triple]] != 0:
                    all_red = False
                    break
            if all_red:
                return False
        
        # Check Blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_index[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
                
        return True

    def solve_structured(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # Precompute membership
        # triple_to_4sets[t_idx] = list of 4-set indices containing this triple
        # triple_to_5sets[t_idx] = list of 5-set indices containing this triple
        triple_to_4sets = [[] for _ in range(num_triples)]
        triple_to_5sets = [[] for _ in range(num_triples)]
        
        # Also need set_to_triples for quick iteration over sets?
        # For local search, we need to know which triples are in a specific set.
        # 4-set i has 4 triples. 5-set j has 10 triples.
        set_4_to_triples = [[] for _ in range(num_4sets)]
        set_5_to_triples = [[] for _ in range(num_5sets)]
        
        for i, s4 in enumerate(four_sets):
            for t in combinations(s4, 3):
                idx = triple_index[t]
                triple_to_4sets[idx].append(i)
                set_4_to_triples[i].append(idx)
                
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                idx = triple_index[t]
                triple_to_5sets[idx].append(i)
                set_5_to_triples[i].append(idx)

        def get_violations(colors):
            """
            Returns (red_violation_sets, blue_violation_sets, total_violations)
            red_violation_sets: list of indices of 4-sets that are all red
            blue_violation_sets: list of indices of 5-sets that are all blue
            """
            red_viol = []
            for i in range(num_4sets):
                # Check if all 4 triples in set i are red (0)
                is_red = True
                for t_idx in set_4_to_triples[i]:
                    if colors[t_idx] != 0:
                        is_red = False
                        break
                if is_red:
                    red_viol.append(i)
            
            blue_viol = []
            for j in range(num_5sets):
                # Check if all 10 triples in set j are blue (1)
                is_blue = True
                for t_idx in set_5_to_triples[j]:
                    if colors[t_idx] != 1:
                        is_blue = False
                        break
                if is_blue:
                    blue_viol.append(j)
                    
            return red_viol, blue_viol, len(red_viol) + len(blue_viol)

        def repair_step(colors, red_viol, blue_viol):
            """
            Try to fix one violation by flipping a triple.
            Returns True if a violation was fixed (or reduced), False otherwise.
            Strategy: 
            1. Pick a random violated 4-set or 5-set.
            2. Try flipping one of its triples to see if it reduces total violations.
               - For Red K4 (all 0), flipping one to 1 breaks the K4. 
                 Risk: might create a Blue K5.
               - For Blue K5 (all 1), flipping one to 0 breaks the K5.
                 Risk: might create a Red K4.
            """
            if not red_viol and not blue_viol:
                return True
            
            # Prioritize larger violations? Or random?
            # Let's pick a random violation to avoid bias
            if red_viol and (not blue_viol or random.random() < 0.5):
                # Pick a red violation
                s4_idx = random.choice(red_viol)
                t_indices = set_4_to_triples[s4_idx]
                # Shuffle to try different ones
                random.shuffle(t_indices)
                
                for t_idx in t_indices:
                    # Flip t_idx to Blue (1)
                    old_color = colors[t_idx]
                    if old_color == 0:
                        colors[t_idx] = 1
                        # Check if this created a Blue K5
                        # We only need to check 5-sets containing t_idx
                        created_blue_viol = False
                        for s5_idx in triple_to_5sets[t_idx]:
                            # Check if s5_idx is now all blue
                            is_blue = True
                            for tt in set_5_to_triples[s5_idx]:
                                if colors[tt] != 1:
                                    is_blue = False
                                    break
                            if is_blue:
                                created_blue_viol = True
                                break
                        
                        if created_blue_viol:
                            # Revert
                            colors[t_idx] = 0
                            continue
                        
                        # If no new blue violation, we fixed a red one and didn't create a blue one.
                        # Note: We might have created a *different* red violation? No, we turned 0->1, so red counts decrease.
                        # So this is a pure improvement (or neutral).
                        return True
            else:
                # Pick a blue violation
                s5_idx = random.choice(blue_viol)
                t_indices = set_5_to_triples[s5_idx]
                random.shuffle(t_indices)
                
                for t_idx in t_indices:
                    # Flip t_idx to Red (0)
                    old_color = colors[t_idx]
                    if old_color == 1:
                        colors[t_idx] = 0
                        # Check if this created a Red K4
                        created_red_viol = False
                        for s4_idx in triple_to_4sets[t_idx]:
                            # Check if s4_idx is now all red
                            is_red = True
                            for tt in set_4_to_triples[s4_idx]:
                                if colors[tt] != 0:
                                    is_red = False
                                    break
                            if is_red:
                                created_red_viol = True
                                break
                        
                        if created_red_viol:
                            # Revert
                            colors[t_idx] = 1
                            continue
                        
                        return True
            
            return False

        # --- Main Search Loop ---
        
        # 1. Structured Initialization: Latin Square based
        # If n is close to a prime power, we can use a finite field.
        # For general n, we use a modular arithmetic pattern.
        # Color (a,b,c) based on (a+b+c) mod 2? No, too regular.
        # Let's use: color = 0 if (a*b + b*c + c*a) % 2 == 0 else 1?
        # Or simply: random, but seeded.
        
        best_colors = None
        best_viol = float('inf')
        
        # Try multiple restarts
        max_restarts = 50
        time_per_restart = (time_limit - 1.0) / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Initialize
            # Use a mix of random and structured
            # Structured: color 0 if (a+b+c) % 3 == 0 else 1? 
            # Let's try a few patterns
            
            pattern = r % 3
            colors = np.zeros(num_triples, dtype=np.int8)
            
            if pattern == 0:
                # Random
                np.random.seed(42 + r)
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            elif pattern == 1:
                # Parity of sum
                for i, (a,b,c) in enumerate(triples):
                    colors[i] = (a + b + c) % 2
            else:
                # Parity of product?
                for i, (a,b,c) in enumerate(triples):
                    colors[i] = (a * b + b * c + c * a) % 2
            
            # Local Search / Repair
            # Run repair until no progress or time slice ends
            start_r = time.time()
            no_progress_count = 0
            
            while time.time() - start_r < time_per_restart:
                red_viol, blue_viol, total_viol = get_violations(colors)
                
                if total_viol == 0:
                    return colors.tolist()
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    no_progress_count = 0
                else:
                    no_progress_count += 1
                
                # If we made no progress in the last 100 steps, break to restart
                if no_progress_count > 200:
                    break
                
                # Try to repair
                improved = repair_step(colors, red_viol, blue_viol)
                if not improved:
                    # If repair failed, do a random flip to escape local minimum
                    # Pick a random triple and flip it
                    t_idx = np.random.randint(0, num_triples)
                    colors[t_idx] = 1 - colors[t_idx]
                    
                    # Check if this helped?
                    # We'll just let the loop re-evaluate.
                    # But if we keep failing to repair, we might loop.
                    # The no_progress_count handles this.
                    # However, a random flip might increase violations.
                    # That's okay for exploration.
                    
        # End of restarts
        
        # Final check
        if best_colors is not None:
            red_viol, blue_viol, total_viol = get_violations(best_colors)
            if total_viol == 0:
                return best_colors.tolist()
        
        return None

    # Try n from 34 downwards
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    
    # Time allocation
    # We have ~115s.
    # 34: 30s
    # 33: 20s
    # 32: 15s
    # 31: 10s
    # 30: 10s
    # 29-20: 1s each (10s)
    # Total: 30+20+15+10+10+10 = 95s. Safe.
    
    time_alloc = {
        34: 30.0,
        33: 20.0,
        32: 15.0,
        31: 10.0,
        30: 10.0,
    }
    
    for n in n_targets:
        if time.time() > deadline - 2.0:
            break
            
        t_limit = time_alloc.get(n, 1.0)
        time_limit = time.time() + t_limit
        if time_limit > deadline - 2.0:
            time_limit = deadline - 2.0
            
        res = solve_structured(n, time_limit)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback: n=4
    # Valid coloring for n=4:
    # Triples: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    # [0, 1, 1, 1] is valid.
    return {'n': 4, 'colors': [0, 1, 1, 1]}

# Test
if __name__ == "__main__":
    res = construct()
    print(res['n'])
