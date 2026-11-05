import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Helper to generate indices
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    # Precomputation for a specific n
    # We will rebuild these for each n as we increase n
    # But rebuilding is O(n^5) for 5-sets mapping, which might be slow for n=30.
    # C(30,5) = 142506. C(30,3) = 4060.
    # Mapping triple -> 5-sets: each triple is in C(27,2)=351 5-sets.
    # Total entries ~ 4060 * 351 ~ 1.4M. This is fine in Python if done once.
    
    def build_structure(n):
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # 4-sets
        sets_4 = list(combinations(range(n), 4))
        set4_triples = []
        ti = triple_index
        for s in sets_4:
            ts = [ti[(s[0], s[1], s[2])], ti[(s[0], s[1], s[3])], ti[(s[0], s[2], s[3])], ti[(s[1], s[2], s[3])]]
            set4_triples.append(ts)
            
        # 5-sets
        sets_5 = list(combinations(range(n), 5))
        set5_triples = []
        for s in sets_5:
            ts = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        ts.append(ti[(s[i], s[j], s[k])])
            set5_triples.append(ts)
            
        # Reverse mapping: triple -> list of set4 indices, triple -> list of set5 indices
        triple_to_set4 = [[] for _ in range(len(triples))]
        for i, ts in enumerate(set4_triples):
            for t_idx in ts:
                triple_to_set4[t_idx].append(i)
                
        triple_to_set5 = [[] for _ in range(len(triples))]
        for i, ts in enumerate(set5_triples):
            for t_idx in ts:
                triple_to_set5[t_idx].append(i)
                
        return {
            'n': n,
            'triples': triples,
            'set4_triples': set4_triples,
            'set5_triples': set5_triples,
            'triple_to_set4': triple_to_set4,
            'triple_to_set5': triple_to_set5,
            'num_triples': len(triples),
            'num_set4': len(sets_4),
            'num_set5': len(sets_5)
        }

    def count_violations(colors, struct):
        v4 = 0
        v5 = 0
        # Check 4-sets (Red K4)
        for ts in struct['set4_triples']:
            # All 4 must be 0
            if colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0 and colors[ts[3]] == 0:
                v4 += 1
        # Check 5-sets (Blue K5)
        for ts in struct['set5_triples']:
            # All 10 must be 1
            # Optimization: if any is 0, not a violation
            all_one = True
            for idx in ts:
                if colors[idx] == 0:
                    all_one = False
                    break
            if all_one:
                v5 += 1
        return v4 + v5

    def delta_violations(t_idx, new_color, colors, struct):
        """
        Calculate change in violations if triple t_idx is flipped to new_color.
        Assumes colors[t_idx] is currently the opposite of new_color.
        """
        old_color = colors[t_idx]
        if old_color == new_color:
            return 0
            
        d_v = 0
        
        # Update 4-sets containing t_idx
        for s4_idx in struct['triple_to_set4'][t_idx]:
            ts = struct['set4_triples'][s4_idx]
            # Count how many are 0 in this set
            # Before flip
            cnt0_before = 0
            for idx in ts:
                if idx == t_idx:
                    if old_color == 0: cnt0_before += 1
                else:
                    if colors[idx] == 0: cnt0_before += 1
            
            # After flip
            cnt0_after = cnt0_before
            if old_color == 0 and new_color == 1:
                cnt0_after -= 1
            elif old_color == 1 and new_color == 0:
                cnt0_after += 1
                
            # Violation if cnt0 == 4
            viol_before = 1 if cnt0_before == 4 else 0
            viol_after = 1 if cnt0_after == 4 else 0
            d_v += (viol_after - viol_before)
            
        # Update 5-sets containing t_idx
        for s5_idx in struct['triple_to_set5'][t_idx]:
            ts = struct['set5_triples'][s5_idx]
            # Count how many are 1 in this set
            cnt1_before = 0
            for idx in ts:
                if idx == t_idx:
                    if old_color == 1: cnt1_before += 1
                else:
                    if colors[idx] == 1: cnt1_before += 1
                    
            cnt1_after = cnt1_before
            if old_color == 1 and new_color == 0:
                cnt1_after -= 1
            elif old_color == 0 and new_color == 1:
                cnt1_after += 1
                
            # Violation if cnt1 == 10
            viol_before = 1 if cnt1_before == 10 else 0
            viol_after = 1 if cnt1_after == 10 else 0
            d_v += (viol_after - viol_before)
            
        return d_v

    best_n = 9
    best_colors = None
    
    # Start with a known valid small n to bootstrap?
    # Or just start random.
    
    # We will try to find valid colorings for n=10, 11, ...
    # Keep the best valid coloring found so far for the highest n.
    
    current_n = 10
    max_n_to_try = 30 # Don't try too high if time is short
    
    # Initial structure for n=10
    struct = build_structure(current_n)
    
    # Initial colors: Random
    colors = np.random.randint(0, 2, size=struct['num_triples']).astype(int)
    
    # We will run SA for each n. If we succeed, we increment n.
    # If we fail, we might try again with more time or different init.
    
    # To be safe, we will iterate n from 10 up.
    # If we find a valid coloring for n, we store it.
    # When moving to n+1, we extend the coloring.
    
    valid_coloring_for_n_minus_1 = None
    valid_n_minus_1 = 9 # Seed is valid for 9
    
    # Actually, let's just run a global search for a specific target N.
    # If we find 0 violations for N, return N.
    # If not, return best N found.
    
    # Strategy: Try N=15 first. If fail, try N=12? No, maximize N.
    # Better: Start with N=10. If valid, extend to 11. If valid, extend to 12...
    
    # Let's implement the extension logic.
    
    def extend_coloring(old_colors, old_struct, new_n):
        """
        Extend a valid coloring for old_n to new_n (old_n + 1).
        Returns initial colors for new_n.
        """
        old_n = old_struct['n']
        new_struct = build_structure(new_n)
        new_colors = np.zeros(new_struct['num_triples'], dtype=int)
        
        # Map old triples to new indices
        # Old triples are (a,b,c) with a<b<c < old_n
        # New triples include those with vertex new_n-1
        
        # We need to map old triple indices to new triple indices
        # This is tricky because the order of combinations changes.
        # Easier: Just create a dict for old triples.
        old_triple_set = {t: i for i, t in enumerate(old_struct['triples'])}
        
        for i, t in enumerate(new_struct['triples']):
            if t[2] < old_n: # Old triple
                new_colors[i] = old_colors[old_triple_set[t]]
            else:
                # New triple, randomize
                new_colors[i] = random.randint(0, 1)
                
        return new_colors, new_struct

    # Main Loop
    current_n = 10
    struct = build_structure(current_n)
    
    # Initial random colors
    colors = np.random.randint(0, 2, size=struct['num_triples']).astype(int)
    
    # We will try to improve `colors` for `current_n` until 0 violations or time up.
    # If 0 violations, we save it and try current_n + 1.
    
    found_valid_for_current_n = False
    best_valid_colors = None
    
    # To speed up, we'll run SA for a fixed number of steps or time slice per n.
    
    while time.time() < deadline - 5.0:
        if found_valid_for_current_n:
            # Move to next n
            current_n += 1
            if current_n > 35: break
            
            # Extend
            if best_valid_colors is not None:
                colors, struct = extend_coloring(best_valid_colors, struct, current_n)
            else:
                # Fallback to random if no previous valid
                struct = build_structure(current_n)
                colors = np.random.randint(0, 2, size=struct['num_triples']).astype(int)
                
            found_valid_for_current_n = False
            # Reset SA params for new n
            T = 1.0
            # Maybe lower T for larger n?
            
        # Simulated Annealing Step
        T = 0.05 # Fixed low temperature for greedy-ish search, or anneal?
        # Let's use a simple Hill Climbing with random restarts if stuck.
        
        # For efficiency, we'll do a loop of flips.
        # If no improvement in X steps, restart with random or perturbation.
        
        max_steps = 5000
        no_improve_count = 0
        current_viol = count_violations(colors, struct)
        
        if current_viol == 0:
            found_valid_for_current_n = True
            best_valid_colors = colors.copy()
            best_n = current_n
            continue
            
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
                
            # Pick a random triple to flip
            t_idx = random.randint(0, struct['num_triples'] - 1)
            new_color = 1 - colors[t_idx]
            
            d_v = delta_violations(t_idx, new_color, colors, struct)
            
            # Accept if better, or with prob exp(-d_v/T) if worse
            if d_v < 0:
                colors[t_idx] = new_color
                current_viol += d_v
                no_improve_count = 0
            else:
                # Simple greedy: only accept if d_v == 0 (neutral) or use SA
                # Let's use SA with T=0.1
                if random.random() < np.exp(-d_v / 0.1):
                    colors[t_idx] = new_color
                    current_viol += d_v
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            if current_viol == 0:
                found_valid_for_current_n = True
                best_valid_colors = colors.copy()
                best_n = current_n
                break
                
            if no_improve_count > 500:
                # Perturb: flip 5 random bits
                for _ in range(5):
                    t_idx = random.randint(0, struct['num_triples'] - 1)
                    colors[t_idx] = 1 - colors[t_idx]
                current_viol = count_violations(colors, struct)
                no_improve_count = 0

    # Return best found
    if best_valid_colors is not None:
        # Ensure we return the correct n
        # The struct corresponding to best_n might not be the current one if we broke out early
        # We need to reconstruct or just return the n and colors from the last valid state
        # We saved best_valid_colors when we found it.
        # We need to return {'n': best_n, 'colors': list(best_valid_colors)}
        # But wait, if we found valid for n=12, then moved to 13 and failed, 
        # best_valid_colors is for n=12.
        return {'n': best_n, 'colors': best_valid_colors.tolist()}
    else:
        # Fallback to seed
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

# EVOLVE-BLOCK-END
