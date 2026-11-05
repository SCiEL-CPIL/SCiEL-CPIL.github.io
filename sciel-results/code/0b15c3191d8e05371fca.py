import itertools
import random
import time
import sys

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Target n=34, fallback to lower n
    for n in range(34, 19, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Precompute structure
        vertices = range(n)
        pairs = list(itertools.combinations(range(n), 2))
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        num_pairs = len(pairs)
        
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # For each triple, store the 3 pairs that define it
        # triple (a,b,c) -> pairs (a,b), (b,c), (a,c)
        triple_pairs = []
        for a, b, c in triples:
            triple_pairs.append((pair_to_idx[(a,b)], pair_to_idx[(b,c)], pair_to_idx[(a,c)]))
            
        # Precompute 4-sets and 5-sets and their triples
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        four_set_triples = []
        for s in four_sets:
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            four_set_triples.append(ts)
            
        five_set_triples = []
        for s in five_sets:
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            five_set_triples.append(ts)
            
        # Reverse mapping: pair_idx -> list of triples containing it
        # And for each triple, which 4-sets and 5-sets it belongs to?
        # For efficient penalty update when a pair changes:
        # 1. Change pair_val[p_idx].
        # 2. Update color of all triples containing p_idx.
        # 3. For each changed triple, update the counts of Red/Blue in its parent 4-sets and 5-sets.
        
        pair_to_triples = [[] for _ in range(num_pairs)]
        for t_idx, p_tuple in enumerate(triple_pairs):
            for p_idx in p_tuple:
                pair_to_triples[p_idx].append(t_idx)
                
        # triple_to_4sets: list of 4-set indices for each triple
        # triple_to_5sets: list of 5-set indices for each triple
        triple_to_4sets = [[] for _ in range(num_triples)]
        for idx, ts in enumerate(four_set_triples):
            for t_idx in ts:
                triple_to_4sets[t_idx].append(idx)
                
        triple_to_5sets = [[] for _ in range(num_triples)]
        for idx, ts in enumerate(five_set_triples):
            for t_idx in ts:
                triple_to_5sets[t_idx].append(idx)

        # Mappings: Sum mod 3 -> Color (0=Red, 1=Blue)
        # We try a few mappings. 
        # Map 0: S=0 -> Red, S!=0 -> Blue
        # Map 1: S!=0 -> Red, S=0 -> Blue
        # Map 2: S in {0,1} -> Red, S=2 -> Blue
        # Map 3: S in {0,2} -> Red, S=1 -> Blue
        # Map 4: S in {1,2} -> Red, S=0 -> Blue
        mappings = [
            lambda s: 0 if s == 0 else 1,
            lambda s: 1 if s == 0 else 0,
            lambda s: 0 if s in [0, 1] else 1,
            lambda s: 0 if s in [0, 2] else 1,
            lambda s: 0 if s in [1, 2] else 1,
        ]
        
        found = False
        for map_func in mappings:
            if time.time() - start_time > time_limit:
                break
                
            # Local Search on pair values
            # State: pair_vals (list of 0,1,2)
            # We try random restarts
            for restart in range(5):
                if time.time() - start_time > time_limit:
                    break
                    
                random.seed(restart * 100 + n)
                pair_vals = [random.randint(0, 2) for _ in range(num_pairs)]
                
                # Calculate initial colors and penalty
                colors = [0] * num_triples
                for t_idx, (p1, p2, p3) in enumerate(triple_pairs):
                    s = (pair_vals[p1] + pair_vals[p2] + pair_vals[p3]) % 3
                    colors[t_idx] = map_func(s)
                    
                # Counts for 4-sets (Red) and 5-sets (Blue)
                red_count_4 = [0] * len(four_set_triples)
                blue_count_5 = [0] * len(five_set_triples)
                
                for t_idx, c in enumerate(colors):
                    for s_idx in triple_to_4sets[t_idx]:
                        if c == 0: red_count_4[s_idx] += 1
                    for s_idx in triple_to_5sets[t_idx]:
                        if c == 1: blue_count_5[s_idx] += 1
                        
                penalty = sum(1 for r in red_count_4 if r == 4) + sum(1 for b in blue_count_5 if b == 10)
                
                if penalty == 0:
                    # Construct final colors list
                    return {'n': n, 'colors': colors}
                    
                # Local Search Loop
                max_iters = 20000
                no_improve_count = 0
                threshold = 500
                
                for _ in range(max_iters):
                    if time.time() - start_time > time_limit:
                        break
                        
                    if penalty == 0:
                        return {'n': n, 'colors': colors}
                        
                    # Pick random pair
                    p_idx = random.randint(0, num_pairs - 1)
                    old_val = pair_vals[p_idx]
                    
                    # Try changing to other 2 values
                    best_new_val = -1
                    best_penalty = penalty
                    
                    for new_val in [0, 1, 2]:
                        if new_val == old_val: continue
                        
                        # Calculate delta penalty
                        # This requires simulating the change.
                        # To avoid deep copy, we can calculate the change in penalty
                        # by looking at the affected triples.
                        
                        # Affected triples:
                        affected_triples = pair_to_triples[p_idx]
                        
                        # We need to know how the color of these triples changes.
                        # And how that affects the counts of their parent sets.
                        
                        # This is complex to do incrementally without state.
                        # Given n=34, num_triples ~ 5984.
                        # Affected triples ~ 32.
                        # Parent sets per triple ~ 60.
                        # Total ops ~ 32 * 60 = 1920 per trial.
                        # 2 trials per iteration.
                        # 20000 iters * 2 * 1920 = 76M ops.
                        # This might be tight in Python.
                        
                        # Optimization: Pre-compute the "current" state of counts.
                        # When we change a triple's color from c_old to c_new:
                        # For each 4-set containing it:
                        #   If c_old == 0 and c_new == 1: red_count decreases by 1.
                        #   If c_old == 1 and c_new == 0: red_count increases by 1.
                        #   Check if it crosses 4 (bad) boundary.
                        # Same for 5-sets and Blue.
                        
                        # Let's implement a helper to estimate penalty change.
                        # We need the current red_count_4 and blue_count_5.
                        
                        delta_penalty = 0
                        # We must be careful: multiple triples in the same set might be affected.
                        # Since we are changing ONE pair, the affected triples are distinct.
                        # But they share parent sets.
                        # We must aggregate changes per parent set before updating penalty.
                        
                        # Change in red_count for each 4-set
                        red_delta = {}
                        blue_delta = {}
                        
                        for t_idx in affected_triples:
                            p1, p2, p3 = triple_pairs[t_idx]
                            # Calculate new sum
                            s_new = 0
                            if p1 == p_idx: s_new += new_val
                            else: s_new += pair_vals[p1]
                            if p2 == p_idx: s_new += new_val
                            else: s_new += pair_vals[p2]
                            if p3 == p_idx: s_new += new_val
                            else: s_new += pair_vals[p3]
                            s_new %= 3
                            
                            c_new = map_func(s_new)
                            c_old = colors[t_idx]
                            
                            if c_old == c_new:
                                continue
                                
                            # Update deltas
                            for s_idx in triple_to_4sets[t_idx]:
                                if c_old == 0 and c_new == 1: # Red -> Blue
                                    red_delta[s_idx] = red_delta.get(s_idx, 0) - 1
                                elif c_old == 1 and c_new == 0: # Blue -> Red
                                    red_delta[s_idx] = red_delta.get(s_idx, 0) + 1
                            
                            for s_idx in triple_to_5sets[t_idx]:
                                if c_old == 0 and c_new == 1: # Red -> Blue
                                    # Blue count increases
                                    blue_delta[s_idx] = blue_delta.get(s_idx, 0) + 1
                                elif c_old == 1 and c_new == 0: # Blue -> Red
                                    # Blue count decreases
                                    blue_delta[s_idx] = blue_delta.get(s_idx, 0) - 1
                            
                        # Calculate penalty change based on deltas
                        for s_idx, delta in red_delta.items():
                            old_count = red_count_4[s_idx]
                            new_count = old_count + delta
                            # Check if badness changed
                            if old_count == 4 and new_count != 4:
                                delta_penalty -= 1
                            elif old_count != 4 and new_count == 4:
                                delta_penalty += 1
                                
                        for s_idx, delta in blue_delta.items():
                            old_count = blue_count_5[s_idx]
                            new_count = old_count + delta
                            if old_count == 10 and new_count != 10:
                                delta_penalty -= 1
                            elif old_count != 10 and new_count == 10:
                                delta_penalty += 1
                                
                        if delta_penalty < 0:
                            new_penalty = penalty + delta_penalty
                            if new_penalty < best_penalty:
                                best_penalty = new_penalty
                                best_new_val = new_val
                                
                    if best_new_val != -1:
                        # Apply change
                        pair_vals[p_idx] = best_new_val
                        # Update colors and counts
                        for t_idx in pair_to_triples[p_idx]:
                            p1, p2, p3 = triple_pairs[t_idx]
                            s_new = (pair_vals[p1] + pair_vals[p2] + pair_vals[p3]) % 3
                            c_new = map_func(s_new)
                            c_old = colors[t_idx]
                            
                            if c_old != c_new:
                                colors[t_idx] = c_new
                                for s_idx in triple_to_4sets[t_idx]:
                                    if c_old == 0: red_count_4[s_idx] -= 1
                                    else: red_count_4[s_idx] += 1
                                for s_idx in triple_to_5sets[t_idx]:
                                    if c_old == 1: blue_count_5[s_idx] -= 1
                                    else: blue_count_5[s_idx] += 1
                                    
                        penalty = best_penalty
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count > threshold:
                            break # Restart

        if found:
            return {'n': n, 'colors': colors}
            
    # Fallback
    n = 1
    return {'n': n, 'colors': []}
