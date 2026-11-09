import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_structures(n):
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        k4s = list(itertools.combinations(range(n), 4))
        k5s = list(itertools.combinations(range(n), 5))
        
        k4_triples = {}
        for i, k4 in enumerate(k4s):
            ts = list(itertools.combinations(k4, 3))
            k4_triples[i] = [triple_index[t] for t in ts]
            
        k5_triples = {}
        for i, k5 in enumerate(k5s):
            ts = list(itertools.combinations(k5, 3))
            k5_triples[i] = [triple_index[t] for t in ts]
            
        k4_neighbors = [[] for _ in range(num_triples)]
        for i, ts in k4_triples.items():
            for t_idx in ts:
                k4_neighbors[t_idx].append(i)
                
        k5_neighbors = [[] for _ in range(num_triples)]
        for i, ts in k5_triples.items():
            for t_idx in ts:
                k5_neighbors[t_idx].append(i)
                
        return triples, triple_index, num_triples, k4s, k5s, k4_triples, k5_triples, k4_neighbors, k5_neighbors

    def solve(n, time_limit):
        if time.time() > time_limit:
            return None
            
        triples, triple_index, num_triples, k4s, k5s, k4_triples, k5_triples, k4_neighbors, k5_neighbors = get_structures(n)
        
        # Init with mod 3 pattern
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        
        red_count = [0] * len(k4s)
        blue_count = [0] * len(k5s)
        
        # triple_violation_count[t] = number of violations this triple is part of
        triple_violation_count = [0] * num_triples
        
        for i in range(num_triples):
            c = colors[i]
            for k4_id in k4_neighbors[i]:
                if c == 0:
                    red_count[k4_id] += 1
            for k5_id in k5_neighbors[i]:
                if c == 1:
                    blue_count[k5_id] += 1
                    
        # Identify violations and update triple counts
        for k4_id, cnt in enumerate(red_count):
            if cnt == 4:
                for t_idx in k4_triples[k4_id]:
                    triple_violation_count[t_idx] += 1
                    
        for k5_id, cnt in enumerate(blue_count):
            if cnt == 10:
                for t_idx in k5_triples[k5_id]:
                    triple_violation_count[t_idx] += 1
                    
        # List of active triples
        active_triples = [i for i, c in enumerate(triple_violation_count) if c > 0]
        
        while active_triples and time.time() < time_limit:
            # Pick a random active triple
            # To avoid O(N) list ops, we can just pick a random index and check?
            # Or maintain the list.
            # If we flip a triple, its count changes.
            # It might become 0 (remove from active) or increase (already active).
            # Other triples in the same K4/K5 might change status.
            
            t_idx = random.choice(active_triples)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            # Update K4s
            for k4_id in k4_neighbors[t_idx]:
                was_violation = (red_count[k4_id] == 4)
                if old_color == 0:
                    red_count[k4_id] -= 1
                else:
                    red_count[k4_id] += 1
                is_violation = (red_count[k4_id] == 4)
                
                if was_violation and not is_violation:
                    # Violation removed
                    for t in k4_triples[k4_id]:
                        triple_violation_count[t] -= 1
                elif not was_violation and is_violation:
                    # New violation
                    for t in k4_triples[k4_id]:
                        triple_violation_count[t] += 1
                        
            # Update K5s
            for k5_id in k5_neighbors[t_idx]:
                was_violation = (blue_count[k5_id] == 10)
                if old_color == 1:
                    blue_count[k5_id] -= 1
                else:
                    blue_count[k5_id] += 1
                is_violation = (blue_count[k5_id] == 10)
                
                if was_violation and not is_violation:
                    for t in k5_triples[k5_id]:
                        triple_violation_count[t] -= 1
                elif not was_violation and is_violation:
                    for t in k5_triples[k5_id]:
                        triple_violation_count[t] += 1
                        
            # Update active_triples list
            # This is tricky to do efficiently in-place.
            # We can rebuild it occasionally?
            # Or just check if t_idx is still active.
            # If triple_violation_count[t_idx] == 0, remove it?
            # But other triples might have become 0 or >0.
            
            # Simple approach: Rebuild active_triples every 100 steps?
            # Or just use a set of active triples?
            # Let's use a set for active_triples to allow O(1) add/remove.
            
        # The above loop logic with active_triples list is messy for updates.
        # Let's stick to the Set of Violating K4s/K5s approach but optimize the choice.
        # Actually, the previous set-based approach was:
        # k4_id = random.choice(list(violating_k4s))
        # This is O(N).
        
        # Let's go back to the Set of Violating K4s/K5s, but if the set is large, 
        # it might be slow. 
        # However, for n=34, if we converge, the set shrinks.
        # In the beginning, it's large.
        
        # Alternative: Pick a random triple from the whole space. 
        # If it is in a violation, flip it.
        # This is O(1) pick + O(degree) check.
        # Probability of hitting an active triple is (Active / Total).
        # If Active is small, this is inefficient.
        
        # Let's use the Set approach but with a fallback:
        # If len(violating_k4s) + len(violating_k5s) > 1000, use random triple sampling.
        # Else, use set choice.
        
        return None # Placeholder

    # The implementation above is getting complex. 
    # Let's use the simple Set approach. It's likely fast enough.
    
    def solve_simple(n, time_limit):
        if time.time() > time_limit:
            return None
            
        triples, triple_index, num_triples, k4s, k5s, k4_triples, k5_triples, k4_neighbors, k5_neighbors = get_structures(n)
        
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        
        red_count = [0] * len(k4s)
        blue_count = [0] * len(k5s)
        
        for i in range(num_triples):
            c = colors[i]
            for k4_id in k4_neighbors[i]:
                if c == 0:
                    red_count[k4_id] += 1
            for k5_id in k5_neighbors[i]:
                if c == 1:
                    blue_count[k5_id] += 1
                    
        violating_k4s = set(i for i, c in enumerate(red_count) if c == 4)
        violating_k5s = set(i for i, c in enumerate(blue_count) if c == 10)
        
        while (violating_k4s or violating_k5s) and time.time() < time_limit:
            if violating_k4s:
                k4_id = random.choice(list(violating_k4s))
                ts = k4_triples[k4_id]
                t_idx = random.choice(ts)
            else:
                k5_id = random.choice(list(violating_k5s))
                ts = k5_triples[k5_id]
                t_idx = random.choice(ts)
                
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            for k4_id in k4_neighbors[t_idx]:
                if old_color == 0:
                    red_count[k4_id] -= 1
                    if red_count[k4_id] == 3:
                        violating_k4s.discard(k4_id)
                else:
                    red_count[k4_id] += 1
                    if red_count[k4_id] == 4:
                        violating_k4s.add(k4_id)
                        
            for k5_id in k5_neighbors[t_idx]:
                if old_color == 1:
                    blue_count[k5_id] -= 1
                    if blue_count[k5_id] == 9:
                        violating_k5s.discard(k5_id)
                else:
                    blue_count[k5_id] += 1
                    if blue_count[k5_id] == 10:
                        violating_k5s.add(k5_id)
                        
        if not violating_k4s and not violating_k5s:
            return colors
            
        return None

    t_limit_34 = start_time + 100.0
    cols = solve_simple(34, t_limit_34)
    if cols is not None:
        return {'n': 34, 'colors': cols}
        
    t_limit_33 = start_time + 130.0
    cols = solve_simple(33, t_limit_33)
    if cols is not None:
        return {'n': 33, 'colors': cols}
        
    t_limit_32 = deadline
    cols = solve_simple(32, t_limit_32)
    if cols is not None:
        return {'n': 32, 'colors': cols}
        
    return {'n': 1, 'colors': [0]}
