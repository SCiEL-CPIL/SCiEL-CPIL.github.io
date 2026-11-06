import random
import math
import time
from itertools import combinations

def construct():
    # We will try to find a valid coloring for n=34
    # If it fails, we try n=33, etc.
    
    def solve_n(n):
        triples = list(combinations(range(n), 3))
        T = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Map each K4 to its triple indices
        k4_triples = []
        for k4 in k4s:
            t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4_triples.append(t_indices)
            
        # Map each K5 to its triple indices
        k5_triples = []
        for k5 in k5s:
            t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(t_indices)
            
        # Map each triple to the K4s and K5s it belongs to
        triple_to_k4s = [[] for _ in range(T)]
        triple_to_k5s = [[] for _ in range(T)]
        
        for i, t_inds in enumerate(k4_triples):
            for t in t_inds:
                triple_to_k4s[t].append(i)
                
        for i, t_inds in enumerate(k5_triples):
            for t in t_inds:
                triple_to_k5s[t].append(i)

        # Initialize with bias p_red
        # Optimal p is around 0.45 to balance expected violations
        p_red = 0.45
        colors = [0 if random.random() < p_red else 1 for _ in range(T)]
        
        # Initialize counts
        red_counts = [0] * len(k4s)
        blue_counts = [0] * len(k5s)
        
        # Count initial violations
        for t_idx, c in enumerate(colors):
            if c == 0: # Red
                for k4_idx in triple_to_k4s[t_idx]:
                    red_counts[k4_idx] += 1
            else: # Blue
                for k5_idx in triple_to_k5s[t_idx]:
                    blue_counts[k5_idx] += 1
        
        total_violations = 0
        violated_k4s = set()
        violated_k5s = set()
        
        for i in range(len(k4s)):
            if red_counts[i] == 4:
                total_violations += 1
                violated_k4s.add(i)
                
        for i in range(len(k5s)):
            if blue_counts[i] == 10:
                total_violations += 1
                violated_k5s.add(i)

        if total_violations == 0:
            return colors

        start_time = time.time()
        timeout = 115
        
        # MinConflicts Algorithm
        # 1. Pick a random violated constraint
        # 2. Pick a random variable in it
        # 3. Assign the value that minimizes conflicts for that variable
        
        while total_violations > 0 and time.time() - start_time < timeout:
            # Pick a violated K4 or K5
            if violated_k4s and (not violated_k5s or random.random() < 0.5):
                # Pick random violated K4
                k4_idx = random.choice(list(violated_k4s))
                # Pick random triple in this K4
                t_idx = random.choice(k4_triples[k4_idx])
            else:
                # Pick random violated K5
                k5_idx = random.choice(list(violated_k5s))
                # Pick random triple in this K5
                t_idx = random.choice(k5_triples[k5_idx])
            
            # Try flipping t_idx
            current_color = colors[t_idx]
            new_color = 1 - current_color
            
            # Calculate delta in violations if we flip t_idx to new_color
            delta = 0
            
            if current_color == 0: # Flipping Red -> Blue
                # Red counts decrease
                for k4_idx in triple_to_k4s[t_idx]:
                    if red_counts[k4_idx] == 4:
                        delta -= 1 # Resolves violation
                    elif red_counts[k4_idx] == 3:
                        pass # No change in violation status (3->2)
                    # If it was <3, no change in violation status
                    
                    # Blue counts increase
                for k5_idx in triple_to_k5s[t_idx]:
                    if blue_counts[k5_idx] == 10:
                        delta -= 1 # Resolves violation (10->9)
                    elif blue_counts[k5_idx] == 9:
                        delta += 1 # Creates violation (9->10)
            else: # Flipping Blue -> Red
                # Blue counts decrease
                for k5_idx in triple_to_k5s[t_idx]:
                    if blue_counts[k5_idx] == 10:
                        delta -= 1 # Resolves violation
                    # If it was <10, no change in violation status (unless it was 10, handled)
                    
                # Red counts increase
                for k4_idx in triple_to_k4s[t_idx]:
                    if red_counts[k4_idx] == 4:
                        pass # 4->5 impossible, but if it was 4, it was violation.
                        # Wait, if it was 4, it's a violation. Flipping one edge to Blue makes it 3 Red.
                        # So if red_counts[k4_idx] == 4, and we flip a Red edge to Blue?
                        # No, we are flipping Blue -> Red here.
                        # So red_counts increases.
                        # If it was 3, it becomes 4 (New Violation).
                        # If it was 4, it becomes 5 (Impossible, max 4 edges).
                        # So if red_counts[k4_idx] == 3:
                        #   delta += 1
                    if red_counts[k4_idx] == 3:
                        delta += 1 # Creates violation (3->4)
                    
            # MinConflicts: Accept if delta < 0 or (delta == 0 and random < 0.5)
            # Actually, standard MinConflicts picks the value that minimizes the number of conflicts for the variable.
            # Here we only have 2 values. We can just check the delta.
            # If delta < 0, accept.
            # If delta == 0, accept with prob 0.5 (to allow wandering).
            # If delta > 0, reject? Or accept with small prob (SA style)?
            # Pure MinConflicts rejects delta > 0. This can get stuck in local minima.
            # Let's add a small probability to accept delta > 0 to escape local minima.
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                # Accept with probability 0.1 to allow escaping local minima
                accept = random.random() < 0.1

            if accept:
                # Update counts
                if current_color == 0: # Red -> Blue
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_counts[k4_idx] -= 1
                        if red_counts[k4_idx] == 4:
                            violated_k4s.discard(k4_idx)
                        elif red_counts[k4_idx] == 3:
                            violated_k4s.add(k4_idx) # Wait, 3 is not a violation. Only 4 is.
                            # Correction:
                            # Violation is count == 4.
                            # If count was 4, it was in violated_k4s. Now 3. Remove.
                            # If count was 3, it was NOT in violated_k4s. Now 2. No change.
                            # So:
                            # if old == 4: remove from set
                            # if old == 3: no change to set
                            pass # Logic below handles this properly
                    
                    # Let's re-evaluate the set updates carefully.
                    # Old state:
                    # If red_counts[k4] was 4, k4 was in violated_k4s.
                    # New state: red_counts[k4] is 3. Not a violation.
                    # So remove k4 from violated_k4s.
                    
                    # If red_counts[k4] was 3, k4 was NOT in violated_k4s.
                    # New state: red_counts[k4] is 2. Not a violation.
                    # No change to set.
                    
                    for k4_idx in triple_to_k4s[t_idx]:
                        old_count = red_counts[k4_idx]
                        red_counts[k4_idx] -= 1
                        new_count = red_counts[k4_idx]
                        if old_count == 4:
                            violated_k4s.discard(k4_idx)
                        # No other transition creates or destroys a violation for K4
                        # (3->2 is safe, 2->1 safe, etc.)
                        # (4->3 is safe, removes violation)
                        
                    for k5_idx in triple_to_k5s[t_idx]:
                        old_count = blue_counts[k5_idx]
                        blue_counts[k5_idx] += 1
                        new_count = blue_counts[k5_idx]
                        if old_count == 9:
                            violated_k5s.add(k5_idx)
                        elif old_count == 10:
                            violated_k5s.discard(k5_idx)
                            
                else: # Blue -> Red
                    for k5_idx in triple_to_k5s[t_idx]:
                        old_count = blue_counts[k5_idx]
                        blue_counts[k5_idx] -= 1
                        new_count = blue_counts[k5_idx]
                        if old_count == 10:
                            violated_k5s.discard(k5_idx)
                        # 9->8 safe
                        
                    for k4_idx in triple_to_k4s[t_idx]:
                        old_count = red_counts[k4_idx]
                        red_counts[k4_idx] += 1
                        new_count = red_counts[k4_idx]
                        if old_count == 3:
                            violated_k4s.add(k4_idx)
                        # 4->5 impossible

                total_violations += delta
                colors[t_idx] = new_color

        if total_violations == 0:
            return colors
        return None

    # Try n=34
    colors = solve_n(34)
    if colors is not None:
        return {'n': 34, 'colors': colors}
        
    # Try n=33
    colors = solve_n(33)
    if colors is not None:
        return {'n': 33, 'colors': colors}
        
    # Try n=32
    colors = solve_n(32)
    if colors is not None:
        return {'n': 32, 'colors': colors}

    # Fallback to seed
    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}

# EVOLVE-BLOCK-START
def construct():
    import random
    import math
    import time
    from itertools import combinations

    def solve_n(n):
        triples = list(combinations(range(n), 3))
        T = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        k4_triples = []
        for k4 in k4s:
            t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4_triples.append(t_indices)
            
        k5_triples = []
        for k5 in k5s:
            t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(t_indices)
            
        triple_to_k4s = [[] for _ in range(T)]
        triple_to_k5s = [[] for _ in range(T)]
        
        for i, t_inds in enumerate(k4_triples):
            for t in t_inds:
                triple_to_k4s[t].append(i)
                
        for i, t_inds in enumerate(k5_triples):
            for t in t_inds:
                triple_to_k5s[t].append(i)

        p_red = 0.45
        colors = [0 if random.random() < p_red else 1 for _ in range(T)]
        
        red_counts = [0] * len(k4s)
        blue_counts = [0] * len(k5s)
        
        for t_idx, c in enumerate(colors):
            if c == 0:
                for k4_idx in triple_to_k4s[t_idx]:
                    red_counts[k4_idx] += 1
            else:
                for k5_idx in triple_to_k5s[t_idx]:
                    blue_counts[k5_idx] += 1
        
        total_violations = 0
        violated_k4s = set()
        violated_k5s = set()
        
        for i in range(len(k4s)):
            if red_counts[i] == 4:
                total_violations += 1
                violated_k4s.add(i)
                
        for i in range(len(k5s)):
            if blue_counts[i] == 10:
                total_violations += 1
                violated_k5s.add(i)

        if total_violations == 0:
            return colors

        start_time = time.time()
        timeout = 115
        
        while total_violations > 0 and time.time() - start_time < timeout:
            if violated_k4s and (not violated_k5s or random.random() < 0.5):
                k4_idx = random.choice(list(violated_k4s))
                t_idx = random.choice(k4_triples[k4_idx])
            else:
                k5_idx = random.choice(list(violated_k5s))
                t_idx = random.choice(k5_triples[k5_idx])
            
            current_color = colors[t_idx]
            new_color = 1 - current_color
            delta = 0
            
            if current_color == 0:
                for k4_idx in triple_to_k4s[t_idx]:
                    if red_counts[k4_idx] == 4:
                        delta -= 1
                for k5_idx in triple_to_k5s[t_idx]:
                    if blue_counts[k5_idx] == 10:
                        delta -= 1
                    elif blue_counts[k5_idx] == 9:
                        delta += 1
            else:
                for k5_idx in triple_to_k5s[t_idx]:
                    if blue_counts[k5_idx] == 10:
                        delta -= 1
                for k4_idx in triple_to_k4s[t_idx]:
                    if red_counts[k4_idx] == 3:
                        delta += 1
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < 0.1

            if accept:
                if current_color == 0:
                    for k4_idx in triple_to_k4s[t_idx]:
                        old_count = red_counts[k4_idx]
                        red_counts[k4_idx] -= 1
                        if old_count == 4:
                            violated_k4s.discard(k4_idx)
                    for k5_idx in triple_to_k5s[t_idx]:
                        old_count = blue_counts[k5_idx]
                        blue_counts[k5_idx] += 1
                        if old_count == 9:
                            violated_k5s.add(k5_idx)
                        elif old_count == 10:
                            violated_k5s.discard(k5_idx)
                else:
                    for k5_idx in triple_to_k5s[t_idx]:
                        old_count = blue_counts[k5_idx]
                        blue_counts[k5_idx] -= 1
                        if old_count == 10:
                            violated_k5s.discard(k5_idx)
                    for k4_idx in triple_to_k4s[t_idx]:
                        old_count = red_counts[k4_idx]
                        red_counts[k4_idx] += 1
                        if old_count == 3:
                            violated_k4s.add(k4_idx)

                total_violations += delta
                colors[t_idx] = new_color

        if total_violations == 0:
            return colors
        return None

    colors = solve_n(34)
    if colors is not None:
        return {'n': 34, 'colors': colors}
        
    colors = solve_n(33)
    if colors is not None:
        return {'n': 33, 'colors': colors}
        
    colors = solve_n(32)
    if colors is not None:
        return {'n': 32, 'colors': colors}

    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
# EVOLVE-BLOCK-END
