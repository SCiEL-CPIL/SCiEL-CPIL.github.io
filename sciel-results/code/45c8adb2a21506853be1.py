import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n = 34
    n = 34
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4_list = list(combinations(range(n), 4))
    num_k4 = len(k4_list)
    k5_list = list(combinations(range(n), 5))
    num_k5 = len(k5_list)
    
    # Map each triple to the K4s and K5s it belongs to
    # And map each K4/K5 to its constituent triples
    k4_data = []
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_data.append(ts)
        
    k5_data = []
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_data.append(ts)

    # Convert to numpy arrays for fast indexing if needed, but lists are fine for iteration in Python if we are careful
    # Actually, for the flip operation, we need to iterate over the K4s and K5s containing the flipped triple.
    # triple_to_k4s[idx] is a list of K4 indices.
    # We need the K4 data (list of triple indices) to update counts? 
    # No, we just need to update the count for the K4.
    # k4_counts[k4_idx] += delta
    # We don't need k4_data for the update, only for initialization.
    
    # Initialize random coloring
    # colors: 0 for Red, 1 for Blue
    colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
    
    # Initialize counts
    # k4_counts[i] = number of red triples in K4 i
    # k5_counts[i] = number of blue triples in K5 i
    
    k4_counts = np.zeros(num_k4, dtype=np.int32)
    k5_counts = np.zeros(num_k5, dtype=np.int32)
    
    # Efficient initialization:
    # Iterate over all triples. If red, increment k4_counts for its K4s. If blue, increment k5_counts for its K5s.
    for t_idx in range(num_triples):
        val = colors[t_idx]
        if val == 0: # Red
            for k4_idx in triple_to_k4s[t_idx]:
                k4_counts[k4_idx] += 1
        else: # Blue
            for k5_idx in triple_to_k5s[t_idx]:
                k5_counts[k5_idx] += 1
                
    # Calculate initial violations
    # Red K4 violation: k4_counts == 4
    # Blue K5 violation: k5_counts == 10
    current_violations = np.sum(k4_counts == 4) + np.sum(k5_counts == 10)
    
    best_violations = current_violations
    best_colors = colors.copy()
    
    def flip(idx):
        """Flip the color of triple idx and update counts and violations."""
        nonlocal current_violations, k4_counts, k5_counts, colors
        
        old_val = colors[idx]
        new_val = 1 - old_val
        colors[idx] = new_val
        
        # Delta for Red count: if 0->1 (Red->Blue), Red count decreases by 1 (delta -1). If 1->0 (Blue->Red), Red count increases by 1 (delta +1).
        # Wait, colors: 0=Red, 1=Blue.
        # k4_counts tracks RED triples.
        # If old_val=0 (Red) -> new_val=1 (Blue): Red count decreases by 1. delta_red = -1.
        # If old_val=1 (Blue) -> new_val=0 (Red): Red count increases by 1. delta_red = +1.
        delta_red = 1 if new_val == 0 else -1
        
        # k5_counts tracks BLUE triples.
        # If old_val=0 (Red) -> new_val=1 (Blue): Blue count increases by 1. delta_blue = +1.
        # If old_val=1 (Blue) -> new_val=0 (Red): Blue count decreases by 1. delta_blue = -1.
        delta_blue = 1 if new_val == 1 else -1
        
        # Update K4 counts and violations
        for k4_idx in triple_to_k4s[idx]:
            old_count = k4_counts[k4_idx]
            new_count = old_count + delta_red
            k4_counts[k4_idx] = new_count
            
            # Violation change
            old_viol = 1 if old_count == 4 else 0
            new_viol = 1 if new_count == 4 else 0
            current_violations += (new_viol - old_viol)
            
        # Update K5 counts and violations
        for k5_idx in triple_to_k5s[idx]:
            old_count = k5_counts[k5_idx]
            new_count = old_count + delta_blue
            k5_counts[k5_idx] = new_count
            
            old_viol = 1 if old_count == 10 else 0
            new_viol = 1 if new_count == 10 else 0
            current_violations += (new_viol - old_viol)

    # WalkSAT / Min-Conflicts with Random Restarts
    # Strategy:
    # 1. Try to find a solution with a limited number of steps.
    # 2. If stuck, restart.
    
    max_restarts = 500
    steps_per_restart = 50000 # Adjust based on speed
    
    # We need a way to pick a "bad" triple to flip in Min-Conflicts.
    # A triple is "bad" if it is part of a violated K4 (as a red edge) or violated K5 (as a blue edge).
    # To make this fast, we can maintain a list of violated K4s and K5s.
    # However, updating this list incrementally is complex.
    # Alternative: Randomly pick a triple. If it's in a violation, flip it. This is slower.
    # Better: Maintain a set of "active" violated K4 indices and K5 indices.
    
    # Let's implement a simpler but effective heuristic:
    # Pick a random triple. Flip it. If violations decrease, keep it. If not, revert?
    # This is Hill Climbing. It gets stuck easily.
    # WalkSAT: Pick a clause (K4 or K5) that is violated. Pick a variable (triple) in that clause at random. Flip it.
    
    # We need to be able to pick a random violated K4 or K5 quickly.
    # We can maintain a list of currently violated K4s and K5s.
    # When a K4 becomes violated, add it to the list. When it becomes unviolated, remove it (or mark as inactive).
    # Using a set for O(1) add/remove and random selection (using a list for random choice and lazy deletion).
    
    violated_k4s = set()
    violated_k5s = set()
    
    # Initialize violated sets
    for i in range(num_k4):
        if k4_counts[i] == 4:
            violated_k4s.add(i)
    for i in range(num_k5):
        if k5_counts[i] == 10:
            violated_k5s.add(i)
    
    current_violations = len(violated_k4s) + len(violated_k5s)
    
    # Update flip to maintain violated sets
    def flip_walksat(idx):
        nonlocal current_violations, k4_counts, k5_counts, colors, violated_k4s, violated_k5s
        
        old_val = colors[idx]
        new_val = 1 - old_val
        colors[idx] = new_val
        
        delta_red = 1 if new_val == 0 else -1
        delta_blue = 1 if new_val == 1 else -1
        
        # Update K4s
        for k4_idx in triple_to_k4s[idx]:
            old_count = k4_counts[k4_idx]
            new_count = old_count + delta_red
            k4_counts[k4_idx] = new_count
            
            was_viol = (old_count == 4)
            is_viol = (new_count == 4)
            
            if was_viol and not is_viol:
                violated_k4s.discard(k4_idx)
                current_violations -= 1
            elif not was_viol and is_viol:
                violated_k4s.add(k4_idx)
                current_violations += 1
                
        # Update K5s
        for k5_idx in triple_to_k5s[idx]:
            old_count = k5_counts[k5_idx]
            new_count = old_count + delta_blue
            k5_counts[k5_idx] = new_count
            
            was_viol = (old_count == 10)
            is_viol = (new_count == 10)
            
            if was_viol and not is_viol:
                violated_k5s.discard(k5_idx)
                current_violations -= 1
            elif not was_viol and is_viol:
                violated_k5s.add(k5_idx)
                current_violations += 1

    # Reset state for the first run
    # (We already initialized counts and colors, but we need to ensure violated sets are consistent)
    # The initialization above did this.
    
    best_violations = current_violations
    best_colors = colors.copy()
    
    # Main Search Loop
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # If we have a good best, we might perturb it. Otherwise, random restart.
        if restart > 0 and random.random() < 0.3 and best_violations < 50:
            # Perturb best
            colors = best_colors.copy()
            # Recount
            k4_counts = np.zeros(num_k4, dtype=np.int32)
            k5_counts = np.zeros(num_k5, dtype=np.int32)
            for t_idx in range(num_triples):
                val = colors[t_idx]
                if val == 0:
                    for k4_idx in triple_to_k4s[t_idx]:
                        k4_counts[k4_idx] += 1
                else:
                    for k5_idx in triple_to_k5s[t_idx]:
                        k5_counts[k5_idx] += 1
            violated_k4s = set(i for i in range(num_k4) if k4_counts[i] == 4)
            violated_k5s = set(i for i in range(num_k5) if k5_counts[i] == 10)
            current_violations = len(violated_k4s) + len(violated_k5s)
            
            # Do a few random flips to escape local min
            for _ in range(10):
                idx = random.randint(0, num_triples - 1)
                flip_walksat(idx)
        else:
            # Random restart
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            k4_counts = np.zeros(num_k4, dtype=np.int32)
            k5_counts = np.zeros(num_k5, dtype=np.int32)
            for t_idx in range(num_triples):
                val = colors[t_idx]
                if val == 0:
                    for k4_idx in triple_to_k4s[t_idx]:
                        k4_counts[k4_idx] += 1
                else:
                    for k5_idx in triple_to_k5s[t_idx]:
                        k5_counts[k5_idx] += 1
            violated_k4s = set(i for i in range(num_k4) if k4_counts[i] == 4)
            violated_k5s = set(i for i in range(num_k5) if k5_counts[i] == 10)
            current_violations = len(violated_k4s) + len(violated_k5s)
            
        if current_violations == 0:
            return {'n': 34, 'colors': colors.tolist()}
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        # WalkSAT Loop
        steps = 0
        max_steps = 20000 # Cap steps per restart to ensure progress
        run_start = time.time()
        
        while steps < max_steps and time.time() < run_start + 1.5: # 1.5s per restart approx
            if current_violations == 0:
                return {'n': 34, 'colors': colors.tolist()}
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # Pick a violated clause
            # Prefer K5s if they are violated? Or random?
            # Let's pick randomly from all violated clauses.
            # To make it faster, we can maintain a list. But set is fine for random.choice if we convert to list? No, that's O(N).
            # We can use a list of violated indices and handle duplicates lazily.
            # For simplicity and given N is small (~24000 K4s, ~200k K5s), and violations are usually sparse,
            # let's just pick a random violated K4 or K5.
            
            # To avoid O(N) list conversion, we can just pick a random K4 index and check if it's violated?
            # No, that's slow if violations are rare.
            # Let's maintain a list of violated K4s and K5s.
            # We'll use a simple list and append/remove.
            # Actually, `random.choice` on a set is not directly supported efficiently in Python without conversion.
            # Let's keep a list `violated_k4_list` and `violated_k5_list` in sync with the sets?
            # Or just convert to list if the set is small?
            # If violations are high, the set is large.
            
            # Alternative: Pick a random triple. If it's in a violation, flip it.
            # This is "Random Conflict".
            
            # Let's stick to: Pick a random violated K4 or K5.
            # We'll maintain lists for fast random access.
            # We need to update these lists in flip_walksat.
            
            # Let's refactor flip_walksat to update lists.
            # This is getting complex. Let's use a simpler heuristic for the inner loop:
            # Pick a random triple. Flip it.
            # This is essentially a random walk. It might be slow.
            
            # Better: Pick a random triple that is part of a violated K4 or K5.
            # How to pick fast?
            # We can precompute for each triple, the list of K4s and K5s.
            # If we pick a random triple, we check if any of its K4s are in violated_k4s or K5s in violated_k5s.
            # If yes, flip it.
            # If no, flip it anyway (to escape).
            
            idx = random.randint(0, num_triples - 1)
            
            # Check if this triple is in a violated clause
            in_violation = False
            for k4_idx in triple_to_k4s[idx]:
                if k4_idx in violated_k4s:
                    in_violation = True
                    break
            if not in_violation:
                for k5_idx in triple_to_k5s[idx]:
                    if k5_idx in violated_k5s:
                        in_violation = True
                        break
                        
            # Flip it
            flip_walksat(idx)
            
            steps += 1
            
        # End of restart loop
        if best_violations == 0:
            break

    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback to smaller n if we failed
        # n=9 is a safe bet
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}

# EVOLVE-BLOCK-END
