import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    # We need all differences to be distinct across all rows.
    # This is equivalent to finding n_rows disjoint Golomb rulers of length k+1.
    # Or more generally, a set of n_rows * (k+1) marks on a line such that
    # differences within each group of k+1 are distinct, AND differences
    # across groups are also distinct?
    # "all differences between two entries of the same row are distinct over all rows"
    # This means the set D = { |a_j - a_i| | row r, i<j } has size total_diffs.
    # So no difference can repeat anywhere.

    # Strategy:
    # 1. Fast Greedy Construction with "Pressure" (minimize max new diff)
    # 2. Local Search / Simulated Annealing to reduce the maximum element (scope)

    def get_diffs_set(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(scope_limit, randomize=False):
        """
        Greedy construction.
        If randomize is True, we add some noise to the selection to escape local minima.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                candidates = []
                
                # Determine search range
                # We want to keep the row compact, so we don't want to jump too far.
                # But we must avoid conflicts.
                # Let's scan from last_val + 1 upwards.
                # To speed up, we can check a window.
                
                found = False
                # Optimization: Instead of checking every x, we can be smarter?
                # For now, linear scan is okay for small k.
                # But we can randomize the start or step?
                
                # Heuristic: Try values that don't create "large" gaps if possible?
                # Actually, "pressure" strategy: minimize the maximum difference introduced.
                # This is naturally done by greedy smallest-x.
                
                # Let's try a few restarts with different random seeds or perturbations.
                
                # To improve over simple greedy, let's try to find a valid x
                # but maybe skip some small x's if they seem "bad"?
                # No, greedy smallest is usually good for packing.
                
                # Let's just do a linear scan but with a cap on how far we look in this iteration?
                # If we don't find one, we return None.
                
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        """
        Local search to minimize the maximum element (scope).
        Moves:
        1. Shrink: Replace an element with a smaller valid one.
        2. Swap: Swap two elements? No, order must be preserved.
        3. Move: Change an element to a different valid value.
        
        We will use a simple hill-climbing / simulated annealing approach.
        Objective: Minimize max(max(r) for r in rows).
        Secondary: Minimize sum of squares or similar to pack tighter?
        Let's stick to minimizing the max element primarily.
        
        We will maintain the set of used differences for O(1) checks.
        """
        rows = [r[:] for r in rows]
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        def get_scope():
            return max(max(r) for r in rows)
        
        best_rows = [r[:] for r in rows]
        best_scope = get_scope()
        
        t_end = time.time() + time_limit
        temperature = 10.0
        cooling = 0.9995
        
        while time.time() < t_end:
            improved = False
            
            # Try to shrink the largest elements
            # Identify the row with the largest element
            max_val = get_scope()
            target_row_idx = -1
            target_pos = -1
            for i, r in enumerate(rows):
                if r[-1] == max_val:
                    target_row_idx = i
                    target_pos = len(r) - 1
                    break
            
            if target_row_idx == -1:
                # Try random moves
                r_idx = random.randint(0, n_rows - 1)
                p_idx = random.randint(1, len(rows[r_idx]) - 1)
                target_row_idx = r_idx
                target_pos = p_idx

            row = rows[target_row_idx]
            pos = target_pos
            old_val = row[pos]
            
            # Remove old differences
            old_diffs = []
            for i in range(len(row)):
                if i != pos:
                    if i < pos:
                        d = old_val - row[i]
                    else:
                        d = row[i] - old_val
                    old_diffs.append(d)
                    used_diffs.discard(d)
            
            # Try to find a new value
            # We want a value smaller than old_val if possible (shrink)
            # Or a random value if we are stuck (annealing)
            
            # Determine valid range
            min_v = row[pos - 1] + 1 if pos > 0 else 1
            max_v = row[pos + 1] - 1 if pos < len(row) - 1 else max_val + 100 # Can't exceed current max too much
            
            # If we are trying to shrink, we look in [min_v, old_val - 1]
            # If we are stuck, we look in a wider range.
            
            candidates = []
            
            # 1. Try to shrink: scan from min_v to old_val - 1
            # This can be slow if the range is large.
            # Optimization: Only try a subset?
            # Let's try all values in [min_v, min(old_val - 1, min_v + 50)] first for speed.
            # Then if no improvement, try random values.
            
            search_end = min(old_val - 1, min_v + 100)
            for x in range(min_v, search_end + 1):
                valid = True
                for i in range(len(row)):
                    if i != pos:
                        if i < pos:
                            d = x - row[i]
                        else:
                            d = row[i] - x
                        if d in used_diffs:
                            valid = False
                            break
                if valid:
                    candidates.append(x)
            
            if candidates:
                # Pick the smallest one (greedy shrink)
                new_val = min(candidates)
                # Commit
                for i in range(len(row)):
                    if i != pos:
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        used_diffs.add(d)
                row[pos] = new_val
                improved = True
                current_scope = get_scope()
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            
            else:
                # No shrink found. Try a random move (annealing)
                # Pick a random x in [min_v, max_v]
                # But we must ensure we don't increase scope too much?
                # Let's try to find ANY valid move, even if it increases scope slightly,
                # to escape local minima.
                
                found_move = False
                for _ in range(20): # Try a few random moves
                    x = random.randint(min_v, max_v)
                    if x == old_val:
                        continue
                    valid = True
                    for i in range(len(row)):
                        if i != pos:
                            if i < pos:
                                d = x - row[i]
                            else:
                                d = row[i] - x
                            if d in used_diffs:
                                valid = False
                                break
                    if valid:
                        # Accept with probability based on temperature?
                        # For simplicity, accept if it doesn't increase scope,
                        # or with probability exp(-delta/T) if it does.
                        new_scope = max(max(r) for r in rows if r is not row)
                        if x > new_scope:
                            new_scope = x
                        
                        if new_scope <= best_scope:
                            # Accept
                            for i in range(len(row)):
                                if i != pos:
                                    if i < pos:
                                        d = x - row[i]
                                    else:
                                        d = row[i] - x
                                    used_diffs.add(d)
                            row[pos] = x
                            found_move = True
                            improved = True
                            break
                        else:
                            # Worse move, accept with prob exp(-(new_scope - best_scope)/T)
                            delta = new_scope - best_scope
                            prob = math.exp(-delta / temperature)
                            if random.random() < prob:
                                for i in range(len(row)):
                                    if i != pos:
                                        if i < pos:
                                            d = x - row[i]
                                        else:
                                            d = row[i] - x
                                        used_diffs.add(d)
                                row[pos] = x
                                found_move = True
                                improved = True
                                break
                
                if not found_move:
                    # Restore old state
                    for d in old_diffs:
                        used_diffs.add(d)
                    row[pos] = old_val
            
            if improved:
                temperature = max(temperature * cooling, 0.1)
            else:
                # Stuck, cool down slower or restart?
                temperature = max(temperature * 0.99, 0.1)
            
            # Safety check: if we run out of time, break
            if time.time() > deadline - 5:
                break

        return best_rows, best_scope

    # Step 1: Greedy construction with multiple restarts
    best_rows = None
    best_scope = float('inf')

    # Try to construct with a tight scope first
    # The counting bound is 112. Known best is 134.
    # Let's try to construct greedily and see what scope we get.
    
    # Increase restarts to find better initial solutions
    for restart in range(50):
        if time.time() > deadline - 40:
            break
            
        # Use a slightly randomized greedy?
        # For now, standard greedy is deterministic.
        # To get variety, we can shuffle the order of rows?
        # Or add noise to the candidate selection?
        
        # Let's try standard greedy first.
        rows = greedy_construct(scope_limit=500)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
        # If we haven't found a good one, try with larger scope?
        # No, greedy smallest-x will always give the smallest possible for that row order.
        # The order of rows matters.
        # Let's try shuffling the "virtual" order?
        # Actually, since rows are independent except for the global diff set,
        # the order in which we build them matters.
        # Let's try building rows in random order?
        # But we need to return a list of rows. The order of rows in the list doesn't matter for validity.
        # So we can just shuffle which "slot" we fill first?
        # No, the diff set is global.
        
        # Let's just do multiple greedy runs. Since it's deterministic, they will be the same.
        # We need to introduce randomness in the greedy step.
        
        # Revised Greedy:
        # Instead of always taking the smallest valid x, take a random valid x from a window?
        # Or take the smallest, but with a small probability take the second smallest?
        
        # Let's implement a "Noisy Greedy"
        
    # Implement Noisy Greedy
    def noisy_greedy_construct(scope_limit, noise_level=0.1):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                found = False
                
                # Collect a few valid candidates
                valid_candidates = []
                for x in range(last_val + 1, min(last_val + 50, scope_limit + 1)):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        valid_candidates.append(x)
                        if len(valid_candidates) > 5: # Limit candidates for speed
                            break
                
                if not valid_candidates:
                    # Try further out
                    for x in range(last_val + 50, scope_limit + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            valid_candidates.append(x)
                            break
                
                if not valid_candidates:
                    return None
                
                # Pick a candidate
                # With probability (1 - noise), pick the smallest.
                # Otherwise, pick a random one from the first few.
                if random.random() < (1 - noise_level):
                    x = valid_candidates[0]
                else:
                    x = random.choice(valid_candidates[:3])
                
                row.append(x)
                for y in row[:-1]:
                    used_diffs.add(x - y)
                found = True
            rows.append(row)
        return rows

    # Restart with noisy greedy
    for restart in range(100):
        if time.time() > deadline - 40:
            break
            
        rows = noisy_greedy_construct(scope_limit=500, noise_level=0.2)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
    if best_rows is None:
        # Fallback to larger scope
        rows = greedy_construct(scope_limit=5000)
        if rows:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
        else:
            # Emergency fallback
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            return rows

    # Step 2: Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope = local_search(best_rows, remaining_time - 5)

    return best_rows
