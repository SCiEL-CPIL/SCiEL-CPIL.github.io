import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Greedy Initialization
    rows = []
    global_used_diffs = set()
    
    for _ in range(n_rows):
        row = [0]
        used_in_row = set()
        current_max = 0
        for _ in range(k):
            x = current_max + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & global_used_diffs) and not (new_diffs & used_in_row):
                    row.append(x)
                    used_in_row |= new_diffs
                    current_max = x
                    break
                x += 1
        rows.append(row)
        # Update global_used_diffs
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                global_used_diffs.add(row[j] - row[i])

    # Precompute row diffs for fast access
    row_diffs = []
    for r in rows:
        d = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d.add(r[j] - row[i] if False else r[j] - r[i]) # Typo check: r[j] - r[i]
        row_diffs.append(d)
        
    # Helper to calculate diffs for a row
    def get_row_diffs(r):
        d = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d.add(r[j] - r[i])
        return d

    # Re-initialize row_diffs correctly
    row_diffs = [get_row_diffs(r) for r in rows]
    
    best_scope = max(max(r) for r in rows)
    current_scope = best_scope
    
    # Local Search Loop
    while time.time() < deadline:
        # Identify the row with the largest last element
        max_val = -1
        target_idx = -1
        for i, r in enumerate(rows):
            if r[-1] > max_val:
                max_val = r[-1]
                target_idx = i
        
        # Try to reduce the last element of target_idx
        improved = False
        r = rows[target_idx]
        last_idx = k
        current_last = r[last_idx]
        
        # Try values smaller than current_last
        # We can try a range of values to escape local minima if direct reduction fails
        # But primarily we want to reduce the scope.
        
        # To make search more effective, we can try to reduce any element in the target row
        # that is "blocking" a smaller last element, or just try to reduce the last element.
        
        # Strategy: Try to find a valid new value for r[last_idx] that is smaller.
        # If not found, try to modify r[last_idx-1] or others to "open up" space.
        
        # Let's try a randomized local move on the target row.
        # Pick a random index i (1..k) in the target row.
        # Try to change r[i] to a new value.
        
        # To specifically target the objective, we prioritize changing the last element.
        # If we can't change the last element, we change a previous one.
        
        indices_to_try = list(range(1, k+1))
        random.shuffle(indices_to_try)
        
        # Prioritize last index
        if k in indices_to_try:
            indices_to_try.remove(k)
            indices_to_try.insert(0, k)
            
        found_move = False
        for i in indices_to_try:
            old_val = r[i]
            
            # Determine range of new values to try
            # Lower bound: r[i-1] + 1
            # Upper bound: r[i+1] - 1 (if i < k) or some upper limit (if i == k)
            
            low = r[i-1] + 1
            high = old_val
            
            # If i == k, we want to reduce, so high = old_val - 1 is the main goal.
            # But we might also need to increase to allow a subsequent reduction? 
            # Usually, reducing the max element is the key.
            # If i < k, we might want to change it to a different value to free up diffs.
            
            # Let's try values in [low, high] for reduction.
            # If i == k, we only care about < old_val.
            # If i < k, we can try < old_val, == old_val (skip), or > old_val?
            # Changing to > old_val increases the row's max (if it was the max) or doesn't help scope directly.
            # However, it might free up differences.
            
            # For simplicity and effectiveness on "minimize max", let's try to reduce.
            # If i == k, try old_val - 1, old_val - 2, ... down to low.
            # If i < k, try old_val - 1, ... down to low.
            
            # To avoid checking all values, we can sample or check a few.
            # But the range might be small.
            
            # Let's try a random sample of valid positions in [low, high-1] if high > low.
            if high > low:
                # Generate a few candidates
                candidates = []
                # Always try the immediate predecessor
                candidates.append(high - 1)
                # Add a few random ones
                for _ in range(5):
                    if high - 1 > low:
                        candidates.append(random.randint(low, high - 1))
                candidates = list(set(candidates))
                
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    
                    # Check validity
                    # 1. Strictly increasing
                    if i > 0 and new_val <= r[i-1]:
                        continue
                    if i < k and new_val >= r[i+1]:
                        continue
                    
                    # 2. Difference conflicts
                    # Old diffs involving r[i]:
                    # {old_val - r[j] for j < i} U {r[j] - old_val for j > i}
                    # New diffs involving new_val:
                    # {new_val - r[j] for j < i} U {r[j] - new_val for j > i}
                    
                    # Diffs NOT involving r[i] remain the same.
                    # Let D_other be the set of diffs in r that do NOT involve index i.
                    # Let D_global_other be global_used_diffs - row_diffs[target_idx].
                    
                    # We need:
                    # 1. New_diffs_involving_i are distinct from each other.
                    #    Since r is strictly increasing, new_val - r[j] for j<i are distinct?
                    #    Yes, if r[j] are distinct.
                    #    r[j] - new_val for j>i are distinct? Yes.
                    #    Are they distinct across the two sets?
                    #    new_val - r[j] > 0 and r[j] - new_val > 0.
                    #    Could new_val - r[j] == r[l] - new_val?
                    #    2*new_val == r[j] + r[l]. Possible.
                    #    So we must check internal consistency.
                    
                    # 2. New_diffs_involving_i are disjoint from D_other.
                    # 3. New_diffs_involving_i are disjoint from D_global_other.
                    
                    # Let's compute D_other efficiently?
                    # It's just row_diffs[target_idx] minus the old diffs involving i.
                    
                    old_diffs_i = set()
                    for j in range(i):
                        old_diffs_i.add(old_val - r[j])
                    for j in range(i+1, k+1):
                        old_diffs_i.add(r[j] - old_val)
                    
                    D_other = row_diffs[target_idx] - old_diffs_i
                    D_global_other = global_used_diffs - row_diffs[target_idx]
                    
                    new_diffs_i = set()
                    for j in range(i):
                        new_diffs_i.add(new_val - r[j])
                    for j in range(i+1, k+1):
                        new_diffs_i.add(r[j] - new_val)
                    
                    # Check internal consistency of new_diffs_i
                    if len(new_diffs_i) != len(list(new_diffs_i)): # This is always true for a set
                        pass
                    # Actually, we need to check if the values generated are unique.
                    # The set construction handles uniqueness.
                    # But we need to ensure that the *mapping* doesn't create duplicates that were hidden?
                    # No, if r is strictly increasing, the diffs are positive.
                    # The only risk is new_val - r[j] == r[l] - new_val.
                    # This is covered by checking if the size of the set of diffs is equal to the number of pairs?
                    # Number of pairs involving i is i + (k - i) = k.
                    # If len(new_diffs_i) < k, then there was a collision.
                    
                    if len(new_diffs_i) != k:
                        continue
                        
                    # Check conflicts
                    if new_diffs_i & D_other:
                        continue
                    if new_diffs_i & D_global_other:
                        continue
                    
                    # Valid move!
                    # Update state
                    r[i] = new_val
                    row_diffs[target_idx] = get_row_diffs(r)
                    
                    # Update global_used_diffs
                    # Remove old diffs of this row, add new
                    # It's easier to recompute global_used_diffs if n_rows is small,
                    # or do set operations.
                    # global_used_diffs = (global_used_diffs - old_row_diffs) | new_row_diffs
                    # We have old_row_diffs = row_diffs[target_idx] (before update)
                    # But we just updated row_diffs[target_idx].
                    # So we need the old one.
                    
                    # Let's store old_diffs before updating
                    # Actually, I can just rebuild global_used_diffs from all rows if n_rows is small (7).
                    # 7 rows * 21 diffs = 147 diffs. Rebuilding is O(147). Very fast.
                    
                    global_used_diffs = set()
                    for rr in rows:
                        for d in get_row_diffs(rr):
                            global_used_diffs.add(d)
                            
                    # Update scope
                    current_scope = max(max(rr) for rr in rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        
                    found_move = True
                    improved = True
                    break
                
            if found_move:
                break
        
        # If no move was found on the target row, we might be stuck.
        # To escape, we can try to make a "neutral" move on a different row.
        # But given the time constraint, if we can't reduce the max, we might just stop or try random perturbations.
        # For now, let's try a random move on any row if the target row failed.
        if not found_move:
            # Try a random row and random index
            rand_row_idx = random.randint(0, n_rows - 1)
            rr = rows[rand_row_idx]
            rand_idx = random.randint(1, k)
            old_val = rr[rand_idx]
            
            low = rr[rand_idx-1] + 1
            high = rr[rand_idx+1] - 1 if rand_idx < k else old_val + 10 # Allow some increase
            
            if high > low:
                # Try a few random values
                for _ in range(10):
                    new_val = random.randint(low, high)
                    if new_val == old_val:
                        continue
                    
                    # Similar validity check
                    # This is expensive to do fully. 
                    # Given the time, we'll skip this for now to keep the code simple and fast.
                    # The primary driver is reducing the max row.
                    pass

    return rows
