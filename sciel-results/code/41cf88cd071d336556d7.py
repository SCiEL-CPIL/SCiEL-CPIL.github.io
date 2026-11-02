import random
import math
import time

def construct(n_rows, k):
    start_time = time.time()
    timeout = 115.0
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initial solution
    rows = greedy_construct()
    
    # Build global difference counts
    # Max difference is bounded by the max element, which is ~112. 
    # Let's use a dictionary for safety.
    diff_counts = {}
    def add_diff(d):
        diff_counts[d] = diff_counts.get(d, 0) + 1
        
    def remove_diff(d):
        diff_counts[d] -= 1
        if diff_counts[d] == 0:
            del diff_counts[d]
            
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                add_diff(r[j] - r[i])
                
    current_max = max(r[-1] for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    def try_decrease_element(row_idx, elem_idx):
        """
        Attempts to decrease rows[row_idx][elem_idx] to the smallest possible value
        that is greater than rows[row_idx][elem_idx-1] and valid.
        Returns True if a change was made, False otherwise.
        """
        r = rows[row_idx]
        if elem_idx == 0:
            return False
            
        current_val = r[elem_idx]
        min_val = r[elem_idx - 1] + 1
        
        if current_val == min_val:
            return False # Cannot go lower
            
        # 1. Remove current differences
        # Differences with j < elem_idx
        for j in range(elem_idx):
            d = current_val - r[j]
            remove_diff(d)
        # Differences with j > elem_idx
        for j in range(elem_idx + 1, len(r)):
            d = r[j] - current_val
            remove_diff(d)
            
        # 2. Find smallest valid v' > min_val
        found_val = -1
        # We can search upwards from min_val
        # Limit search to current_val to ensure we only decrease
        for v_prime in range(min_val, current_val):
            valid = True
            # Check diffs with j < elem_idx
            for j in range(elem_idx):
                d = v_prime - r[j]
                if diff_counts.get(d, 0) > 0:
                    valid = False
                    break
            if not valid:
                continue
            # Check diffs with j > elem_idx
            for j in range(elem_idx + 1, len(r)):
                d = r[j] - v_prime
                if diff_counts.get(d, 0) > 0:
                    valid = False
                    break
            if valid:
                found_val = v_prime
                break
                
        if found_val == -1:
            # Revert: Add back the original differences
            for j in range(elem_idx):
                d = current_val - r[j]
                add_diff(d)
            for j in range(elem_idx + 1, len(r)):
                d = r[j] - current_val
                add_diff(d)
            return False
        else:
            # Commit the change
            r[elem_idx] = found_val
            # Add new differences
            for j in range(elem_idx):
                d = found_val - r[j]
                add_diff(d)
            for j in range(elem_idx + 1, len(r)):
                d = r[j] - found_val
                add_diff(d)
            return True

    # Main Loop
    # We iterate through rows and elements in a random order, trying to decrease them.
    # If no improvement in a full pass, we might need to "kick" by increasing an element?
    # For now, let's try multiple passes. If stuck, we can try a randomized restart of the greedy construction?
    # Or just keep trying. The greedy construction is usually close.
    
    # To make it more robust, we can try to decrease elements in a random order.
    # If we get stuck, we can try to "shuffle" the rows? No, rows are interchangeable.
    
    # Let's run a loop of "passes". In each pass, we try to decrease every element.
    # If no element can be decreased in a pass, we stop? 
    # Or we try a "random kick": pick a random element and increase it slightly to allow others to decrease?
    # Given the time limit, simple hill climbing might get stuck.
    # Let's add a "kick" mechanism: if no improvement in last X seconds, 
    # pick a random element and set it to a random valid value (possibly higher).
    
    last_improvement_time = start_time
    
    while time.time() - start_time < timeout:
        improved = False
        # Randomize the order of elements to try
        indices = [(r_idx, e_idx) for r_idx in range(n_rows) for e_idx in range(1, k + 1)]
        random.shuffle(indices)
        
        for r_idx, e_idx in indices:
            if try_decrease_element(r_idx, e_idx):
                improved = True
                last_improvement_time = time.time()
                # Update current max
                current_max = max(r[-1] for r in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
                    
        if not improved:
            # Stuck. Try a kick.
            # Pick a random row and element, and try to move it to a random valid position.
            # We allow increasing the value here.
            r_idx = random.randint(0, n_rows - 1)
            e_idx = random.randint(1, k)
            r = rows[r_idx]
            current_val = r[e_idx]
            min_val = r[e_idx - 1] + 1
            max_val = r[e_idx + 1] - 1 if e_idx < k else current_max + 10 # Allow some growth
            
            # Remove current diffs
            for j in range(e_idx):
                remove_diff(current_val - r[j])
            for j in range(e_idx + 1, len(r)):
                remove_diff(r[j] - current_val)
            
            # Try to find a random valid v_prime
            # We'll try a few random candidates
            found = False
            for _ in range(20):
                # Pick a random value in [min_val, max_val]
                # Bias towards smaller values? Uniform is fine for kick.
                v_prime = random.randint(min_val, max_val)
                if v_prime == current_val:
                    continue
                valid = True
                for j in range(e_idx):
                    d = v_prime - r[j]
                    if diff_counts.get(d, 0) > 0:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(e_idx + 1, len(r)):
                    d = r[j] - v_prime
                    if diff_counts.get(d, 0) > 0:
                        valid = False
                        break
                if valid:
                    found = True
                    r[e_idx] = v_prime
                    for j in range(e_idx):
                        add_diff(v_prime - r[j])
                    for j in range(e_idx + 1, len(r)):
                        add_diff(r[j] - v_prime)
                    break
                    
            if not found:
                # Revert
                for j in range(e_idx):
                    add_diff(current_val - r[j])
                for j in range(e_idx + 1, len(r)):
                    add_diff(r[j] - current_val)
            
            # If we kicked, we might have increased the max.
            # Continue loop to see if we can decrease it back.
            
        # If we are stuck for a long time, maybe restart with a different greedy seed?
        # But the greedy seed is deterministic. 
        # We can randomize the greedy construction order? 
        # The greedy construction builds rows one by one. 
        # We can try to construct rows in a random order? 
        # Actually, the rows are indistinguishable in the final set.
        
    return best_rows
