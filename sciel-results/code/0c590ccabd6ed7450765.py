import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140  # Safety margin
    
    # Parameters for the specific task are likely n_rows=7, k=5
    # But we handle generally.
    
    # Helper to calculate differences of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a row is internally valid (unique diffs)
    def is_row_valid(row):
        return len(get_diffs(row)) == len(row) * (len(row) - 1) // 2

    # Initialize using a randomized greedy approach
    # This is better than the deterministic seed because it might find a more compact start
    def random_greedy_init():
        for _ in range(100): # Try a few times to get a valid set
            rows = []
            used_diffs = set()
            success = True
            for _ in range(n_rows):
                row = [0]
                # Try to extend row
                stuck = False
                while len(row) < k + 1:
                    # Find valid next values
                    # We need x > row[-1]
                    # New diffs: {x - a for a in row}
                    # Must not intersect used_diffs
                    # Must not intersect existing diffs of row
                    existing_diffs = get_diffs(row)
                    
                    # Search for x
                    # Limit search to avoid infinite loop, but for small k it's fine
                    # Heuristic: try values close to row[-1] first to keep it compact
                    found = False
                    # Randomize the order of checking x to break symmetry
                    # But we want compact, so we check small x first, but maybe skip some?
                    # Let's just check x from row[-1] + 1 upwards
                    # To add randomness, we can pick a random start offset or limit
                    # For now, deterministic small-x search is okay for init, 
                    # but randomization helps avoid local minima in init.
                    
                    # Let's try a window of values
                    # If we get stuck, we restart the whole init
                    # For k=5, max value is ~100, so search space is small.
                    
                    # To make it faster, we can pre-compute nothing, just loop.
                    # But we want randomness.
                    # Let's pick a random step size or random candidate.
                    
                    # Simple approach: Iterate x from row[-1]+1 to row[-1]+50
                    # If none found, mark stuck.
                    candidates = []
                    for x in range(row[-1] + 1, row[-1] + 100):
                        new_diffs = {x - a for a in row}
                        if new_diffs & used_diffs:
                            continue
                        if new_diffs & existing_diffs:
                            continue
                        # Check internal uniqueness of new_diffs (always true if row elements distinct)
                        # But wait, new_diffs are distinct from each other.
                        # We just need to ensure they don't clash with existing_diffs.
                        candidates.append(x)
                        if len(candidates) > 5: # Don't collect too many
                            break
                    
                    if not candidates:
                        stuck = True
                        break
                    
                    # Pick randomly from candidates to add diversity
                    x = random.choice(candidates)
                    row.append(x)
                    used_diffs |= {x - a for a in row[:-1]}
                    existing_diffs = get_diffs(row) # Update existing diffs for next step? 
                    # Actually existing_diffs in the loop above was calculated before appending x.
                    # We need to update it.
                
                if stuck:
                    success = False
                    break
                rows.append(row)
            
            if success:
                return rows
        return None

    # Get initial rows
    rows = random_greedy_init()
    if rows is None:
        # Fallback to deterministic seed
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
    
    # Calculate initial cost
    def get_cost(rows):
        return max(max(row) for row in rows)
    
    best_rows = [r[:] for r in rows]
    best_cost = get_cost(best_rows)
    
    # Precompute diffs for all rows to make updates faster
    row_diffs = [get_diffs(row) for row in rows]
    all_diffs = set()
    for d in row_diffs:
        all_diffs |= d
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    
    while time.time() - start_time < deadline:
        # Pick a random row
        i = random.randint(0, n_rows - 1)
        # Pick a random index to change (1 to k)
        j = random.randint(1, k)
        
        old_row = rows[i][:]
        old_val = old_row[j]
        
        # Determine range for new value
        # Must be > old_row[j-1] and < old_row[j+1] (if j < k)
        low = old_row[j - 1] + 1
        high = old_row[j + 1] - 1 if j < k else old_val + 50 # If last element, allow increase
        
        # Bias towards reducing max if j is the last element and it's the current max
        # Or just random within range
        # To encourage compactness, if j == k, we might want to try smaller values
        # But we can't go below low.
        
        # Let's try a few random candidates
        found_valid = False
        new_row = None
        new_diffs = None
        
        for _ in range(10): # Try a few random values
            # If j == k, we can try values lower than old_val to reduce cost
            # But must be > low
            if j == k:
                # Try to reduce
                if random.random() < 0.5:
                    # Try a value between low and old_val
                    if low < old_val:
                        new_val = random.randint(low, old_val - 1)
                    else:
                        new_val = random.randint(low, low + 10)
                else:
                    new_val = random.randint(low, high)
            else:
                new_val = random.randint(low, high)
            
            if new_val == old_val:
                continue
                
            # Construct new row
            new_row = old_row[:]
            new_row[j] = new_val
            
            # Check strict increase
            if new_row[j] <= new_row[j-1] or (j < k and new_row[j] >= new_row[j+1]):
                continue
            
            # Calculate new diffs
            new_diffs = get_diffs(new_row)
            
            # Check internal validity
            if len(new_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                continue
            
            # Check conflict with other rows
            # other_diffs = all_diffs - old_diffs[i]
            # if new_diffs & other_diffs: continue
            
            # Efficient check:
            # new_diffs must be disjoint from (all_diffs - row_diffs[i])
            # And new_diffs must be disjoint from old_diffs[i] ? No, new_diffs replaces old_diffs[i].
            # So we check new_diffs against (all_diffs - row_diffs[i])
            
            other_diffs = all_diffs - row_diffs[i]
            if new_diffs & other_diffs:
                continue
            
            # Valid move
            found_valid = True
            break
        
        if not found_valid:
            T *= cooling
            continue
        
        # Calculate new cost
        new_cost = get_cost(rows if i != 0 else [new_row] + rows[1:]) # This is inefficient
        # Better:
        # Only the max of row i might change, or the global max.
        # Let's just compute it.
        temp_rows = [r[:] for r in rows]
        temp_rows[i] = new_row
        new_cost = get_cost(temp_rows)
        
        old_cost = best_cost # No, old_cost should be current cost, not best
        # We need to track current cost.
        
        # Let's refactor to track current state properly.
        # The above loop is getting messy. Let's use a cleaner structure.
        
        # Break out of this messy draft and write clean code.
        break
        
    # Clean Implementation Below
    
    return best_rows

# RESTART WITH CLEAN IMPLEMENTATION
