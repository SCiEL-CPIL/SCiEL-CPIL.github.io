import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Greedy Initialization
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
    
    # Calculate initial max
    current_max = max(max(row) for row in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    best_used = set(used) # Not strictly needed if we recompute or maintain carefully, but for SA we maintain state
    
    # State for SA
    # We maintain `used` as a set of integers.
    # We also need to be able to undo moves or recalculate.
    # Given k is small, recalculating diffs for a row is fast.
    # But checking global conflict is O(1) if we have the set.
    
    # Let's define a function to get the cost
    # Cost = max_value * 1000 + num_conflicts
    # Actually, we want valid solutions first.
    # If we restrict moves to only valid ones, we can't escape local minima easily.
    # So we allow invalid moves with a penalty.
    
    def get_cost(rows, used_diffs):
        max_val = max(max(r) for r in rows)
        # Count conflicts? 
        # It's easier to just check validity in the move generation if we maintain the set correctly.
        # But for SA, we need to accept invalid states.
        # So we need a way to count conflicts.
        # Re-counting conflicts is O(n_rows * k^2). 7 * 36 = 252. This is fine.
        
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        
        # Count duplicates
        from collections import Counter
        c = Counter(all_diffs)
        conflicts = sum(count - 1 for count in c.values() if count > 1)
        
        return max_val * 1000 + conflicts

    current_rows = [r[:] for r in rows]
    current_max = best_max
    current_conflicts = 0 # Greedy is valid
    
    # Precompute indices for moves
    # Move: pick row r, index i (1..k), new value v
    # v must be > row[i-1] and < row[i+1] (if i < k)
    # If i == k, v can be anything > row[k-1] and < current_max + some bound?
    # To reduce max, we should try v < current_max.
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)
        
        row = current_rows[r_idx]
        old_val = row[i_idx]
        
        # Determine range for new value
        low = row[i_idx - 1] + 1
        high = row[i_idx + 1] - 1 if i_idx < k else current_max + 10 # Allow expansion? No, we want to shrink.
        # Let's bias towards smaller values to reduce max.
        # If we want to reduce max, we should only try values < current_max.
        # But sometimes we need to move things around to allow reduction.
        # Let's try a range [low, min(high, current_max)]
        
        if low > high:
            T *= cooling
            continue
            
        # Bias: 80% chance to pick a value < current_max if possible
        # If i_idx == k, high is effectively infinity, but we cap at current_max to try to shrink.
        # If i_idx < k, high is fixed by next element.
        
        if i_idx == k:
            # Try to move the largest element down
            if low <= current_max:
                new_val = random.randint(low, current_max)
            else:
                new_val = random.randint(low, low + 5) # Fallback
        else:
            new_val = random.randint(low, high)
            
        if new_val == old_val:
            continue
            
        # Perform move
        row[i_idx] = new_val
        
        # Calculate new cost
        # We need to check if the row is still strictly increasing.
        # low/high logic ensures row[i-1] < new_val < row[i+1].
        
        new_cost = get_cost(current_rows, None) # We pass None to ignore used_diffs in get_cost, it recalcs
        old_cost = current_max * 1000 + current_conflicts
        
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_max = max(current_max, new_val) # Actually max of all rows
            # Recalc max properly
            current_max = max(max(r) for r in current_rows)
            
            # Recalc conflicts for state tracking? 
            # get_cost already calculated it. Let's just store the cost components or re-calc.
            # For simplicity, let's just track best valid solution.
            # We don't strictly need current_conflicts for acceptance if we use get_cost, 
            # but we need it for old_cost. 
            # Let's just re-calc old_cost every time? No, that's slow.
            # Let's maintain current_conflicts.
            
            # To maintain current_conflicts efficiently is hard with this simple get_cost.
            # Let's just use get_cost for both old and new? 
            # get_cost is ~250 ops. 100k steps * 500 ops = 50M ops. Python might be slow.
            # 120s is plenty for 100k-1M steps in Python if optimized.
            
            # Let's optimize get_cost or just use it.
            
            # Update state
            # We don't maintain 'used' set in this loop version, we just check validity via get_cost.
            # This is O(N^2) per step.
            
            # Check if valid and better
            if new_cost < 1000: # No conflicts
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
            
        else:
            # Reject move, revert
            row[i_idx] = old_val
            
        T *= cooling
        
    # Return best valid solution
    return best_rows

# The above draft has a flaw: get_cost is called twice per iteration (once for new, once for old if we don't store it).
# And get_cost is O(N^2).
# Let's refine the cost calculation to be faster or maintain the state.
# Actually, for N=7, K=5, N*K=35 elements. Pairs ~ 7*15 = 105 diffs.
# 105 diffs is very small. Counter on 105 items is fast.
# So get_cost is very fast.

# Let's rewrite cleanly.
