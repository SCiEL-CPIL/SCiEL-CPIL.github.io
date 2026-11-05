import random
import math
import time

def construct(n_rows, k):
    # 1. Initial Solution: Greedy
    # This provides a valid upper bound to start from.
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
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Helper to calculate cost and details
    def analyze(rows):
        m = max(max(r) for r in rows)
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        
        seen = set()
        conflicts = 0
        for d in all_diffs:
            if d in seen:
                conflicts += 1
            else:
                seen.add(d)
        
        return m, conflicts

    cur_m, cur_conf = analyze(rows)
    
    # Cost function:
    # We want to minimize M.
    # We strongly penalize conflicts.
    # If valid (conf=0), cost = M.
    # If invalid, cost = M + conflicts * penalty.
    # The penalty should be high enough to prioritize finding *a* valid solution,
    # but the search should be able to jump to higher M if it reduces conflicts significantly.
    # Actually, a better strategy for "navigating invalid states":
    # Cost = M + conflicts * 100.
    # If we find a valid solution with M=100, cost=100.
    # An invalid solution with M=50 and 1 conflict has cost 150.
    # An invalid solution with M=10 and 1 conflict has cost 110.
    # This encourages lowering M even if invalid, as long as conflicts don't explode.
    
    PENALTY = 50.0
    
    def get_cost(m, conf):
        return m + conf * PENALTY

    current_cost = get_cost(cur_m, cur_conf)
    
    # SA Parameters
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    import time
    start_time = time.time()
    time_limit = 115  # 120s limit, leave buffer
    
    # Pre-calculate differences for faster conflict checking?
    # For small k and n_rows, O(N^2) is fine.
    # N = n_rows * k(k+1)/2. For 4 rows, k=8, N = 4*36 = 144 diffs. Very fast.
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Generate Move
        r_idx = random.randint(0, n_rows - 1)
        
        # Store old state for potential revert
        old_row = rows[r_idx][:]
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Shift suffix
            if k == 0:
                continue
            j_idx = random.randint(1, k)  # 1 to k
            # Determine delta range based on potential space
            # Keep delta small to avoid large jumps that break ordering easily
            delta = random.choice([-2, -1, 1, 2])
            
            # Check ordering constraint: row[j-1] < row[j]+delta < row[j+1] (if j<k)
            # And row[k] + delta should be reasonable (not negative, etc.)
            if j_idx > 0:
                if rows[r_idx][j_idx-1] >= rows[r_idx][j_idx] + delta:
                    continue
            if j_idx < k:
                if rows[r_idx][j_idx] + delta >= rows[r_idx][j_idx+1]:
                    continue
            
            # Apply shift
            for j in range(j_idx, k+1):
                rows[r_idx][j] += delta
                
        elif move_type < 0.9:
            # Change single element
            j_idx = random.randint(1, k)
            # Try to move into a gap
            left = rows[r_idx][j_idx-1]
            right = rows[r_idx][j_idx+1] if j_idx < k else rows[r_idx][j_idx] + 10 # Arbitrary bound for last
            # Actually, for last element, we just want to increase or decrease if valid
            if j_idx == k:
                # Can decrease if > left + 1
                if rows[r_idx][j_idx] > left + 1:
                    delta = random.choice([-1, -2, 1, 2])
                    new_val = rows[r_idx][j_idx] + delta
                    if new_val > left:
                        rows[r_idx][j_idx] = new_val
                    else:
                        continue
                else:
                    # Can only increase
                    delta = random.choice([1, 2, 5])
                    rows[r_idx][j_idx] += delta
            else:
                # Must fit between left and right
                min_val = left + 1
                max_val = right - 1
                if min_val > max_val:
                    continue
                # Randomly pick a new value in the range, or step from current
                # Stepping is safer for local search
                delta = random.choice([-2, -1, 1, 2])
                new_val = rows[r_idx][j_idx] + delta
                if min_val <= new_val <= max_val:
                    rows[r_idx][j_idx] = new_val
                else:
                    # Try to snap to boundary if step was out of bounds?
                    # Or just reject
                    continue
                    
        else:
            # Row swap (Symmetry breaking)
            if n_rows > 1:
                other_idx = random.randint(0, n_rows - 1)
                if other_idx != r_idx:
                    rows[r_idx], rows[other_idx] = rows[other_idx], rows[r_idx]
                else:
                    continue
            else:
                continue

        new_m, new_conf = analyze(rows)
        new_cost = get_cost(new_m, new_conf)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            # Update best if valid
            if new_conf == 0 and new_m < best_max:
                best_max = new_m
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            rows[r_idx] = old_row
            
        T *= cooling
        
        # Optional: If we have a valid solution, we can periodically try to "tighten"
        # This is handled by the random moves potentially decreasing M.
        
    return best_rows
