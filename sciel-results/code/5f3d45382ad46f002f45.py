import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to generate a row given a set of used differences
    def generate_row(used_diffs, k, randomize=False, p_smallest=0.7, window=5):
        row = [0]
        # We need k more elements
        for _ in range(k):
            candidates = []
            # We need to find x > row[-1] such that {x - a for a in row} are not in used_diffs
            # And also distinct within the row (which is implied if x-a are distinct and we check against used_diffs which includes previous row diffs? 
            # Wait, used_diffs contains diffs from ALL rows. 
            # Within the current row, we must ensure new diffs are distinct from each other too.
            # Since x is increasing, x - row[i] are distinct for different i.
            # So we just need to check against used_diffs.
            
            # Optimization: We don't know the next x. We iterate x upwards.
            # To make it faster, we can just iterate.
            x = row[-1] + 1
            found = False
            valid_xs = []
            
            # Limit search to avoid infinite loops if no solution exists in this branch
            # A reasonable upper bound for x in a single step?
            # The max difference is M. M is around 134.
            # So x will not exceed M.
            while x < 200: # Hard cap for safety
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    if len(valid_xs) >= window:
                        break
                    # If we want the smallest, we could break here if not randomizing, 
                    # but we want a window for randomization.
                x += 1
            
            if not valid_xs:
                return None # Failed to generate
            
            if randomize:
                if random.random() < p_smallest:
                    row.append(valid_xs[0])
                else:
                    row.append(random.choice(valid_xs))
            else:
                row.append(valid_xs[0])
                
            # Update used_diffs for the next step in this row generation
            # Note: We must add the new diffs to used_diffs locally for the next iteration
            # But we don't want to modify the global used_diffs until the row is complete and accepted.
            # So we maintain a local set or pass a copy? 
            # For efficiency, we can just keep the local `row` and check against `used_diffs` + `local_diffs`.
            # Actually, the check `new_diffs & used_diffs` is against the GLOBAL used diffs.
            # We also need to ensure `new_diffs` don't conflict with each other (they don't, as x is unique) 
            # or with previous diffs in the SAME row.
            # Previous diffs in the same row are NOT in `used_diffs` yet.
            # So we must check against `used_diffs` AND `current_row_diffs`.
            
            # Let's refactor generate_row to be correct.
            
        return row

    # Corrected generate_row
    def generate_row_correct(used_diffs, k, randomize=False, p_smallest=0.7, window=5):
        row = [0]
        current_row_diffs = set()
        
        for _ in range(k):
            candidates = []
            x = row[-1] + 1
            # We need to find x such that {x - a for a in row} are disjoint from (used_diffs U current_row_diffs)
            forbidden = used_diffs | current_row_diffs
            
            while x < 200:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
                
            row.append(next_x)
            # Update current_row_diffs
            current_row_diffs |= {next_x - a for a in row[:-1]}
            
        return row

    # Initial deterministic greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_row_diffs = set()
        for _ in range(k):
            x = row[-1] + 1
            forbidden = used | current_row_diffs
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    row.append(x)
                    current_row_diffs |= new_diffs
                    break
                x += 1
        rows.append(row)
        used |= current_row_diffs
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        # Pick a random row to regenerate
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx from used
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        # Regenerate
        new_row = generate_row_correct(used, k, randomize=True, p_smallest=0.8, window=3)
        
        if new_row is None:
            # Rollback
            used |= old_diffs
            continue
        
        # Check if new_row is valid against used (it should be by construction)
        # Calculate new max
        new_max_val = max(new_row)
        # The global max is max(max of other rows, new_max_val)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0:
            # Accept
            rows[r_idx] = new_row
            used |= set(d for i in range(len(new_row)) for j in range(i+1, len(new_row)) for d in [new_row[j]-new_row[i]])
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Accept with probability
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            
            if random.random() < prob:
                rows[r_idx] = new_row
                used |= set(d for i in range(len(new_row)) for j in range(i+1, len(new_row)) for d in [new_row[j]-new_row[i]])
                current_max = new_global_max
            else:
                # Rollback
                used |= old_diffs
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
