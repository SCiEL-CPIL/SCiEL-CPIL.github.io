import random
import math
import time

def construct(n_rows, k):
    # Max possible difference we care about. 
    # Theoretical bound is 112, best known 134. 
    # Let's allow up to 200 for safety during search.
    MAX_DIFF = 200
    
    def get_diff_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                mask |= (1 << (row[j] - row[i]))
        return mask

    def gen_row(used_mask, mode='greedy'):
        row = [0]
        local_mask = 0
        for _ in range(k):
            start_x = row[-1] + 1
            valid_candidates = []
            # Scan a window
            # Optimization: We can break early if we find enough candidates
            for x in range(start_x, start_x + 150):
                # Calculate new diffs mask
                new_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    if d > MAX_DIFF:
                        continue # Should not happen if window is small
                    bit = 1 << d
                    if (used_mask & bit) or (local_mask & bit):
                        conflict = True
                        break
                    new_mask |= bit
                
                if not conflict:
                    valid_candidates.append((x, new_mask))
                    if mode == 'greedy':
                        break
                    if len(valid_candidates) >= 5:
                        break
            
            if not valid_candidates:
                return None
            
            if mode == 'greedy':
                x, new_mask = valid_candidates[0]
            else:
                # Weighted random
                # Prefer smaller x
                weights = [1.0 / (c[0] - start_x + 1) for c in valid_candidates]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                x, new_mask = valid_candidates[-1]
                for c, w in zip(valid_candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        x, new_mask = c
                        break
            
            row.append(x)
            local_mask |= new_mask
            
        return row

    def get_max_val(rws):
        return max(max(r) for r in rws)

    start_time = time.time()
    timeout = 115.0
    
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution using bitsets
    def initial_greedy():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = gen_row(used_mask, mode='greedy')
            if row is None:
                return None, None
            rows.append(row)
            used_mask |= get_diff_mask(row)
        return rows, used_mask

    init_rows, init_used_mask = initial_greedy()
    if init_rows:
        best_max = get_max_val(init_rows)
        best_rows = [r[:] for r in init_rows]
    else:
        # Fallback
        return [[0] + list(range(1, k+1)) for _ in range(n_rows)]

    current_rows = [r[:] for r in init_rows]
    current_used_mask = init_used_mask
    current_max = best_max
    
    T = 15.0
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        if T < 0.5:
            break
            
        # Pick a row to reconstruct
        r = random.randint(0, n_rows - 1)
        
        old_row = current_rows[r]
        old_row_mask = get_diff_mask(old_row)
        
        # Remove old row's mask
        current_used_mask ^= old_row_mask # XOR works if we assume no overlap, but OR/AND is safer. 
        # Actually, since the row's diffs are unique in the set, we can just do:
        # But wait, if another row had the same diff? No, "all differences ... are distinct over all rows".
        # So the mask for the row is a subset of the global mask, and no other row has those bits.
        # So XOR is safe to remove.
        
        # Generate new row
        new_row = gen_row(current_used_mask, mode='random')
        
        if new_row is None:
            current_used_mask ^= old_row_mask
            T *= cooling
            continue
        
        new_row_mask = get_diff_mask(new_row)
        
        # Calculate new global max
        # We can optimize this: 
        # old_max was known. 
        # If old_row contained the global max, we need to recompute or know the 2nd max.
        # Simple recompute is O(N*K) = 4*8 = 32 ops. Very fast.
        temp_rows = current_rows[:]
        temp_rows[r] = new_row
        new_global_max = get_max_val(temp_rows)
        
        delta = new_global_max - current_max
        
        if delta < 0:
            # Accept
            current_rows[r] = new_row
            current_used_mask |= new_row_mask
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [row[:] for row in current_rows]
        else:
            # Accept with prob exp(-delta / T)
            if delta > 0 and random.random() < math.exp(-delta / T):
                current_rows[r] = new_row
                current_used_mask |= new_row_mask
                current_max = new_global_max
            else:
                # Revert
                current_used_mask ^= old_row_mask

        T *= cooling

    if best_rows is None:
        return init_rows
        
    return best_rows
