import random
import math
import time

def construct(n_rows, k):
    start_time = time.time()
    TIME_LIMIT = 115.0
    
    def greedy_init():
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    def get_max_val(rws):
        return max(max(r) for r in rws)

    def is_valid_move(rows, r, j, new_val, used_diffs):
        if j > 0 and new_val <= rows[r][j-1]:
            return False
        if j < k and new_val >= rows[r][j+1]:
            return False
        
        old_val = rows[r][j]
        if new_val == old_val:
            return True
            
        new_diffs = set()
        for idx, val in enumerate(rows[r]):
            if idx == j:
                continue
            d = abs(new_val - val)
            new_diffs.add(d)
            
        old_diffs = set()
        for idx, val in enumerate(rows[r]):
            if idx == j:
                continue
            d = abs(old_val - val)
            old_diffs.add(d)
            
        U_minus_old = used_diffs - old_diffs
        if new_diffs & U_minus_old:
            return False
            
        return True

    def apply_move(rows, r, j, new_val, used_diffs):
        old_val = rows[r][j]
        for val in rows[r]:
            if val != old_val:
                d = abs(old_val - val)
                used_diffs.discard(d)
        
        for val in rows[r]:
            if val != new_val:
                d = abs(new_val - val)
                used_diffs.add(d)
        
        rows[r][j] = new_val

    def revert_move(rows, r, j, old_val, used_diffs):
        new_val = rows[r][j]
        for val in rows[r]:
            if val != new_val:
                d = abs(new_val - val)
                used_diffs.discard(d)
        
        for val in rows[r]:
            if val != old_val:
                d = abs(old_val - val)
                used_diffs.add(d)
        
        rows[r][j] = old_val

    best_rows = None
    best_max = float('inf')
    
    num_starts = 50
    for start_iter in range(num_starts):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        rows, used_diffs = greedy_init()
        random.shuffle(rows)
        
        current_max = get_max_val(rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
            
        T = 50.0 + start_iter * 5.0
        cooling = 0.995
        iterations = 10000
        random.seed(42 + start_iter)
        
        for _ in range(iterations):
            if time.time() - start_time > TIME_LIMIT:
                break
            if T < 0.01:
                break
                
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            old_val = rows[r][j]
            
            min_v = rows[r][j-1] + 1
            max_v = rows[r][j+1] - 1 if j < k else current_max + 10
            
            if min_v > max_v:
                continue
                
            if random.random() < 0.7 and old_val > min_v:
                new_val = random.randint(min_v, old_val)
            else:
                new_val = random.randint(min_v, max_v)
                
            if new_val == old_val:
                continue
                
            if not is_valid_move(rows, r, j, new_val, used_diffs):
                continue
                
            apply_move(rows, r, j, new_val, used_diffs)
            new_max = get_max_val(rows)
            
            delta = new_max - current_max
            
            if delta < 0:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                if random.random() < math.exp(-delta / T):
                    current_max = new_max
                else:
                    revert_move(rows, r, j, old_val, used_diffs)
                    
            T *= cooling
            
    return best_rows
