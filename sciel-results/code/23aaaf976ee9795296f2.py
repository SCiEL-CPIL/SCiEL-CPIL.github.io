import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def randomized_greedy(window_size=5, bias=0.0):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                while len(candidates) < window_size:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                    x += 1
                    if x > 5000:
                        break
                
                if not candidates:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            break
                        x += 1
                
                if bias > 0:
                    weights = [1.0 / (c + 1) for c in candidates]
                    total = sum(weights)
                    r = random.random() * total
                    cumulative = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = i
                            break
                    chosen_x = candidates[chosen_idx]
                elif bias < 0:
                    weights = [c for c in candidates]
                    total = sum(weights)
                    r = random.random() * total
                    cumulative = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = i
                            break
                    chosen_x = candidates[chosen_idx]
                else:
                    chosen_x = random.choice(candidates)
                
                row.append(chosen_x)
                used_diffs.update({chosen_x - a for a in row[:-1]})
            
            rows.append(row)
        
        return rows, used_diffs
    
    def local_improve(rows, used_diffs):
        rows = [r[:] for r in rows]
        used_diffs = set(used_diffs)
        
        max_iters = 300
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            max_val = max(max(r) for r in rows)
            target_idx = None
            for i, r in enumerate(rows):
                if max(r) == max_val:
                    target_idx = i
                    break
            
            if target_idx is None:
                break
            
            target_row = rows[target_idx]
            last_elem = target_row[-1]
            lower_bound = target_row[-2] + 1
            
            old_diffs = {last_elem - a for a in target_row[:-1]}
            
            found = False
            for x in range(lower_bound, last_elem):
                new_diffs = {x - a for a in target_row[:-1]}
                
                conflict = False
                for d in new_diffs:
                    if d in used_diffs and d not in old_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    target_row[-1] = x
                    found = True
                    break
            
            if not found:
                break
        
        return rows, used_diffs
    
    best_rows = None
    best_max_val = float('inf')
    
    biases = [0.0, 0.5, -0.5, 1.0, -1.0]
    
    while time.time() - start_time < time_limit:
        bias = random.choice(biases)
        rows, used_diffs = randomized_greedy(bias=bias)
        rows, used_diffs = local_improve(rows, used_diffs)
        
        current_max = max(max(r) for r in rows)
        if current_max < best_max_val:
            best_max_val = current_max
            best_rows = [r[:] for r in rows]
    
    return best_rows
