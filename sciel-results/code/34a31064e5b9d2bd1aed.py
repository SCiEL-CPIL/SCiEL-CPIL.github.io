import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # Initial greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Simulated Annealing
    current_solution = [r[:] for r in best_solution]
    current_score = best_score
    
    T = 10.0 # Initial temperature
    cooling = 0.99
    min_T = 0.1
    
    # Track time
    start_time = time.time()
    
    while time.time() < deadline - 5:
        if T < min_T:
            # Restart with higher T if we have time?
            # Or just stop.
            break
            
        neighbor, neighbor_score = _generate_neighbor(current_solution, n_rows, k)
        
        if neighbor is None:
            # No valid move found in sample
            T *= cooling
            continue
            
        delta = neighbor_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_solution = neighbor
            current_score = neighbor_score
            
            if current_score < best_score:
                best_score = current_score
                best_solution = [r[:] for r in current_solution]
                
        T *= cooling
        
    return best_solution

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 2000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _generate_neighbor(rows, n_rows, k):
    r_idx = random.randint(0, n_rows - 1)
    row = rows[r_idx]
    pos = random.randint(1, k)
    
    # Compute forbidden diffs
    forbidden = set()
    for i in range(n_rows):
        if i == r_idx: continue
        r = rows[i]
        for a in range(len(r)):
            for b in range(a+1, len(r)):
                forbidden.add(r[b] - r[a])
                
    for a in range(len(row)):
        if a == pos: continue
        for b in range(a+1, len(row)):
            if b == pos: continue
            forbidden.add(row[b] - row[a])
            
    low = row[pos - 1] + 1
    high = row[pos + 1] - 1 if pos < k else row[pos] + 50
    
    if high < low:
        return None, None
        
    candidates = []
    if high - low < 100:
        candidates = list(range(low, high + 1))
    else:
        candidates = random.sample(range(low, high + 1), 100)
        
    random.shuffle(candidates)
    
    for new_val in candidates:
        new_diffs = set()
        valid = True
        
        for i in range(pos):
            d = new_val - row[i]
            if d in forbidden or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
            
        if not valid: continue
        
        for j in range(pos + 1, len(row)):
            d = row[j] - new_val
            if d in forbidden or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
            
        if valid:
            new_row = row[:]
            new_row[pos] = new_val
            
            new_solution = [r[:] for r in rows]
            new_solution[r_idx] = new_row
            
            score = max(max(r) for r in new_solution)
            return new_solution, score
            
    return None, None
