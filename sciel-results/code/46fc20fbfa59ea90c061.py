import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # First, get a valid baseline using deterministic greedy
    def greedy_deterministic():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Get initial solution
    try:
        current_rows = greedy_deterministic()
    except Exception:
        # Fallback: return empty or minimal valid
        return [[0] + list(range(1, k+1)) for _ in range(n_rows)]
    
    def get_score(rows):
        return max(max(row) for row in rows)
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def is_valid(rows):
        used = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Simulated annealing
    current_score = get_score(current_rows)
    best_rows = [row[:] for row in current_rows]
    best_score = current_score
    
    # Precompute used diffs for current solution
    current_used = get_used_diffs(current_rows)
    
    # Temperature schedule
    initial_temp = 50.0
    final_temp = 0.1
    cooling_rate = 0.999
    
    temp = initial_temp
    
    import math
    
    while time.time() < deadline:
        # Pick a random row and element (not the first)
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, k)  # indices 1..k
        
        old_val = current_rows[r_idx][e_idx]
        
        # Try to move this element to a smaller value
        # Find a valid new position
        # The new value must be > current_rows[r_idx][e_idx-1] and < current_rows[r_idx][e_idx+1] if e_idx < k
        # Actually, it must maintain strict increase.
        
        # Let's try values from old_val - 1 down to some lower bound
        # Lower bound: current_rows[r_idx][e_idx-1] + 1
        # Upper bound: old_val - 1 (we want to decrease)
        
        lower_bound = current_rows[r_idx][e_idx - 1] + 1
        upper_bound = old_val - 1
        
        if upper_bound < lower_bound:
            # Can't decrease, maybe try increase? No, we want to minimize max.
            # But increasing might allow others to decrease? For now, just skip.
            # Or we can try to increase slightly to allow a global rearrangement?
            # Let's stick to decreasing for now to reduce scope.
            # If we can't decrease, we might still want to shuffle to escape local minima.
            # Let's allow small random moves (increase or decrease) to explore.
            # But primarily we want to decrease the max.
            
            # If we can't decrease, let's try a random valid move (could be increase or decrease)
            # to escape local minima.
            search_range = 10
            candidates = []
            for delta in range(-search_range, search_range + 1):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= lower_bound:
                    continue
                if e_idx < k:
                    if new_val >= current_rows[r_idx][e_idx + 1]:
                        continue
                
                # Check if new_val creates conflicts
                # Remove old diffs involving this element
                old_diffs = set()
                for i in range(len(current_rows[r_idx])):
                    if i == e_idx:
                        continue
                    if i < e_idx:
                        old_diffs.add(old_val - current_rows[r_idx][i])
                    else:
                        old_diffs.add(current_rows[r_idx][i] - old_val)
                
                # Free these diffs
                for d in old_diffs:
                    current_used.discard(d)
                
                # Check new diffs
                new_diffs = set()
                for i in range(len(current_rows[r_idx])):
                    if i == e_idx:
                        continue
                    if i < e_idx:
                        new_diffs.add(new_val - current_rows[r_idx][i])
                    else:
                        new_diffs.add(current_rows[r_idx][i] - new_val)
                
                # Check if new_diffs are free
                if not (new_diffs & current_used):
                    candidates.append((new_val, new_diffs))
                
                # Restore old diffs
                current_used.update(old_diffs)
            
            if candidates:
                # Pick a random candidate
                new_val, new_diffs = random.choice(candidates)
                
                # Apply move
                old_diffs = set()
                for i in range(len(current_rows[r_idx])):
                    if i == e_idx:
                        continue
                    if i < e_idx:
                        old_diffs.add(old_val - current_rows[r_idx][i])
                    else:
                        old_diffs.add(current_rows[r_idx][i] - old_val)
                
                for d in old_diffs:
                    current_used.discard(d)
                current_used.update(new_diffs)
                
                current_rows[r_idx][e_idx] = new_val
                new_score = get_score(current_rows)
                
                # Accept or reject
                if new_score < current_score or random.random() < math.exp((current_score - new_score) / temp):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_rows = [row[:] for row in current_rows]
                else:
                    # Revert
                    for d in new_diffs:
                        current_used.discard(d)
                    current_used.update(old_diffs)
                    current_rows[r_idx][e_idx] = old_val
            else:
                # No valid move found in small range
                pass
            
        else:
            # Can decrease
            # Try to find the smallest valid value in [lower_bound, old_val - 1]
            # Or pick a random valid value in that range to explore more
            found_move = False
            
            # We can try to jump to a random valid position in the range
            # Or greedily try to decrease as much as possible?
            # Let's try a random valid position to escape local minima.
            search_range = min(20, old_val - lower_bound)
            if search_range <= 0:
                search_range = 1
            
            candidates = []
            for delta in range(1, search_range + 1):
                new_val = old_val - delta
                if new_val < lower_bound:
                    break
                
                # Check if new_val creates conflicts
                old_diffs = set()
                for i in range(len(current_rows[r_idx])):
                    if i == e_idx:
                        continue
                    if i < e_idx:
                        old_diffs.add(old_val - current_rows[r_idx][i])
                    else:
                        old_diffs.add(current_rows[r_idx][i] - old_val)
                
                # Free these diffs
                for d in old_diffs:
                    current_used.discard(d)
                
                # Check new diffs
                new_diffs = set()
                for i in range(len(current_rows[r_idx])):
                    if i == e_idx:
                        continue
                    if i < e_idx:
                        new_diffs.add(new_val - current_rows[r_idx][i])
                    else:
                        new_diffs.add(current_rows[r_idx][i] - new_val)
                
                # Check if new_diffs are free
                if not (new_diffs & current_used):
                    candidates.append((new_val, new_diffs))
                
                # Restore old diffs
                current_used.update(old_diffs)
            
            if candidates:
                # Pick a random candidate, biased towards larger delta (smaller new_val)
                # Weight by 1/delta or just uniform?
                # Let's pick uniformly for now.
                new_val, new_diffs = random.choice(candidates)
                
                # Apply move
                old_diffs = set()
                for i in range(len(current_rows[r_idx])):
                    if i == e_idx:
                        continue
                    if i < e_idx:
                        old_diffs.add(old_val - current_rows[r_idx][i])
                    else:
                        old_diffs.add(current_rows[r_idx][i] - old_val)
                
                for d in old_diffs:
                    current_used.discard(d)
                current_used.update(new_diffs)
                
                current_rows[r_idx][e_idx] = new_val
                new_score = get_score(current_rows)
                
                # Accept or reject
                if new_score < current_score or random.random() < math.exp((current_score - new_score) / temp):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_rows = [row[:] for row in current_rows]
                else:
                    # Revert
                    for d in new_diffs:
                        current_used.discard(d)
                    current_used.update(old_diffs)
                    current_rows[r_idx][e_idx] = old_val
        
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
    
    # Final verification
    if is_valid(best_rows):
        return best_rows
    else:
        # Fallback to greedy
        return greedy_deterministic()
