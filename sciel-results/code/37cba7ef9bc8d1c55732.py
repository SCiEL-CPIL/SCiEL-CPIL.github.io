import time
import random

def _backtrack_fixed_length(target_len, deadline, seed=42):
    """Find a valid sequence of exactly target_len elements using backtracking."""
    rng = random.Random(seed)
    values = []
    sums = set()
    
    def backtrack(pos, min_val):
        if time.time() > deadline:
            return False
        if pos == target_len:
            return True
        
        candidates = list(range(min_val, 257))
        rng.shuffle(candidates)
        
        for c in candidates:
            new_sums = []
            total = c
            new_sums.append(total)
            for a in reversed(values):
                total += a
                new_sums.append(total)
            
            if any(s in sums for s in new_sums):
                continue
            
            values.append(c)
            sums.update(new_sums)
            
            if backtrack(pos + 1, c + 1):
                return True
            
            values.pop()
            sums.difference_update(new_sums)
            
            if time.time() > deadline:
                return False
        
        return False
    
    if backtrack(0, 1):
        return values
    return None

def _greedy_smart(deadline):
    """Greedy with lookahead scoring."""
    values = []
    sums = set()
    
    while True:
        if time.time() > deadline:
            break
        
        last = values[-1] if values else 0
        best_candidate = None
        best_score = -1
        best_new_sums = None
        
        for c in range(last + 1, 257):
            new_sums = []
            total = c
            new_sums.append(total)
            for a in reversed(values):
                total += a
                new_sums.append(total)
            
            if any(s in sums for s in new_sums):
                continue
            
            score = 0
            for fc in range(c + 1, 257, 5):
                fc_sums = []
                fc_total = fc
                fc_sums.append(fc_total)
                for a in reversed(values):
                    fc_total += a
                    fc_sums.append(fc_total)
                if not any(s in sums or s in new_sums for s in fc_sums):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_candidate = c
                best_new_sums = set(new_sums)
        
        if best_candidate is None:
            break
        
        values.append(best_candidate)
        sums.update(best_new_sums)
    
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best = []
    
    for target_len in range(1, 25):
        if time.time() > deadline:
            break
        result = _backtrack_fixed_length(target_len, deadline, seed=42 + target_len)
        if result is not None and len(result) > len(best):
            best = result
    
    if time.time() < deadline:
        greedy_result = _greedy_smart(deadline)
        if len(greedy_result) > len(best):
            best = greedy_result
    
    if time.time() < deadline:
        values = []
        sums = set()
        for x in range(1, 257):
            if values and x <= values[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        if len(values) > len(best):
            best = values
    
    return best
