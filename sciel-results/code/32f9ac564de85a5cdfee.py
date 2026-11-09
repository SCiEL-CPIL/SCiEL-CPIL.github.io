"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_valid(seq):
    """Check if seq is strictly increasing and all consecutive block sums are distinct."""
    if not seq:
        return False
    if any(seq[i+1] <= seq[i] for i in range(len(seq)-1)):
        return False
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _compute_sums(seq):
    """Return set of all nonempty consecutive block sums."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _try_append(seq, sums, x):
    """Try to append x to seq. Returns (True, new_sums) if valid, else (False, None)."""
    tails = [x]
    total = x
    for a in reversed(seq):
        total += a
        tails.append(total)
    # Check for conflicts
    for t in tails:
        if t in sums:
            return False, None
    new_sums = sums | set(tails)
    return True, new_sums

def greedy_smallest(deadline):
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return values

def greedy_largest(deadline):
    values, sums = [], set()
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return sorted(values)

def backtracking_search(initial, deadline, max_nodes=500000):
    """Depth-first backtracking with greedy pruning."""
    if not initial:
        return []
    
    sums = _compute_sums(initial)
    best = list(initial)
    nodes = 0
    
    def dfs(seq, sums, candidates):
        nonlocal best, nodes
        if time.time() > deadline or nodes > max_nodes:
            return
        nodes += 1
        if len(seq) > len(best):
            best = list(seq)
        if not candidates:
            return
        
        # Try candidates in order of "promisingness" - smaller values first for more room
        for i, x in enumerate(candidates):
            ok, new_sums = _try_append(seq, sums, x)
            if ok:
                remaining = candidates[i+1:]
                seq.append(x)
                dfs(seq, new_sums, remaining)
                seq.pop()
                if time.time() > deadline or nodes > max_nodes:
                    return
    
    candidates = [c for c in range(initial[-1]+1, 257)]
    dfs(list(initial), sums, candidates)
    return best

def local_search(start, deadline):
    """
    Improved local search:
    1. Greedy extend.
    2. Iteratively try to replace elements (from end to start) with nearby values 
       that allow a better greedy extension.
    """
    if not start:
        return start
    
    current = list(start)
    best = list(current)
    
    # 1. Greedy extend
    sums = _compute_sums(current)
    for x in range(current[-1]+1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(current, sums, x)
        if ok:
            current.append(x)
            sums = new_sums
    if len(current) > len(best):
        best = list(current)
        
    # 2. Local improvement
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        # Try to modify each position from the end
        for i in range(len(current)-1, -1, -1):
            if time.time() > deadline - 1:
                break
            orig = current[i]
            lb = current[i-1] + 1 if i > 0 else 1
            ub = current[i+1] - 1 if i < len(current)-1 else 256
            
            # Try candidates
            candidates_to_try = set()
            for delta in range(1, 15):
                candidates_to_try.add(orig - delta)
                candidates_to_try.add(orig + delta)
            
            # Sort by distance to orig, or just iterate
            for new_val in sorted(candidates_to_try):
                if new_val < lb or new_val > ub or new_val == orig:
                    continue
                
                trial = current.copy()
                trial[i] = new_val
                
                # Quick check: increasing
                if i > 0 and trial[i] <= trial[i-1]: continue
                if i < len(trial)-1 and trial[i] >= trial[i+1]: continue
                
                # Check distinct sums
                sums_trial = set()
                valid = True
                for j in range(len(trial)):
                    s = 0
                    for k in range(j, len(trial)):
                        s += trial[k]
                        if s in sums_trial:
                            valid = False
                            break
                        sums_trial.add(s)
                    if not valid:
                        break
                
                if valid:
                    # Try to extend greedily
                    ext_seq = list(trial)
                    ext_sums = sums_trial
                    for x in range(ext_seq[-1]+1, 257):
                        if time.time() > deadline - 1:
                            break
                        ok, new_sums = _try_append(ext_seq, ext_sums, x)
                        if ok:
                            ext_seq.append(x)
                            ext_sums = new_sums
                    
                    if len(ext_seq) > len(best):
                        best = list(ext_seq)
                        current = ext_seq
                        improved = True
                        break
                    elif len(ext_seq) == len(best) and ext_seq[-1] < best[-1]:
                        # Same length, but smaller last element (more room)
                        best = list(ext_seq)
                        current = ext_seq
                        improved = True
                        break

    return best

def _try_extend_greedy(seq, sums, candidates, deadline):
    """Greedy extension of seq with candidates."""
    for x in candidates:
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(seq, sums, x)
        if ok:
            seq.append(x)
            sums = new_sums
    return seq, sums

def simulated_annealing(deadline):
    """SA approach: start with a good greedy solution, perturb and optimize."""
    # Start with greedy_smallest
    start = greedy_smallest(deadline)
    if not start:
        return start
    
    current = list(start)
    current_sums = _compute_sums(current)
    best = list(current)
    best_len = len(current)
    
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline and temperature > 0.1:
        # Generate perturbation: replace one element
        i = random.randint(0, len(current)-1)
        orig = current[i]
        # Try a few nearby values
        attempts = 0
        while attempts < 5:
            new_val = orig + random.randint(-5, 5)
            if new_val < 1 or new_val > 256:
                attempts += 1
                continue
            if i > 0 and new_val <= current[i-1]:
                attempts += 1
                continue
            if i < len(current)-1 and new_val >= current[i+1]:
                attempts += 1
                continue
            
            trial = current.copy()
            trial[i] = new_val
            
            # Check validity
            sums = set()
            valid = True
            for j in range(len(trial)):
                s = 0
                for k in range(j, len(trial)):
                    s += trial[k]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid:
                # Try to extend
                ext, ext_sums = _try_extend_greedy(trial, sums, [c for c in range(trial[-1]+1, 257)], deadline)
                
                if len(ext) > best_len:
                    current = ext
                    current_sums = ext_sums
                    best = list(current)
                    best_len = len(current)
                elif len(ext) >= best_len:
                    # Accept with some probability for equal length
                    if random.random() < 0.1:
                        current = ext
                        current_sums = ext_sums
                break
            attempts += 1
        
        temperature *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # Generate diverse starting points
    candidates = []
    
    # Strategy 1: Greedy smallest
    if time.time() < deadline:
        c1 = greedy_smallest(deadline)
        candidates.append(c1)
    
    # Strategy 2: Greedy largest
    if time.time() < deadline:
        c2 = greedy_largest(deadline)
        candidates.append(c2)
    
    # Strategy 3: Multiple greedy variants with different starting points
    for start in range(1, 50, 3):
        if time.time() > deadline - 5:
            break
        values, sums = [], set()
        for x in range(start, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(values, sums, x)
            if ok:
                values.append(x)
                sums = new_sums
        if values:
            candidates.append(values)
    
    # Find best candidate so far
    best = max(candidates, key=len) if candidates else []
    
    # Refine best candidates using local search
    if time.time() < deadline - 10 and best:
        ls_result = local_search(best, deadline)
        if len(ls_result) > len(best):
            best = ls_result
            
    # Refine with backtracking
    if time.time() < deadline - 5 and best:
        bt_result = backtracking_search(best, deadline, max_nodes=200000)
        if len(bt_result) > len(best):
            best = bt_result
    
    # Final greedy extension
    if best and time.time() < deadline:
        sums = _compute_sums(best)
        for x in range(best[-1]+1, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(best, sums, x)
            if ok:
                best.append(x)
                sums = new_sums
    
    return best if best else greedy_smallest(deadline)
# EVOLVE-BLOCK-END
