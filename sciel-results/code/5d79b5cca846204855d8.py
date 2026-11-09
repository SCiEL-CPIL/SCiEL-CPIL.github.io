"""Structural change: Golomb-ruler-based backtracking with increasing-gap constraint."""
import time
import random

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _golomb_backtrack(deadline):
    """
    Backtracking on prefix sums P[0..n] forming a Golomb ruler.
    Gaps g[i] = P[i+1] - P[i] must be strictly increasing and in [1,256].
    All pairwise differences P[j]-P[i] (i<j) must be distinct.
    Maximize n (number of gaps = sequence length).
    """
    best_seq = []
    best_len = 0
    
    # P[0] = 0 always
    # State: P list, diffs set, last_gap
    def backtrack(P, diffs, last_gap, depth):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return
        
        n = len(P) - 1  # number of gaps
        if n > best_len:
            best_len = n
            best_seq = [P[i+1] - P[i] for i in range(n)]
        
        # Try to extend
        # Next gap must be > last_gap and <= 256
        # New mark = P[-1] + gap
        # Check all new differences are unique
        
        start_gap = last_gap + 1
        for gap in range(start_gap, 257):
            new_mark = P[-1] + gap
            # Check all differences from new_mark to existing marks
            valid = True
            new_diffs = []
            for i in range(len(P)):
                d = new_mark - P[i]
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue
            
            # Recurse
            new_P = P + [new_mark]
            new_diffs_set = diffs | set(new_diffs)
            backtrack(new_P, new_diffs_set, gap, depth + 1)
            
            if time.time() > deadline:
                return
            
            # Pruning: if we've found a good solution, try to skip some gaps
            # Heuristic: if best_len >= 12, be more aggressive with pruning
            if best_len >= 12 and gap > last_gap + 3:
                # Skip some values to diversify
                if (gap - last_gap) % 2 == 0:
                    continue
    
    # Start with P = [0], no diffs, last_gap = 0
    backtrack([0], set(), 0, 0)
    
    return best_seq

def _golomb_beam(deadline):
    """
    Beam search on Golomb ruler prefix sums with increasing gap constraint.
    More parallel exploration than pure backtracking.
    """
    best_seq = []
    best_len = 0
    
    # Beam entries: (P_list, diffs_frozenset, last_gap)
    # Start: P=[0], diffs=set(), last_gap=0
    beam = [([0], frozenset(), 0)]
    K = 500  # beam width
    
    while time.time() < deadline:
        new_beam = []
        seen = set()
        
        for P, diffs, last_gap in beam:
            n = len(P) - 1
            if n > best_len:
                best_len = n
                best_seq = [P[i+1] - P[i] for i in range(n)]
            
            # Try extending
            start_gap = last_gap + 1
            # Limit number of candidates to keep beam manageable
            max_candidates = 30
            count = 0
            for gap in range(start_gap, 257):
                if count >= max_candidates:
                    break
                new_mark = P[-1] + gap
                valid = True
                new_diffs = []
                for i in range(len(P)):
                    d = new_mark - P[i]
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    count += 1
                    continue
                count += 1
                
                new_P = P + [new_mark]
                new_diffs_set = frozenset(diffs | set(new_diffs))
                key = (tuple(new_P), new_diffs_set)
                if key in seen:
                    continue
                seen.add(key)
                new_beam.append((new_P, new_diffs_set, gap))
        
        if not new_beam:
            break
        
        # Prune: sort by length desc, then by sum of P (smaller = more room)
        new_beam.sort(key=lambda x: (-len(x[0]), sum(x[0])))
        if len(new_beam) > K:
            new_beam = new_beam[:K]
        
        beam = new_beam
        
        # Early termination if beam is much shorter than best
        if beam and len(beam[0][0]) - 1 < best_len - 3:
            break
    
    return best_seq

def _simulated_annealing(deadline):
    """
    Simulated annealing: start from a valid sequence, try to insert new elements
    or modify existing ones to grow the sequence.
    """
    # Start from greedy smallest
    seq = _construct_greedy_smallest()
    if not seq:
        seq = _construct_powers()
    
    best_seq = list(seq)
    best_len = len(seq)
    
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            t = 0
            for j in range(i, len(s)):
                t += s[j]
                sums.add(t)
        return sums
    
    def is_valid(s):
        if not s:
            return True
        if s[0] < 1 or s[-1] > 256:
            return False
        for i in range(1, len(s)):
            if s[i] <= s[i-1]:
                return False
        return len(get_sums(s)) == len(s) * (len(s) + 1) // 2
    
    current = list(seq)
    current_sums = get_sums(current)
    T = 10.0  # initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            break
        
        # Random move: try to insert a new element at a random position
        n = len(current)
        if n >= 30:  # limit max length to avoid too slow checks
            break
        
        # Choose move type
        move_type = random.choice(['insert', 'replace', 'swap'])
        
        trial = list(current)
        
        if move_type == 'insert':
            # Try inserting a new element
            pos = random.randint(0, n)
            if pos == 0:
                lo, hi = 1, current[0] - 1 if n > 0 else 256
            elif pos == n:
                lo, hi = current[-1] + 1, 256
            else:
                lo, hi = current[pos-1] + 1, current[pos] - 1
            
            if lo > hi:
                continue
            
            val = random.randint(lo, hi)
            trial.insert(pos, val)
        
        elif move_type == 'replace':
            # Replace an element with a nearby one
            pos = random.randint(0, n - 1)
            if pos == 0:
                lo, hi = 1, current[0] - 1 if n > 1 else 256
            elif pos == n - 1:
                lo, hi = current[pos-1] + 1, 256
            else:
                lo, hi = current[pos-1] + 1, current[pos+1] - 1
            
            if lo > hi:
                continue
            
            new_val = random.randint(lo, hi)
            if new_val == current[pos]:
                continue
            trial[pos] = new_val
        
        else:  # swap
            if n < 2:
                continue
            i, j = random.sample(range(n), 2)
            trial[i], trial[j] = trial[j], trial[i]
            # Check if still strictly increasing
            for k in range(1, n):
                if trial[k] <= trial[k-1]:
                    trial = list(current)
                    break
        
        if not is_valid(trial):
            continue
        
        # Accept based on simulated annealing
        if len(trial) > len(current):
            current = trial
            current_sums = get_sums(current)
            if len(current) > best_len:
                best_len = len(current)
                best_seq = list(current)
        else:
            # Same length or shorter: accept with probability exp(-delta/T)
            delta = len(current) - len(trial)  # positive if getting worse
            if delta > 0:
                if random.random() < (2.718281828 ** (-delta / T)):
                    current = trial
                    current_sums = get_sums(current)
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    # Start with guaranteed valid baselines
    best = _construct_powers()
    best_len = len(best)
    
    if time.time() > deadline - 3:
        return best
    
    # Try greedy smallest (fast, deterministic)
    try:
        g = _construct_greedy_smallest()
        if len(g) > best_len:
            best = g
            best_len = len(g)
    except Exception:
        pass
    
    # Golomb ruler beam search (structural: works in prefix-sum space)
    try:
        gb = _golomb_beam(deadline)
        if len(gb) > best_len:
            best = gb
            best_len = len(gb)
    except Exception:
        pass
    
    # Simulated annealing on sequence space
    try:
        sa = _simulated_annealing(deadline)
        if len(sa) > best_len:
            best = sa
            best_len = len(sa)
    except Exception:
        pass
    
    # Final verification
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
