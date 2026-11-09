import time
import random
import sys
from collections import defaultdict

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

def _construct_greedy_randomized(seed):
    """Greedy with randomized value ordering."""
    rng = random.Random(seed)
    values, sums = [], set()
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    for x in candidates:
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
    return values

def _beam_search(deadline, beam_width=100, rng=None):
    """
    Beam search to find long sequences.
    State: (seq, sums, last_val)
    Heuristic: Prefer states with more extension options? Or just keep longest?
    Since all states at same depth have same length, we need a tie-breaker.
    We can prefer states with smaller last_val (more room) or more available next vals.
    """
    if rng is None:
        rng = random.Random()
    
    # Initial state
    current_beam = [([], set(), 0)]
    best_seq = []
    
    while time.time() < deadline:
        next_beam = []
        for seq, sums, last_val in current_beam:
            n = len(seq)
            # Generate all valid extensions
            for v in range(last_val + 1, 257):
                total = v
                new_sums_list = [total]
                valid = True
                for i in range(n - 1, -1, -1):
                    total += seq[i]
                    if total in sums:
                        valid = False
                        break
                    new_sums_list.append(total)
                if valid:
                    next_beam.append((seq + [v], sums.union(new_sums_list), v))
        
        if not next_beam:
            break
        
        # Score and prune
        # Heuristic: 
        # 1. Longer is better (all same length here)
        # 2. Smaller last_val is better (more room for future large gaps? No, gaps must increase. 
        #    Actually, smaller last_val means we haven't used up the range [1, 256] as much?
        #    Wait, gaps must be increasing. So if last_val is small, we have more small numbers available?
        #    No, we skipped them. 
        #    Actually, if last_val is large, we are forced to pick even larger numbers, which might conflict sooner.
        #    So smaller last_val is generally better for longevity.
        # 3. Number of available next steps? Expensive to compute.
        
        # Let's use last_val as primary tie-breaker (smaller is better).
        # And maybe add some randomness.
        
        next_beam.sort(key=lambda x: (x[2], rng.random()))
        
        # Keep top beam_width
        current_beam = next_beam[:beam_width]
        
        # Update best
        if len(current_beam[0][0]) > len(best_seq):
            best_seq = current_beam[0][0]
            
    return best_seq

def _local_repair(seq, deadline, rng):
    """Try to extend or repair a sequence."""
    if not _check_valid(seq):
        return seq
    
    best_seq = list(seq)
    
    # Try extending
    for _ in range(50):
        if time.time() > deadline:
            break
        last = best_seq[-1] if best_seq else 0
        candidates = []
        sums = set()
        n = len(best_seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += best_seq[j]
                sums.add(s)
        for v in range(last + 1, 257):
            total = v
            ok = True
            for i in range(n - 1, -1, -1):
                total += best_seq[i]
                if total in sums:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        
        if not candidates:
            break
        
        v = rng.choice(candidates)
        new_seq = best_seq + [v]
        if _check_valid(new_seq):
            best_seq = new_seq
    
    return best_seq

def _multi_restart(deadline):
    """Multiple strategies."""
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    # Beam search
    if time.time() < deadline - 10:
        rng = random.Random(123)
        seq = _beam_search(deadline - 10, beam_width=200, rng=rng)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
    # Local repair on best
    if time.time() < deadline - 5:
        rng = random.Random(456)
        repaired = _local_repair(best_seq, deadline - 5, rng)
        if len(repaired) > best_len:
            best_seq = repaired
            best_len = len(repaired)
            
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    try:
        result = _multi_restart(deadline)
        if len(result) > best_len:
            best = result
            best_len = len(best)
    except Exception:
        pass
    
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
