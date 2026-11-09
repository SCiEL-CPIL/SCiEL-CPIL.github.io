"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

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

def beam_search(deadline, beam_width=50):
    """
    Beam search over sequence prefixes.
    State: (sequence_tuple, sum_set_frozen)
    At each step, expand each beam state with all valid next values,
    keep only top beam_width by (length, -last_value) to favor longer sequences
    with smaller last elements (more room to grow).
    """
    # Initial beam: single-element sequences
    beam = []
    for x in range(1, 257):
        beam.append(((x,), frozenset({x})))
    
    best_seq = []
    
    while time.time() < deadline:
        # Expand all states in beam
        next_beam = {}
        for seq_tuple, sums_frozen in beam:
            last_val = seq_tuple[-1]
            seq_list = list(seq_tuple)
            
            # Try all possible next values
            for x in range(last_val + 1, 257):
                ok, new_sums = _try_append(seq_list, set(sums_frozen), x)
                if ok:
                    new_seq = seq_tuple + (x,)
                    new_sums_frozen = frozenset(new_sums)
                    # Use the sequence itself as key to deduplicate
                    key = new_seq
                    if key not in next_beam:
                        next_beam[key] = (new_seq, new_sums_frozen)
        
        if not next_beam:
            break
        
        # Score each state: prefer longer, then smaller last element
        scored = []
        for seq_tuple, sums_frozen in next_beam.values():
            scored.append((len(seq_tuple), -seq_tuple[-1], seq_tuple, sums_frozen))
        
        # Sort and keep top beam_width
        scored.sort(reverse=True)
        beam = [(s, sf) for _, _, s, sf in scored[:beam_width]]
        
        # Update best
        best_in_beam = max(beam, key=lambda x: len(x[0]))
        if len(best_in_beam[0]) > len(best_seq):
            best_seq = list(best_in_beam[0])
        
        # If beam is empty or no progress, break
        if not beam:
            break
        
        # Check if we can still grow
        max_last = max(s[-1] for s, _ in beam)
        if max_last >= 256:
            # Can't grow any more from these states
            break
    
    return best_seq

def _greedy_extend_from(seq, deadline):
    """Greedily extend a sequence with smallest valid values."""
    if not seq:
        return []
    sums = _compute_sums(seq)
    current = list(seq)
    for x in range(current[-1] + 1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(current, sums, x)
        if ok:
            current.append(x)
            sums = new_sums
    return current

def _greedy_extend_randomized(seq, deadline, num_trials=10):
    """Try multiple greedy extensions with randomized value ordering."""
    if not seq:
        return []
    best = list(seq)
    
    for _ in range(num_trials):
        if time.time() > deadline:
            break
        current = list(seq)
        sums = _compute_sums(current)
        
        # Build candidate list with some randomization
        candidates = list(range(current[-1] + 1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(current, sums, x)
            if ok:
                current.append(x)
                sums = new_sums
        
        if len(current) > len(best):
            best = current
    
    return best

def _swap_local_search(seq, deadline, max_iterations=200):
    """
    Local search by swapping adjacent elements or replacing single elements
    to enable better greedy extensions.
    """
    if not seq or len(seq) < 2:
        return seq
    
    current = list(seq)
    best = list(current)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try replacing each element with a nearby value
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            orig = current[i]
            lb = current[i-1] + 1 if i > 0 else 1
            ub = current[i+1] - 1 if i < len(current) - 1 else 256
            
            # Try a few nearby values
            for delta in range(1, 8):
                for new_val in [orig - delta, orig + delta]:
                    if new_val < lb or new_val > ub:
                        continue
                    
                    trial = current.copy()
                    trial[i] = new_val
                    
                    # Validate
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
                    
                    if not valid:
                        continue
                    
                    # Try to extend
                    ext = _greedy_extend_from(trial, deadline)
                    
                    if len(ext) > len(best):
                        best = ext
                        current = ext
                        improved = True
                        break
                    elif len(ext) == len(best) and ext[-1] < best[-1]:
                        best = ext
                        current = ext
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # Phase 1: Get a good initial solution via multiple greedy strategies
    candidates = []
    
    # Greedy smallest
    if time.time() < deadline:
        c1 = greedy_smallest(deadline)
        candidates.append(c1)
    
    # Greedy largest
    if time.time() < deadline:
        c2 = greedy_largest(deadline)
        candidates.append(c2)
    
    # Multiple greedy starts
    for start in range(1, 100, 5):
        if time.time() > deadline - 10:
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
    
    best = max(candidates, key=len) if candidates else []
    
    # Phase 2: Beam search to find better structures
    if time.time() < deadline - 20:
        bs_result = beam_search(deadline - 5, beam_width=30)
        if len(bs_result) > len(best):
            best = bs_result
    
    # Phase 3: Local search on the best candidate
    if time.time() < deadline - 10 and best:
        ls_result = _swap_local_search(best, deadline - 5)
        if len(ls_result) > len(best):
            best = ls_result
    
    # Phase 4: Randomized greedy extensions
    if time.time() < deadline - 5 and best:
        rg_result = _greedy_extend_randomized(best, deadline, num_trials=20)
        if len(rg_result) > len(best):
            best = rg_result
    
    # Final greedy extension
    if best and time.time() < deadline:
        final = _greedy_extend_from(best, deadline)
        if len(final) > len(best):
            best = final
    
    return best if best else greedy_smallest(deadline)
# EVOLVE-BLOCK-END
