"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _check_valid(seq):
    """Check if all consecutive block sums in seq are distinct."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _beam_search(width=200, max_time=45):
    """
    Beam search to find the longest valid sequence.
    State: (seq, sums_set, max_val)
    Heuristic: Prefer smaller max_val to allow more future elements.
    """
    start_time = time.time()
    
    # Initial state
    beam = [([], set(), 0)]
    best_seq = []
    best_len = 0
    
    while beam:
        if time.time() - start_time > max_time:
            break
            
        next_candidates = []
        
        for seq, sums_set, max_val in beam:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
            # Try adding next values
            for x in range(max_val + 1, 257):
                # Check validity
                new_sums = []
                s = 0
                for val in reversed(seq):
                    s += val
                    new_sums.append(s)
                new_sums.append(x)
                
                valid = True
                for ns in new_sums:
                    if ns in sums_set:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_seq = seq + [x]
                new_sums_set = sums_set.union(new_sums)
                new_max_val = x
                
                next_candidates.append((new_seq, new_sums_set, new_max_val))
        
        if not next_candidates:
            break
            
        # Select top `width` candidates
        # Sort by heuristic:
        # 1. Max Val (ascending)
        # 2. Sum of elements (ascending)
        next_candidates.sort(key=lambda state: (state[2], sum(state[0])))
        
        beam = next_candidates[:width]
        
    return best_seq

def construct():
    """
    Combines Beam Search with Random Restarts for robustness.
    """
    best = []
    
    # Strategy 1: Beam Search
    seq_beam = _beam_search(width=200, max_time=40)
    if len(seq_beam) > len(best):
        best = seq_beam
        
    # Strategy 2: Greedy Forward (fast, provides a baseline)
    values = []
    sums = set()
    for x in range(1, 257):
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
        
    # Strategy 3: Random Greedy Restarts
    for seed in range(100):
        rng = random.Random(seed)
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        S = []
        for x in candidates:
            candidate = sorted(S + [x])
            if _check_valid(candidate):
                S.append(x)
        if len(S) > len(best):
            best = S
            
    return best
# EVOLVE-BLOCK-END
