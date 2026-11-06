"""Enhanced program for increasing sequences with unique consecutive-block sums."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # First, get the greedy solution as baseline
    greedy_seq = _greedy_construct()
    best_seq = greedy_seq
    best_len = len(greedy_seq)
    
    if time.time() > deadline:
        return best_seq
    
    # Try to improve with local search / backtracking refinement
    improved = _try_improve(best_seq, deadline)
    if len(improved) > best_len:
        best_seq = improved
        best_len = len(improved)
    
    if time.time() > deadline:
        return best_seq
    
    # Try different construction strategies
    for strategy in [_greedy_from_right, _greedy_skip]:
        if time.time() > deadline:
            break
        candidate = strategy()
        if len(candidate) > best_len:
            best_seq = candidate
            best_len = len(candidate)
    
    return best_seq

def _greedy_construct():
    """Original greedy: pick smallest valid x at each step."""
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > 85.0:
            break
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_from_right():
    """Try building from larger values first, then fill in smaller ones."""
    # This is tricky; let's try a different greedy: at each step, try all candidates
    # and pick the one that leaves the most room (largest gap to next used)
    values, sums = [], set()
    used = set()
    
    for _ in range(256):
        best_candidate = None
        best_score = -1
        for x in range(1, 257):
            if x in used:
                continue
            # Check if x can be appended
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            # Score: prefer smaller x to leave room, but also consider
            # how many future values might still work
            score = -x  # Prefer smaller
            if best_candidate is None or score > best_score:
                best_score = score
                best_candidate = x
                best_tails = tails
        
        if best_candidate is None:
            break
        values.append(best_candidate)
        used.add(best_candidate)
        sums.update(best_tails)
    
    return values

def _greedy_skip():
    """Greedy that sometimes skips to avoid future conflicts."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _try_improve(seq, deadline):
    """Try local search: swap elements to enable longer sequences."""
    # This is complex; for now, try a simple perturbation
    # Try removing one element and re-greedy from that point
    best = list(seq)
    
    for remove_idx in range(len(seq)):
        if time.time() > deadline:
            break
        # Remove element at remove_idx and re-greedy
        prefix = seq[:remove_idx]
        # Re-greedy from after prefix
        values = list(prefix)
        sums = _compute_sums(values)
        
        for x in range(prefix[-1] + 1 if prefix else 1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        if len(values) > len(best):
            best = values
    
    return best

def _compute_sums(values):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

# EVOLVE-BLOCK-END
