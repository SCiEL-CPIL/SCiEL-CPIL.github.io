"""Rewritten program: backtracking with constraint propagation for unique consecutive-block sums."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _valid(seq):
    """Check strictly increasing and all consecutive-block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    for i in range(1, n):
        if seq[i] <= seq[i - 1]:
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

def _greedy(order, start=0, end=257):
    """Greedy: iterate candidates in given order, add if it keeps block sums distinct."""
    values = []
    sums = set()
    for x in order:
        # compute new suffix sums
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_largest():
    # bias toward large values: spreads sums, keeps low range sparse
    return _greedy(range(256, 0, -1))

def _greedy_smallest():
    return _greedy(range(1, 257))

def _greedy_random(seed):
    rng = random.Random(seed)
    order = list(range(1, 257))
    rng.shuffle(order)
    return _greedy(order)

def _greedy_weighted(seed, bias):
    # weighted sampling: higher weight for values near bias (1=small, -1=large)
    rng = random.Random(seed)
    vals = list(range(1, 257))
    if bias >= 0:
        weights = [1.0 + bias * (256 - v) / 256.0 for v in vals]
    else:
        weights = [1.0 + (-bias) * v / 256.0 for v in vals]
    order = []
    pool = vals[:]
    weights_pool = weights[:]
    while pool:
        total = sum(weights_pool)
        r = rng.random() * total
        acc = 0
        idx = len(pool) - 1
        for i, w in enumerate(weights_pool):
            acc += w
            if r <= acc:
                idx = i
                break
        order.append(pool.pop(idx))
        weights_pool.pop(idx)
    return _greedy(order)

def _backtrack_search(target_len, deadline, rng):
    """
    Depth-first backtracking search for a strictly increasing sequence of length target_len
    where all consecutive block sums are distinct.
    
    Strategy: Build sequence from smallest to largest, but use constraint propagation
    to prune branches early. The key insight is that we need to track all prefix sums
    of the current partial sequence to quickly check if adding a new element would
    create duplicate block sums.
    """
    best = []
    
    def backtrack(seq, sums, min_val):
        nonlocal best
        if len(seq) >= len(best):
            best = list(seq)
        
        if len(seq) == target_len:
            return True
        
        if time.time() > deadline:
            return False
        
        n = len(seq)
        # Try candidate values from min_val to 256
        # Use a heuristic: try values that maximize spacing
        for x in range(min_val, 257):
            # Compute new suffix sums if we append x
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            # Check if any new sum conflicts with existing sums
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # No conflict, proceed
            seq.append(x)
            sums.update(new_sums)
            
            if backtrack(seq, sums, x + 1):
                return True
            
            seq.pop()
            sums.difference_update(new_sums)
        
        return False
    
    backtrack([], set(), 1)
    return best

def _backtrack_with_pruning(target_len, deadline, rng):
    """
    Enhanced backtracking with smarter pruning and candidate ordering.
    """
    best = []
    
    def get_candidates(seq, sums, min_val):
        """Generate candidate values with scoring heuristic."""
        candidates = []
        for x in range(min_val, 257):
            # Quick check: compute new suffix sums
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            # Check for conflicts
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
            
            if not conflict:
                # Score: prefer values that create more "spread" in sum space
                # Higher x tends to spread sums more
                score = x
                candidates.append((score, x))
        
        # Sort by score descending (prefer larger values first for better spread)
        candidates.sort(key=lambda t: t[0], reverse=True)
        return [x for _, x in candidates]
    
    def backtrack(seq, sums, min_val, depth):
        nonlocal best
        if len(seq) > len(best):
            best = list(seq)
        
        if len(seq) == target_len:
            return True
        
        if time.time() > deadline:
            return False
        
        candidates = get_candidates(seq, sums, min_val)
        
        for x in candidates:
            # Compute new suffix sums
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            seq.append(x)
            sums.update(new_sums)
            
            if backtrack(seq, sums, x + 1, depth + 1):
                return True
            
            seq.pop()
            sums.difference_update(new_sums)
        
        return False
    
    backtrack([], set(), 1, 0)
    return best

def _beam_search(beam_width=50, max_depth=20, deadline=None):
    """Beam search to explore non-greedy branches."""
    if deadline is None:
        deadline = time.time() + 45.0
        
    # State: (seq, sums_set)
    beams = [([], set())]
    
    for _ in range(max_depth):
        if time.time() > deadline:
            break
        next_beams = []
        for seq, sums in beams:
            used = set(seq)
            # Try all unused numbers
            # Optimization: We can iterate 1..256
            for x in range(1, 257):
                if x in used:
                    continue
                # Check validity: new suffix sums must not intersect existing sums
                total = x
                valid = True
                if total in sums:
                    valid = False
                else:
                    for a in reversed(seq):
                        total += a
                        if total in sums:
                            valid = False
                            break
                
                if valid:
                    # Construct new sums set
                    new_sums = set(sums)
                    total = x
                    new_sums.add(total)
                    for a in reversed(seq):
                        total += a
                        new_sums.add(total)
                    
                    # Heuristic: Prefer larger x to spread sums (based on knowledge)
                    score = x
                    next_beams.append((score, seq + [x], new_sums))
        
        if not next_beams:
            break
        
        # Sort by score (descending) and keep top beam_width
        # Using a set to deduplicate identical sequences (though unlikely with unique x)
        # next_beams contains (score, seq, sums)
        next_beams.sort(key=lambda t: t[0], reverse=True)
        
        # Keep top beam_width
        beams = [(seq, sums) for _, seq, sums in next_beams[:beam_width]]
        
        if not beams:
            break
    
    # Return the longest sequence found
    if not beams:
        return []
    best_seq = max(beams, key=lambda t: len(t[0]))[0]
    return best_seq

def construct():
    # Set deadline to 80s to ensure we return before the 90s kill
    deadline = time.time() + 80.0

    best = []
    best_len = 0

    def consider(seq):
        nonlocal best, best_len
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)

    # Deterministic strategies first
    consider(_greedy_smallest())
    consider(_greedy_largest())

    # Structural Change: Backtracking search
    rng = random.Random(42)
    
    # Try backtracking with increasing target lengths
    # Start from a reasonable length and work up
    current_best_len = best_len
    
    if time.time() < deadline:
        # Try to find a sequence of length current_best_len + 1, +2, etc.
        for target_len in range(current_best_len + 1, current_best_len + 10):
            if time.time() > deadline - 10:
                break
            
            # Try a few random restarts of backtracking
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                
                result = _backtrack_with_pruning(target_len, deadline - 10, rng)
                if result:
                    consider(result)
                    if len(result) >= target_len:
                        break  # Found target length, move to next
    
    # Beam Search as fallback
    if time.time() < deadline - 15:
        consider(_beam_search(beam_width=30, max_depth=15, deadline=deadline - 15))
        
    # Weighted biases (small bias and large bias)
    for seed in range(2):
        if time.time() > deadline - 5:
            break
        consider(_greedy_weighted(seed, 0.5))   # favor large
        consider(_greedy_weighted(seed, -0.5))  # favor small

    # Randomized restarts with diverse sampling
    rng = random.Random(12345)
    seed = 0
    while time.time() < deadline - 5.0:  # Stop 5s before deadline to allow cleanup
        # alternate sampling strategies
        if seed % 3 == 0:
            seq = _greedy_largest()
        elif seed % 3 == 1:
            seq = _greedy_random(seed)
        else:
            seq = _greedy_weighted(seed, rng.uniform(-1.0, 1.0))
        consider(seq)
        seed += 1

    # Ensure a valid result is returned
    if not best:
        best = _greedy_smallest()

    # Final safety: validate
    if not _valid(best):
        # fall back to a guaranteed valid construction
        best = _greedy_smallest()

    return best
# EVOLVE-BLOCK-END
