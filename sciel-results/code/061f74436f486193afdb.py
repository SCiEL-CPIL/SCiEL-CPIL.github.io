"""Rewritten solver: randomized greedy with multiple strategies + local search polish."""
import time
import random
import sys

def _greedy_build(candidates, existing, sums):
    """
    Given a list of candidate values (sorted ascending), and current sequence 'existing'
    with set 'sums' of all consecutive block sums, try to append candidates in order.
    Returns (new_sequence, new_sums, success_count).
    """
    seq = list(existing)
    s = set(sums)
    added = 0
    for x in candidates:
        if not seq or x > seq[-1]:
            # Compute all new consecutive block sums ending at x
            total = x
            tails = [x]
            for a in reversed(seq):
                total += a
                tails.append(total)
            if not s.intersection(tails):
                seq.append(x)
                s.update(tails)
                added += 1
    return seq, s, added


def _greedy_from_scratch(candidates, limit=256):
    """
    Build a sequence from scratch using greedy with given candidate ordering.
    candidates: iterable of values to try (in order).
    """
    seq = []
    sums = set()
    for x in candidates:
        if x < 1 or x > limit:
            continue
        if not seq or x > seq[-1]:
            total = x
            tails = [x]
            for a in reversed(seq):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
    return seq, sums


def _local_search_polish(seq, sums, time_limit, rng):
    """
    Try to improve the sequence by:
    1. Trying to insert new elements at various positions (but must maintain strictly increasing).
    2. Actually, since the sequence must be strictly increasing, we can only append.
    3. Try replacing some elements with larger ones to free up sum space.
    
    For simplicity, we'll try:
    - Append more elements using a randomized greedy on remaining candidates.
    - Try to find a better ending by backtracking the last few elements.
    """
    best_seq = list(seq)
    best_sums = set(sums)
    best_len = len(best_seq)
    
    # Strategy: try to append more using randomized candidates
    used = set(best_seq)
    remaining = [x for x in range(1, 257) if x not in used and (not best_seq or x > best_seq[-1])]
    
    if not remaining:
        return best_seq, best_sums, best_len
    
    # Try multiple random orderings of remaining candidates
    attempts = 0
    while time.time() < time_limit and attempts < 200:
        attempts += 1
        # Shuffle remaining, but bias toward larger values to spread sums
        shuffled = remaining[:]
        # Weight toward larger values
        weights = [x**0.5 for x in shuffled]
        # Simple approach: just shuffle
        rng.shuffle(shuffled)
        
        new_seq, new_sums, added = _greedy_build(shuffled, best_seq, best_sums)
        if len(new_seq) > best_len:
            best_seq = new_seq
            best_sums = new_sums
            best_len = len(best_seq)
            # Update remaining
            used = set(best_seq)
            remaining = [x for x in range(1, 257) if x not in used and (not best_seq or x > best_seq[-1])]
            if not remaining:
                break
    
    # Try backtracking: remove last k elements and try different endings
    for k in range(1, min(6, best_len)):
        if time.time() > time_limit:
            break
        truncated = best_seq[:-k]
        # Recompute sums for truncated
        tr_sums = set()
        for i in range(len(truncated)):
            s = 0
            for j in range(i, len(truncated)):
                s += truncated[j]
                tr_sums.add(s)
        
        # Try to extend from truncated
        used = set(truncated)
        remaining = [x for x in range(1, 257) if x not in used and (not truncated or x > truncated[-1])]
        
        if not remaining:
            continue
        
        # Try multiple random orderings
        for _ in range(50):
            if time.time() > time_limit:
                break
            shuffled = remaining[:]
            rng.shuffle(shuffled)
            new_seq, new_sums, added = _greedy_build(shuffled, truncated, tr_sums)
            if len(new_seq) > best_len:
                best_seq = new_seq
                best_sums = new_sums
                best_len = len(best_seq)
    
    return best_seq, best_sums, best_len


def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s to be safe within 90s
    
    rng = random.Random(42)
    
    best_seq = []
    best_sums = set()
    best_len = 0
    
    # Strategy 1: Simple greedy (ascending)
    seq, sums = _greedy_from_scratch(range(1, 257))
    if len(seq) > best_len:
        best_seq, best_sums, best_len = seq, sums, len(seq)
    
    # Strategy 2: Greedy with largest-first bias
    # Try candidates in descending order, but we need strictly increasing sequence
    # So we pick the largest available that doesn't conflict
    # Actually, for strictly increasing, we must pick in ascending order of value.
    # The "largest bias" means: among candidates that can be added, pick the largest.
    
    def greedy_largest():
        seq = []
        sums = set()
        remaining = set(range(1, 257))
        while remaining and time.time() < deadline:
            # Find the largest x in remaining that can be appended
            best_x = None
            for x in sorted(remaining, reverse=True):
                if not seq or x > seq[-1]:
                    total = x
                    tails = [x]
                    for a in reversed(seq):
                        total += a
                        tails.append(total)
                    if not sums.intersection(tails):
                        best_x = x
                        break
            if best_x is None:
                break
            seq.append(best_x)
            sums.update(tails)
            remaining.discard(best_x)
        return seq, sums
    
    seq, sums = greedy_largest()
    if len(seq) > best_len:
        best_seq, best_sums, best_len = seq, sums, len(seq)
    
    # Strategy 3: Randomized restarts with different seeds
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random subset of candidates, biased
        base = list(range(1, 257))
        rng.shuffle(base)
        
        # Mix: some ascending, some random
        if trial % 3 == 0:
            candidates = sorted(base)
        elif trial % 3 == 1:
            # Bias toward larger
            candidates = sorted(base, key=lambda x: -x)
            # But we need to try in a way that allows strictly increasing
            # So we'll use the greedy_largest approach with this candidate set
            # Actually, let's just do random order and let greedy handle it
            candidates = base
        else:
            # Weighted: favor mid-to-large values
            candidates = base[:]
            rng.shuffle(candidates)
        
        seq, sums = _greedy_from_scratch(candidates)
        if len(seq) > best_len:
            best_seq, best_sums, best_len = seq, sums, len(seq)
    
    # Local search polish on best sequence
    if time.time() < deadline - 5:
        best_seq, best_sums, best_len = _local_search_polish(
            best_seq, best_sums, deadline - 2, rng
        )
    
    # Final validation
    # Verify strictly increasing
    for i in range(1, len(best_seq)):
        if best_seq[i] <= best_seq[i-1]:
            # Invalid, fall back to simple greedy
            seq, sums = _greedy_from_scratch(range(1, 257))
            return seq
    
    # Verify all values in [1, 256]
    for x in best_seq:
        if x < 1 or x > 256:
            seq, sums = _greedy_from_scratch(range(1, 257))
            return seq
    
    # Verify all consecutive block sums are distinct
    block_sums = set()
    for i in range(len(best_seq)):
        s = 0
        for j in range(i, len(best_seq)):
            s += best_seq[j]
            if s in block_sums:
                # Conflict found, fall back
                seq, sums = _greedy_from_scratch(range(1, 257))
                return seq
            block_sums.add(s)
    
    return best_seq


if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
