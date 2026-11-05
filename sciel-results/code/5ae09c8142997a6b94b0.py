"""Structural change: backtracking + simulated annealing for better exploration."""
import time
import random
import sys

def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _try_append(seq, x, sums):
    """Try to append x to seq. Returns (success, new_sums) if successful."""
    if not seq or x > seq[-1]:
        total = x
        tails = [x]
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            new_sums = sums | set(tails)
            return True, new_sums
    return False, sums

def _greedy_build(candidates, existing, sums):
    """Greedy append with given candidate order."""
    seq = list(existing)
    s = set(sums)
    for x in candidates:
        if not seq or x > seq[-1]:
            total = x
            tails = [x]
            for a in reversed(seq):
                total += a
                tails.append(total)
            if not s.intersection(tails):
                seq.append(x)
                s.update(tails)
    return seq, s

def _backtracking_search(seq, sums, remaining, max_depth, deadline, rng):
    """
    Backtracking search to extend the sequence.
    Try to add more elements by exploring different choices.
    Returns the best sequence found.
    """
    best_seq = list(seq)
    best_len = len(best_seq)
    
    if time.time() > deadline or max_depth <= 0 or not remaining:
        return best_seq
    
    # Sort remaining candidates in a smart order
    # Prefer candidates that are "spread out" from existing
    sorted_rem = sorted(remaining, key=lambda x: (abs(x - (seq[-1] if seq else 128)), x))
    
    for x in sorted_rem:
        if time.time() > deadline:
            break
        if not seq or x > seq[-1]:
            total = x
            tails = [x]
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            if not sums.intersection(tails):
                new_seq = seq + [x]
                new_sums = sums | set(tails)
                new_remaining = [r for r in remaining if r != x]
                
                # Recursively try to extend
                sub_best = _backtracking_search(new_seq, new_sums, new_remaining, max_depth - 1, deadline, rng)
                if len(sub_best) > best_len:
                    best_seq = sub_best
                    best_len = len(best_seq)
    
    return best_seq

def _simulated_annealing(seq, sums, deadline, rng):
    """
    Simulated annealing: try to improve by removing and re-adding elements.
    Key move: remove one element and try to replace it with a better one,
    or remove last k elements and rebuild.
    """
    best_seq = list(seq)
    best_sums = set(sums)
    best_len = len(best_seq)
    
    temperature = 50.0
    cooling = 0.95
    min_temp = 0.5
    
    while time.time() < deadline and temperature > min_temp:
        # Generate a neighbor
        if len(best_seq) < 2:
            break
        
        # Move 1: Remove last element and try a different one
        if rng.random() < 0.4:
            if len(best_seq) >= 2:
                removed = best_seq[-1]
                truncated = best_seq[:-1]
                tr_sums = _compute_sums(truncated)
                
                # Try to add a different element
                used = set(truncated)
                candidates = [x for x in range(1, 257) if x not in used and x > truncated[-1]]
                if candidates:
                    # Try several candidates
                    rng.shuffle(candidates)
                    for x in candidates[:20]:
                        success, new_sums = _try_append(truncated, x, tr_sums)
                        if success:
                            new_seq = truncated + [x]
                            new_len = len(new_seq)
                            # Accept if better, or with probability if worse
                            if new_len >= best_len:
                                best_seq = new_seq
                                best_sums = new_sums
                                best_len = new_len
                            else:
                                # Simulated annealing acceptance
                                delta = best_len - new_len
                                prob = min(1.0, temperature / max(delta, 1))
                                if rng.random() < prob:
                                    best_seq = new_seq
                                    best_sums = new_sums
                                    best_len = new_len
        # Move 2: Remove a random middle element and try to fill the gap
        elif rng.random() < 0.3:
            if len(best_seq) >= 3:
                idx = rng.randint(1, len(best_seq) - 2)  # Not first or last
                removed = best_seq[idx]
                new_seq = best_seq[:idx] + best_seq[idx+1:]
                new_sums = _compute_sums(new_seq)
                new_len = len(new_seq)
                
                # Try to add elements to extend
                used = set(new_seq)
                candidates = [x for x in range(1, 257) if x not in used]
                # Filter for valid positions
                valid_candidates = []
                for x in candidates:
                    # Check if x can be inserted at position idx
                    if idx < len(new_seq) and (idx == 0 or new_seq[idx-1] < x) and (idx >= len(new_seq) or x < new_seq[idx]):
                        # Check sum conflicts
                        # This is complex; for now, just try appending
                        pass
                
                # Simpler: just try to extend the end
                if new_seq:
                    ext_candidates = [x for x in candidates if x > new_seq[-1]]
                    if ext_candidates:
                        rng.shuffle(ext_candidates)
                        for x in ext_candidates[:10]:
                            success, new_sums2 = _try_append(new_seq, x, new_sums)
                            if success:
                                final_seq = new_seq + [x]
                                final_len = len(final_seq)
                                if final_len >= best_len:
                                    best_seq = final_seq
                                    best_sums = new_sums2
                                    best_len = final_len
                                else:
                                    delta = best_len - final_len
                                    prob = min(1.0, temperature / max(delta, 1))
                                    if rng.random() < prob:
                                        best_seq = final_seq
                                        best_sums = new_sums2
                                        best_len = final_len
        # Move 3: Try to insert a new element in the middle (if space allows)
        else:
            if len(best_seq) >= 1:
                # Try to insert between existing elements
                used = set(best_seq)
                # Find gaps
                for i in range(len(best_seq) - 1):
                    if best_seq[i+1] - best_seq[i] > 1:
                        gap_candidates = [x for x in range(best_seq[i]+1, best_seq[i+1]) if x not in used]
                        if gap_candidates:
                            rng.shuffle(gap_candidates)
                            for x in gap_candidates[:5]:
                                # Check if inserting x at position i+1 is valid
                                # Need to check all new consecutive block sums
                                new_seq = best_seq[:i+1] + [x] + best_seq[i+1:]
                                # Check strictly increasing
                                valid = True
                                for j in range(1, len(new_seq)):
                                    if new_seq[j] <= new_seq[j-1]:
                                        valid = False
                                        break
                                if valid:
                                    new_sums = _compute_sums(new_seq)
                                    # Check for duplicates
                                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                                        new_len = len(new_seq)
                                        if new_len >= best_len:
                                            best_seq = new_seq
                                            best_sums = new_sums
                                            best_len = new_len
                                        else:
                                            delta = best_len - new_len
                                            prob = min(1.0, temperature / max(delta, 1))
                                            if rng.random() < prob:
                                                best_seq = new_seq
                                                best_sums = new_sums
                                                best_len = new_len
                            break  # Only try one gap per iteration
        
        temperature *= cooling
    
    return best_seq, best_sums, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe within 90s
    
    rng = random.Random(42)
    
    best_seq = []
    best_sums = set()
    best_len = 0
    
    # Strategy 1: Simple greedy (ascending)
    seq, sums = _greedy_build(range(1, 257), [], set())
    if len(seq) > best_len:
        best_seq, best_sums, best_len = seq, sums, len(seq)
    
    # Strategy 2: Greedy with largest-first bias
    def greedy_largest():
        seq = []
        sums = set()
        remaining = set(range(1, 257))
        while remaining and time.time() < deadline:
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
    for trial in range(100):
        if time.time() > deadline - 15:
            break
        
        base = list(range(1, 257))
        rng.shuffle(base)
        
        if trial % 3 == 0:
            candidates = sorted(base)
        elif trial % 3 == 1:
            candidates = base
        else:
            candidates = base[:]
            rng.shuffle(candidates)
        
        seq, sums = _greedy_build(candidates, [], set())
        if len(seq) > best_len:
            best_seq, best_sums, best_len = seq, sums, len(seq)
    
    # Strategy 4: Backtracking search from best sequence
    if time.time() < deadline - 10:
        used = set(best_seq)
        remaining = [x for x in range(1, 257) if x not in used and (not best_seq or x > best_seq[-1])]
        if remaining:
            bt_result = _backtracking_search(best_seq, best_sums, remaining, 4, deadline - 5, rng)
            if len(bt_result) > best_len:
                best_seq = bt_result
                best_sums = _compute_sums(best_seq)
                best_len = len(best_seq)
    
    # Strategy 5: Simulated annealing
    if time.time() < deadline - 5:
        best_seq, best_sums, best_len = _simulated_annealing(best_seq, best_sums, deadline - 2, rng)
    
    # Final validation
    for i in range(1, len(best_seq)):
        if best_seq[i] <= best_seq[i-1]:
            seq, sums = _greedy_build(range(1, 257), [], set())
            return seq
    
    for x in best_seq:
        if x < 1 or x > 256:
            seq, sums = _greedy_build(range(1, 257), [], set())
            return seq
    
    block_sums = set()
    for i in range(len(best_seq)):
        s = 0
        for j in range(i, len(best_seq)):
            s += best_seq[j]
            if s in block_sums:
                seq, sums = _greedy_build(range(1, 257), [], set())
                return seq
            block_sums.add(s)
    
    return best_seq


if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
