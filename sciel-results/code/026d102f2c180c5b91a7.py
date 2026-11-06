"""Improved solver for increasing sequences with unique consecutive-block sums."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Greedy baseline first (fast, gives a guaranteed valid sequence)
    def greedy():
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

    best = greedy()
    best_len = len(best)

    # If we already have a good length, try to improve via local search / backtracking
    # The greedy approach tends to get stuck; we can try swapping or replacing elements.
    # Key insight: the constraint is that all O(n^2) consecutive block sums are distinct.
    # For n elements, there are n(n+1)/2 block sums, all in range [1, 256*n] roughly.
    # For n=15: 120 sums. For n=20: 210 sums. For n=25: 325 sums.
    # The maximum possible sum is 256*25 = 6400, so we have plenty of room.

    # Try a smarter construction: use larger gaps to avoid sum collisions.
    # A geometric-ish sequence might work well.
    
    def try_construct_with_gaps(gaps):
        """Try to build sequence using specified gap pattern."""
        values = []
        sums = set()
        for i, gap in enumerate(gaps):
            x = 1 + sum(gaps[:i]) if i > 0 else 1
            # Actually let's just use the gaps as differences
            if i == 0:
                x = 1
            else:
                x = values[-1] + gap
            if x > 256:
                break
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                return None
            values.append(x)
            sums.update(tails)
        return values

    # Try various gap patterns
    # Start with small gaps and increase
    import itertools
    
    # Try to extend the greedy result by backtracking on the last few elements
    # This is a local improvement approach
    
    def backtrack_improve(seq, max_depth=5, time_limit=70.0):
        """Try to extend or modify the tail of the sequence to get a longer one."""
        if time.time() > time_limit:
            return seq
        
        n = len(seq)
        # Try replacing the last k elements with different values
        for k in range(1, min(max_depth, n) + 1):
            if time.time() > time_limit:
                return seq
            
            prefix = seq[:n-k]
            sums_prefix = set()
            # Compute all block sums for prefix
            for i in range(len(prefix)):
                total = 0
                for j in range(i, len(prefix)):
                    total += prefix[j]
                    sums_prefix.add(total)
            
            # Now try to extend prefix with k new elements
            def extend(prefix, sums_so_far, remaining_k, start_val):
                if time.time() > time_limit:
                    return None
                if remaining_k == 0:
                    return []
                
                best_result = None
                for x in range(start_val, 257):
                    if time.time() > time_limit:
                        break
                    
                    # Check if x can be added
                    new_sums = set()
                    total = x
                    for a in reversed(prefix):
                        total += a
                        new_sums.add(total)
                    new_sums.add(x)
                    
                    if sums_so_far.intersection(new_sums):
                        continue
                    
                    # Recurse
                    result = extend(prefix + [x], sums_so_far | new_sums, remaining_k - 1, x + 1)
                    if result is not None:
                        return [x] + result
                
                return None
            
            result = extend(prefix, sums_prefix, k, prefix[-1] + 1 if prefix else 1)
            if result is not None:
                new_seq = prefix + result
                if len(new_seq) > len(seq):
                    seq = new_seq
        
        return seq

    # Try to improve by backtracking
    improved = backtrack_improve(best, max_depth=3, time_limit=deadline - 10)
    if len(improved) > best_len:
        best = improved
        best_len = len(best)

    return best
