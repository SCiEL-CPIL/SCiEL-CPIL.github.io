import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe (90s total budget)
    
    n = 12
    # Initial best: powers of 2
    best_set = [1 << i for i in range(n)]
    best_max = max(best_set)  # 2048
    
    def _is_valid(arr):
        """Check if all 2^n subset sums are distinct."""
        if len(arr) != n:
            return False
        if len(set(arr)) != n:
            return False
        if any(x <= 0 for x in arr):
            return False
        if any(x > 1048576 for x in arr):
            return False
        sums = [0]
        for x in arr:
            new_sums = [s + x for s in sums]
            all_sums = sums + new_sums
            if len(set(all_sums)) != len(all_sums):
                return False
            sums = all_sums
        return True

    def _local_search(arr, max_time):
        """Try to reduce elements via local search. Returns improved arr."""
        arr = arr[:]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > max_time:
                    break
                old_val = arr[idx]
                # Try to decrease
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in set(arr):
                        continue
                    candidate = arr[:]
                    candidate[idx] = new_val
                    if _is_valid(candidate):
                        arr = candidate
                        improved = True
                        break
        return arr

    def _greedy_construct():
        """
        Greedy construction: iteratively select the smallest positive integer
        not yet used which preserves distinctness of all subset sums.
        This is the core strategy from the directive.
        """
        current = []
        used = set()
        sums = {0}
        
        for i in range(n):
            # Find smallest x > 0 not in used such that all subset sums remain distinct
            x = 1
            while True:
                if x in used:
                    x += 1
                    continue
                
                # Check if adding x preserves distinctness
                # New sums = sums U (sums + x)
                # They are disjoint iff x is not in the set of differences of sums
                # Diff set: {a - b | a, b in sums}
                # But we can just check directly:
                new_sums_set = set(sums)
                for s in sums:
                    if s + x in new_sums_set:
                        break
                    new_sums_set.add(s + x)
                else:
                    # No collision
                    current.append(x)
                    used.add(x)
                    sums = new_sums_set
                    break
                x += 1
                # Safety: if x gets too large, break (shouldn't happen for n=12)
                if x > 100000:
                    break
            
            if x > 100000:
                break
        
        return current if len(current) == n else None

    # Phase 1: Try greedy construction
    if time.time() < deadline - 30:
        greedy_result = _greedy_construct()
        if greedy_result is not None:
            # Polish it with local search
            polished = _local_search(greedy_result, deadline - 10)
            if _is_valid(polished):
                if max(polished) < best_max:
                    best_max = max(polished)
                    best_set = polished

    # Phase 2: Beam search with local search polish
    if time.time() < deadline - 20:
        # Beam search
        # State: (arr, sums_set)
        # We'll maintain a beam of partial sequences
        beam = [([1], {0, 1})]
        
        for step in range(1, n):
            if time.time() > deadline - 10:
                break
            
            next_beam = []
            for arr, sums in beam:
                # Greedy smallest valid x
                x = 1
                # We need to find candidates that work
                # To be efficient, we find the smallest valid x, then a few more
                candidates = []
                for x in range(1, 5000):
                    if x in arr:
                        continue
                    # Check collision
                    valid = True
                    for s in sums:
                        if s + x in sums:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= 20:
                            break
                
                for x in candidates:
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    # Since we checked, len(new_sums) should be 2*len(sums)
                    next_beam.append((arr + [x], new_sums))
            
            # Sort by max element and keep top beam_width
            next_beam.sort(key=lambda item: max(item[0]))
            beam_width = 5
            beam = next_beam[:beam_width]
        
        # Polish final beam entries
        for arr, sums in beam:
            if time.time() > deadline - 5:
                break
            if len(arr) == n:
                polished = _local_search(arr, deadline - 2)
                if _is_valid(polished):
                    if max(polished) < best_max:
                        best_max = max(polished)
                        best_set = polished

    # Phase 3: Random restarts with local search
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            # Start from best_set and perturb
            arr = best_set[:]
            random.shuffle(arr)
            # Try to reduce each element
            for idx in range(n):
                if time.time() > deadline - 3:
                    break
                old_val = arr[idx]
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in set(arr):
                        continue
                    candidate = arr[:]
                    candidate[idx] = new_val
                    if _is_valid(candidate):
                        arr = candidate
                        break
            if _is_valid(arr):
                if max(arr) < best_max:
                    best_max = max(arr)
                    best_set = arr

    return best_set
