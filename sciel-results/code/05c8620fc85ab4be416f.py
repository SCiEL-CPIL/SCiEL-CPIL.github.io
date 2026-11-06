import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s kill

    def validate_seq(seq):
        """Validate a sequence: strictly increasing, in [1,256], all consecutive block sums distinct"""
        n = len(seq)
        if n == 0:
            return True
        for i in range(n):
            if seq[i] < 1 or seq[i] > 256:
                return False
            if i > 0 and seq[i] <= seq[i-1]:
                return False
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def can_add_fast(prefix_sums, used_sums, x, last_val):
        """Check if x can be appended given current prefix_sums and used_sums.
        Returns (new_prefix, new_sums) if valid, else None."""
        if x <= last_val or x > 256:
            return None
        new_prefix = prefix_sums[-1] + x
        new_sums = []
        for p in prefix_sums:
            s = new_prefix - p
            if s in used_sums:
                return None
            new_sums.append(s)
        # Check new_sums are all distinct (they should be since prefix_sums are distinct)
        # but also check they don't collide with each other
        if len(set(new_sums)) != len(new_sums):
            return None
        return (new_prefix, new_sums)
    
    def greedy_smallest(start_val=1):
        """Greedy: always pick the smallest valid next element, using fast can_add"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        n = 0
        while n < 256:
            found = False
            for x in range(last_val + 1, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    new_prefix, new_sums = result
                    seq.append(x)
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    last_val = x
                    n += 1
                    found = True
                    break
            if not found:
                break
        return seq
    
    def greedy_randomized(start_val=1, seed_offset=0):
        """Randomized greedy: at each step, pick from valid candidates with bias"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        rng = random.Random(seed_offset)
        n = 0
        while n < 256:
            valid_candidates = []
            for x in range(last_val + 1, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    valid_candidates.append((x, result))
            if not valid_candidates:
                break
            # Pick with bias toward smaller values but with randomness
            # Use weighted random: lower values get higher weight
            if len(valid_candidates) <= 3:
                x, (new_prefix, new_sums) = valid_candidates[0]
            else:
                # Weighted selection: favor smaller values
                weights = [1.0 / (i + 1) for i in range(len(valid_candidates))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = i
                        break
                x, (new_prefix, new_sums) = valid_candidates[chosen]
            seq.append(x)
            prefix_sums.append(new_prefix)
            used_sums.update(new_sums)
            last_val = x
            n += 1
        return seq
    
    def csp_backtrack_from(start_seq, target_len, max_time=5.0):
        """CSP-style backtracking extending from start_seq to target_len."""
        seq = list(start_seq)
        prefix_sums = [0]
        for v in seq:
            prefix_sums.append(prefix_sums[-1] + v)
        used_sums = set()
        for i in range(len(seq)):
            for j in range(i+1, len(seq)+1):
                used_sums.add(prefix_sums[j] - prefix_sums[i])
        
        deadline = start_time + max_time
        found = [False]
        last_val = seq[-1] if seq else 0

        def dfs(depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return

            cur_last = seq[-1] if seq else 0
            cur_prefix = prefix_sums[-1]

            for x in range(cur_last + 1, 257):
                if found[0] or time.time() > deadline:
                    return

                new_prefix = cur_prefix + x
                new_sums = []
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)

                if not valid:
                    continue

                if len(set(new_sums)) != len(new_sums):
                    continue

                # Commit
                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(depth + 1)

                # Backtrack
                seq.pop()
                prefix_sums.pop()
                for s in new_sums:
                    used_sums.discard(s)

                if found[0]:
                    return

        dfs(len(seq))
        if found[0]:
            return list(seq)
        return None
    
    # Strategy 1: Pure greedy smallest (fast version)
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Many randomized greedy attempts with varied parameters
    # Focus on getting diverse long sequences
    for attempt in range(100):
        if time.time() - start_time > time_limit - 20:
            break
        # Vary strategy: some pure random, some biased
        if attempt % 3 == 0:
            seq = greedy_randomized(seed_offset=attempt * 7 + 1)
        elif attempt % 3 == 1:
            # Try starting with a random small value
            start_val = random.randint(1, 10)
            seq = greedy_randomized(start_val=start_val, seed_offset=1000 + attempt * 13)
        else:
            # Try with different bias by shuffling candidate order
            seq = greedy_randomized(seed_offset=5000 + attempt * 23)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: CSP backtracking to extend beyond greedy
    # Use the best greedy solution as a starting point
    for target in range(best_len + 1, min(best_len + 6, 40)):
        if time.time() - start_time > time_limit - 10:
            break
        remaining_time = min(3.0, time_limit - 10 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        result = csp_backtrack_from(best_seq, target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 4: More randomized attempts with different biases
    for attempt in range(50):
        if time.time() - start_time > time_limit - 5:
            break
        # Vary the bias by using different seed offsets and start values
        start_val = random.randint(1, 20)
        seq = greedy_randomized(start_val=start_val, seed_offset=10000 + attempt * 17)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        # Fallback to a known valid sequence
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq

if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
