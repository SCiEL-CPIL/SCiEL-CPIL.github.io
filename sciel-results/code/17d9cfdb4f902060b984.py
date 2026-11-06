import time
import random
import sys

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
        seen = set()
        for s in new_sums:
            if s in seen:
                return None
            seen.add(s)
        return (new_prefix, new_sums)
    
    def greedy_smallest(start_val=1):
        """Greedy: always pick the smallest valid next element, using fast can_add"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        for _ in range(256):
            found = False
            for x in range(start_val, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    new_prefix, new_sums = result
                    seq.append(x)
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    last_val = x
                    start_val = x + 1
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
        for _ in range(256):
            valid_candidates = []
            for x in range(start_val, 257):
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
            start_val = x + 1
        return seq
    
    def csp_backtrack(target_len, max_time=5.0):
        """CSP-style backtracking with forward checking: maintain forbidden sums explicitly."""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        deadline = start_time + max_time
        found = [False]

        def dfs(last_val, depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return

            # Forward checking: determine which candidate values are immediately invalid
            # A candidate x is invalid if:
            #   1. x <= last_val or x > 256
            #   2. For some existing prefix p, (prefix_sums[-1] + x) - p is in used_sums
            # We can prune: for each existing prefix p, x must not satisfy
            #   prefix_sums[-1] + x - p in used_sums
            # i.e., x not in {s + p - prefix_sums[-1] for s in used_sums for p in prefix_sums}
            # But that's expensive. Instead, just iterate and check.

            # To speed up, precompute forbidden x values from the last prefix
            # new_prefix = prefix_sums[-1] + x
            # For each existing prefix p: new_prefix - p must not be in used_sums
            # => x must not be in {s + p - prefix_sums[-1] for s in used_sums for p in prefix_sums}
            # This is O(|used_sums| * |prefix_sums|) which can be large.
            # Instead, iterate candidates and check incrementally.

            for x in range(last_val + 1, 257):
                if found[0] or time.time() > deadline:
                    return

                new_prefix = prefix_sums[-1] + x
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

                # Check new_sums for internal collisions
                seen = set()
                for s in new_sums:
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)

                if not valid:
                    continue

                # Commit
                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(x, depth + 1)

                # Backtrack
                seq.pop()
                prefix_sums.pop()
                for s in new_sums:
                    used_sums.discard(s)

                if found[0]:
                    return

        dfs(0, 0)
        if found[0]:
            return list(seq)
        return None
    
    # Strategy 1: Pure greedy smallest (fast version)
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Try different starting points with fast greedy
    for start in range(1, 30):
        if time.time() - start_time > time_limit - 10:
            break
        seq = greedy_smallest(start)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: Many randomized greedy attempts (fast)
    for attempt in range(50):
        if time.time() - start_time > time_limit - 15:
            break
        seq = greedy_randomized(seed_offset=attempt * 7 + 1)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 4: CSP backtracking to extend beyond greedy
    # Try to find sequences of length best_len + 1, +2, etc.
    for target in range(best_len + 1, min(best_len + 8, 35)):
        if time.time() - start_time > time_limit - 8:
            break
        remaining_time = min(5.0, time_limit - 8 - (time.time() - start_time))
        if remaining_time < 1.0:
            break
        result = csp_backtrack(target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 5: More randomized attempts with different biases
    for attempt in range(30):
        if time.time() - start_time > time_limit - 5:
            break
        # Vary the bias by using different seed offsets and start values
        start_val = random.randint(1, 20)
        seq = greedy_randomized(start_val=start_val, seed_offset=1000 + attempt * 13)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        # Fallback to a known valid sequence
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq
