import time
import random
import sys
import heapq

def construct():
    start_time = time.time()
    time_limit = 75.0  # Safety margin for 90s kill

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
        # Check new_sums are all distinct
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
            if len(valid_candidates) <= 3:
                x, (new_prefix, new_sums) = valid_candidates[0]
            else:
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

    def count_available(prefix_sums, used_sums, last_val):
        """Count how many valid next values exist from this state."""
        count = 0
        for x in range(last_val + 1, 257):
            result = can_add_fast(prefix_sums, used_sums, x, last_val)
            if result is not None:
                count += 1
        return count

    def dijkstra_search(max_depth, max_time=10.0):
        """Dijkstra-like search: explore states in order of most available continuations.
        State: (last_val, prefix_sums_tuple, used_sums_frozen)
        Priority: -count_available (more available = higher priority = explored first)
        This is a best-first search to find the longest sequence."""
        deadline = start_time + max_time
        best_seq = []
        best_len = 0

        # State: (priority, last_val, prefix_sums_tuple, used_sums_frozenset, seq_tuple)
        # priority = negative available count (min-heap, so most available first)
        # We use a counter to break ties
        counter = 0
        pq = []
        initial_state = (0, (0,), frozenset(), ())
        heapq.heappush(pq, (0, counter, 0, (0,), frozenset(), ()))
        counter += 1

        # To avoid revisiting states, track visited (last_val, prefix_sums, used_sums)
        # But this could be huge. Instead, limit by depth and use a cap on queue size.
        max_queue_size = 50000

        while pq and time.time() < deadline:
            if len(pq) > max_queue_size:
                break

            neg_avail, _, last_val, prefix_sums, used_sums, seq_tuple = heapq.heappop(pq)

            depth = len(seq_tuple)

            # Update best
            if depth > best_len:
                best_len = depth
                best_seq = list(seq_tuple)

            if depth >= max_depth:
                continue

            # Check time
            if time.time() > deadline:
                break

            # Find all valid next values
            valid_nexts = []
            for x in range(last_val + 1, 257):
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
                seen = set()
                for s in new_sums:
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if valid:
                    valid_nexts.append((x, new_prefix, new_sums))

            if not valid_nexts:
                continue

            # For each valid next, compute the available count for the child state
            # and push to priority queue
            for x, new_prefix, new_sums in valid_nexts:
                new_prefix_sums = prefix_sums + (new_prefix,)
                new_used_sums = used_sums.union(new_sums)
                new_seq = seq_tuple + (x,)

                # Compute available count for child (expensive but gives good heuristic)
                # To save time, use a cheaper heuristic: 256 - x (upper bound on remaining values)
                # combined with a quick sample
                avail_estimate = 256 - x
                # Quick check: how many of the next 20 values are valid?
                quick_count = 0
                for x2 in range(x + 1, min(x + 21, 257)):
                    np2 = new_prefix + x2
                    v2 = True
                    for p in new_prefix_sums:
                        s2 = np2 - p
                        if s2 in new_used_sums:
                            v2 = False
                            break
                    if v2:
                        quick_count += 1

                # Priority: negative (quick_count * 10 + avail_estimate) to prefer states with more options
                priority = -(quick_count * 10 + avail_estimate)
                heapq.heappush(pq, (priority, counter, x, new_prefix_sums, new_used_sums, new_seq))
                counter += 1

            # Safety: if queue is getting too large, stop
            if len(pq) > max_queue_size:
                break

        return best_seq

    def csp_backtrack(target_len, max_time=5.0):
        """CSP-style backtracking with forward checking."""
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

                seen = set()
                for s in new_sums:
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)

                if not valid:
                    continue

                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(x, depth + 1)

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

    # Strategy 1: Pure greedy smallest
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Try different starting points with fast greedy
    for start in range(1, 30):
        if time.time() - start_time > time_limit - 20:
            break
        seq = greedy_smallest(start)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: Randomized greedy attempts
    for attempt in range(30):
        if time.time() - start_time > time_limit - 20:
            break
        seq = greedy_randomized(seed_offset=attempt * 7 + 1)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 4: Dijkstra-like best-first search
    # Try to find sequences longer than current best
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        dijkstra_time = min(remaining_time - 5, 30)
        dijkstra_result = dijkstra_search(best_len + 5, max_time=dijkstra_time)
        if dijkstra_result and len(dijkstra_result) > best_len:
            best_len = len(dijkstra_result)
            best_seq = list(dijkstra_result)

    # Strategy 5: CSP backtracking to extend
    remaining_time = time_limit - (time.time() - start_time)
    for target in range(best_len + 1, min(best_len + 6, 35)):
        if time.time() - start_time > time_limit - 5:
            break
        remaining = min(4.0, time_limit - 5 - (time.time() - start_time))
        if remaining < 1.0:
            break
        result = csp_backtrack(target, max_time=remaining)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 6: More randomized attempts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        for attempt in range(20):
            if time.time() - start_time > time_limit - 3:
                break
            start_val = random.randint(1, 20)
            seq = greedy_randomized(start_val=start_val, seed_offset=1000 + attempt * 13)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq
