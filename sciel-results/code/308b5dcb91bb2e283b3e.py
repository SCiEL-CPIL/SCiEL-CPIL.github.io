import time
import random
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_tails(values, x):
        """Compute all consecutive block sums ending at x."""
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        return tails

    def is_valid_extension(values, sums, x):
        """Check if appending x to values creates no duplicate block sums."""
        tails = compute_tails(values, x)
        return not sums.intersection(tails), tails

    # Phase 1: Greedy baseline (fast)
    def greedy_build():
        values = []
        sums = set()
        for x in range(1, 257):
            if values and x <= values[-1]:
                continue
            valid, tails = is_valid_extension(values, sums, x)
            if valid:
                values.append(x)
                sums.update(tails)
        return values, sums

    best_seq, best_sums = greedy_build()
    best_len = len(best_seq)

    # Phase 2: Iterative deepening DFS on the DAG of valid prefixes
    # Node = (last_value, frozenset_of_sums, length)
    # Edge = append a valid next value
    # We search for the longest path from the empty prefix

    def dfs_extend(values, sums, target_len, max_val, visited_states, deadline):
        """
        Try to extend values to reach target_len.
        Returns (success, new_values, new_sums) or (False, None, None).
        Uses memoization via visited_states to prune.
        """
        if time.time() > deadline:
            return False, None, None

        if len(values) == target_len:
            return True, list(values), set(sums)

        last_val = values[-1] if values else 0

        # Heuristic pruning: if we need k more elements, and the minimum possible
        # sum of k elements starting from last_val+1 exceeds what we can fit, prune
        k_needed = target_len - len(values)
        min_possible_sum = sum(range(last_val + 1, last_val + 1 + k_needed))
        # All block sums must be distinct, and there are O(n^2) of them
        # The maximum block sum is at most 256 * target_len
        # We need at least target_len*(target_len+1)/2 distinct sums
        # Each sum is at most 256*target_len
        # So we need target_len*(target_len+1)/2 <= 256*target_len
        # => (target_len+1)/2 <= 256 => target_len <= 511 (not binding for our range)

        # State key for memoization: (last_val, frozenset(sums))
        # This is expensive; use a lighter key
        state_key = (last_val, len(values), min(sums) if sums else 0, max(sums) if sums else 0)

        if state_key in visited_states:
            return False, None, None

        # Try candidates from last_val+1 to max_val
        # Heuristic: try values that create more "spread" in sums first
        # Actually, for finding ANY extension of target_len, we want to be systematic
        # Try smaller values first to leave room for more elements

        for x in range(last_val + 1, max_val + 1):
            if time.time() > deadline:
                return False, None, None

            valid, tails = is_valid_extension(values, sums, x)
            if not valid:
                continue

            new_values = values + [x]
            new_sums = sums | set(tails)

            # Check if we can still reach target_len with remaining budget
            # The remaining elements must be > x, so at least k_needed-1 more elements
            # Each must be at least x+1, x+2, ...
            # This is a loose bound

            success, res_values, res_sums = dfs_extend(
                new_values, new_sums, target_len, max_val, visited_states, deadline
            )
            if success:
                return True, res_values, res_sums

        visited_states.add(state_key)
        return False, None, None

    # Phase 3: Iterative deepening to find the maximum length
    # Start from best_len and try to extend by 1, 2, 3, ...
    max_possible = 16  # upper bound estimate; will be refined
    # Theoretical upper bound: n elements give n*(n+1)/2 block sums,
    # each at most 256*n, so n*(n+1)/2 <= 256*n => n+1 <= 512 => n <= 511
    # But practically much less due to distinctness constraints

    deadline = start_time + DEADLINE - 5

    # Try to extend beyond current best
    for target_len in range(best_len + 1, min(best_len + 12, 30)):
        if time_left() < 8:
            break

        visited_states = set()
        success, res_values, res_sums = dfs_extend(
            best_seq, best_sums, target_len, 256, visited_states, deadline
        )
        if success:
            best_seq, best_sums, best_len = res_values, res_sums, target_len
        else:
            # If we couldn't extend by 1, try a different starting point
            # Restart from empty with iterative deepening
            break

    # Phase 4: If DFS from greedy didn't help, try from scratch with iterative deepening
    if time_left() > 15:
        for target_len in range(max(best_len + 1, 10), min(best_len + 10, 25)):
            if time_left() < 8:
                break

            visited_states = set()
            deadline_inner = start_time + DEADLINE - 5
            success, res_values, res_sums = dfs_extend(
                [], set(), target_len, 256, visited_states, deadline_inner
            )
            if success:
                if target_len > best_len:
                    best_seq, best_sums, best_len = res_values, res_sums, target_len

    # Phase 5: Local improvement - try to insert elements into the best sequence
    if time_left() > 10:
        current = list(best_seq)
        current_sums = set(best_sums)

        improved = True
        while improved and time_left() > 5:
            improved = False
            for j in range(len(current) + 1):
                if time_left() < 5:
                    break
                lo_ins = current[j - 1] + 1 if j > 0 else 1
                hi_ins = current[j] - 1 if j < len(current) else 256

                for x in range(lo_ins, hi_ins + 1):
                    new_seq = current[:j] + [x] + current[j:]
                    # Compute all consecutive block sums
                    new_sums = set()
                    valid = True
                    for a in range(len(new_seq)):
                        s = 0
                        for b in range(a, len(new_seq)):
                            s += new_seq[b]
                            if s in new_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        if not valid:
                            break
                    if valid:
                        current = new_seq
                        current_sums = new_sums
                        improved = True
                        break
                if improved:
                    break

        if len(current) > best_len:
            best_seq, best_len = current, len(current)

    return best_seq
