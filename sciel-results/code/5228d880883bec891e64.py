import time
import sys

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

# Max possible sum is 284 + 284 = 568
SUMS_SIZE = 570
sum_bits = [1 << i for i in range(SUMS_SIZE)]


def construct():
    global best_set, best_size, start_time

    start_time = time.time()

    # Initial prescribed set
    initial_set = [1, 2, 5, 11]
    initial_mask = 0

    # Mark initial sums
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_mask |= sum_bits[s]

    best_set = initial_set[:]
    best_size = len(initial_set)

    # Candidates are 12 to 284
    # Filter out candidates that conflict with the initial set immediately
    candidates = []
    for x in range(12, 285):
        valid = True
        # Check 2*x
        if initial_mask & sum_bits[2 * x]:
            valid = False
        # Check x + y for y in initial_set
        if valid:
            for y in initial_set:
                s = x + y
                if initial_mask & sum_bits[s]:
                    valid = False
                    break
        if valid:
            candidates.append(x)

    # Strategy 1: Greedy warm-start to find a good initial solution
    def greedy_solve(order, time_budget=2.0):
        nonlocal best_set, best_size
        if time.time() - start_time > time_limit - 5:
            return

        s = initial_set[:]
        mask = initial_mask
        t0 = time.time()

        for x in order:
            if time.time() - t0 > time_budget:
                break
            if time.time() - start_time > time_limit - 5:
                break

            # Check if x can be added
            if mask & sum_bits[2 * x]:
                continue

            can_add = True
            new_mask = mask
            for y in s:
                s_val = x + y
                if new_mask & sum_bits[s_val]:
                    can_add = False
                    break
                new_mask |= sum_bits[s_val]

            if not can_add:
                continue

            new_mask |= sum_bits[2 * x]
            s.append(x)
            mask = new_mask

        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # Try greedy with different orderings
    # Descending (spread out)
    greedy_solve(candidates[::-1], time_budget=1.0)
    # Ascending
    greedy_solve(candidates[:], time_budget=1.0)

    # Strategy 2: Compute conflict degree for each candidate
    # A candidate's "degree" is how many other candidates it conflicts with
    # Lower degree = more flexible = should be tried earlier in DFS

    def compute_conflict_degree(cand_list, base_mask):
        """Compute for each candidate how many other candidates it conflicts with."""
        n = len(cand_list)
        degrees = [0] * n

        for i in range(n):
            xi = cand_list[i]
            # Build mask of sums involving xi
            mask_i = base_mask | sum_bits[2 * xi]
            for y in initial_set:
                mask_i |= sum_bits[xi + y]

            for j in range(i + 1, n):
                xj = cand_list[j]
                # Check if xi and xj conflict
                # They conflict if any sum from xi (with initial_set) equals any sum from xj (with initial_set),
                # or if xi + xj is already in mask_i or would be in mask_j, etc.
                # Simpler: check if adding xj to {initial_set + xi} is valid
                test_mask = mask_i
                can_add_j = True
                for y in initial_set:
                    s_val = xj + y
                    if test_mask & sum_bits[s_val]:
                        can_add_j = False
                        break
                    test_mask |= sum_bits[s_val]
                if can_add_j:
                    if test_mask & sum_bits[2 * xj]:
                        can_add_j = False
                    else:
                        test_mask |= sum_bits[2 * xj]

                if not can_add_j:
                    degrees[i] += 1
                    degrees[j] += 1

        return degrees

    # Compute degrees and sort by ascending degree (most flexible first)
    # But also consider value - larger values might be better for spreading
    degrees = compute_conflict_degree(candidates, initial_mask)

    # Sort by (degree, -value) - lower degree first, then higher value
    indexed_candidates = list(enumerate(candidates))
    indexed_candidates.sort(key=lambda idx_deg: (degrees[idx_deg[0]], -candidates[idx_deg[0]]))
    sorted_candidates = [c for _, c in indexed_candidates]

    # Strategy 3: DFS with bitmask, trying to reach target sizes
    # Use the sorted candidates for better pruning

    def dfs(current_set, mask, start_idx, target_size, deadline):
        global best_set, best_size

        if time.time() > deadline:
            return False

        current_size = len(current_set)

        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]

        if current_size == target_size:
            return True

        # Pruning: If we can't reach target_size with remaining candidates
        if current_size + (len(sorted_candidates) - start_idx) < target_size:
            return False

        for i in range(start_idx, len(sorted_candidates)):
            x = sorted_candidates[i]

            # Check if x can be added
            if mask & sum_bits[2 * x]:
                continue

            can_add = True
            new_mask = mask
            for y in current_set:
                s_val = x + y
                if new_mask & sum_bits[s_val]:
                    can_add = False
                    break
                new_mask |= sum_bits[s_val]

            if not can_add:
                continue

            new_mask |= sum_bits[2 * x]

            current_set.append(x)
            if dfs(current_set, new_mask, i + 1, target_size, deadline):
                return True
            current_set.pop()

        return False

    # Try sizes from 20 down, with time allocation
    # Give more time to higher targets since they're more valuable
    target_time_alloc = {20: 30.0, 19: 20.0, 18: 15.0, 17: 10.0, 16: 8.0, 15: 6.0}

    for target in range(20, best_size, -1):
        if time.time() - start_time > time_limit - 5:
            break

        alloc = target_time_alloc.get(target, 5.0)
        deadline = time.time() + alloc

        # Reset state
        init_set_copy = initial_set[:]

        if dfs(init_set_copy, initial_mask, 0, target, deadline):
            # Found a solution of this size, try to go higher
            continue

    # Strategy 4: Local search polish - try to improve current best
    def local_search_polish(s, time_budget=10.0):
        nonlocal best_set, best_size
        if time.time() - start_time > time_limit - 5:
            return

        s = s[:]
        sums_set = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums_set.add(s[i] + s[j])

        t0 = time.time()
        all_vals = set(range(1, 285))
        prescribed_set = {1, 2, 5, 11}

        while time.time() - t0 < time_budget and time.time() - start_time < time_limit - 5:
            improved = False

            # Try adding a random element
            for _ in range(200):
                if time.time() - t0 > time_budget:
                    break
                if time.time() - start_time > time_limit - 5:
                    return

                # Pick a random candidate not in s
                available = [x for x in range(1, 285) if x not in s]
                if not available:
                    break
                x = available[ord(chr(97 + (int(time.time() * 1000) % 26))) % len(available)]

                # Check if x can be added
                if 2 * x in sums_set:
                    continue
                can_add = True
                for y in s:
                    if x + y in sums_set:
                        can_add = False
                        break
                if not can_add:
                    continue

                s.append(x)
                sums_set.add(2 * x)
                for y in s[:-1]:
                    sums_set.add(x + y)

                if len(s) > best_size:
                    best_size = len(s)
                    best_set = s[:]
                improved = True
                break

            if not improved:
                # Try swap: remove one non-prescribed, add one new
                removable = [x for x in s if x not in prescribed_set]
                if not removable:
                    break

                x_remove = removable[int(time.time() * 1000) % len(removable)]
                s.remove(x_remove)

                # Recompute sums
                sums_set = set()
                for i in range(len(s)):
                    for j in range(i, len(s)):
                        sums_set.add(s[i] + s[j])

                # Try to add a better element
                for _ in range(100):
                    if time.time() - t0 > time_budget:
                        break
                    if time.time() - start_time > time_limit - 5:
                        return

                    available = [x for x in range(1, 285) if x not in s]
                    if not available:
                        break
                    x_add = available[int(time.time() * 1000) % len(available)]

                    if 2 * x_add in sums_set:
                        continue
                    can_add = True
                    for y in s:
                        if x_add + y in sums_set:
                            can_add = False
                            break
                    if not can_add:
                        continue

                    s.append(x_add)
                    sums_set.add(2 * x_add)
                    for y in s[:-1]:
                        sums_set.add(x_add + y)

                    if len(s) > best_size:
                        best_size = len(s)
                        best_set = s[:]
                    break

    # Polish the best solution found
    if time.time() - start_time < time_limit - 15 and best_size > 4:
        local_search_polish(best_set, time_budget=min(10.0, time_limit - (time.time() - start_time) - 10))

    # Verify and return
    result = sorted(best_set)
    # Final validation
    sums_final = set()
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s_val = result[i] + result[j]
            if s_val in sums_final:
                valid = False
                break
            sums_final.add(s_val)
        if not valid:
            break

    if not valid:
        return initial_set[:]

    return result


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
