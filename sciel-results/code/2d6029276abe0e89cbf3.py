import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    def compute_sums(elements):
        """Compute all non-empty subset sums. Returns set or None if duplicates."""
        sums = set()
        for x in elements:
            if x in sums:
                return None
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return None
                new_sums.add(ns)
            if x in new_sums:
                return None
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums

    def check_divisibility(sums):
        """Check no two distinct sums divide each other. Returns True if valid."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i + 1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True

    def is_valid(elements):
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        if any(x < 1 or x > 256 for x in elements):
            return False
        sums = compute_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)

    def can_add(x, current_elements, current_sums):
        """
        Check if x can be added to current_elements.
        Returns (new_sums, True) if valid, (None, False) otherwise.
        """
        if x in current_elements:
            return None, False
        if x in current_sums:
            return None, False

        new_sums = {x}
        for s in current_sums:
            ns = s + x
            if ns in current_sums or ns in new_sums:
                return None, False
            new_sums.add(ns)

        all_sums = current_sums.union(new_sums)

        # Check divisibility among new_sums
        ns_list = sorted(new_sums)
        for i in range(len(ns_list)):
            si = ns_list[i]
            for j in range(i + 1, len(ns_list)):
                sj = ns_list[j]
                if sj % si == 0:
                    return None, False

        # Check divisibility between new_sums and current_sums
        for sn in new_sums:
            for sc in current_sums:
                if sn == sc:
                    return None, False
                if sn > sc:
                    if sn % sc == 0:
                        return None, False
                else:
                    if sc % sn == 0:
                        return None, False

        return all_sums, True

    def greedy_init():
        """Greedy initialization with randomized large-first ordering."""
        current_elements = []
        current_sums = set()
        candidates = list(range(256, 1, -1))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                break
            new_sums, valid = can_add(x, current_elements, current_sums)
            if valid:
                current_elements.append(x)
                current_sums = new_sums

        return current_elements, current_sums

    def branch_and_bound_search(initial_set, time_limit, max_size=15):
        """
        Branch-and-bound to find a maximum valid set.
        Uses the initial_set as a starting point and tries to extend it.
        """
        best_found = list(initial_set)
        best_sums = compute_sums(initial_set) if initial_set else set()

        def try_extend(current_elements, current_sums, remaining_candidates):
            nonlocal best_found, best_sums

            if time.time() > time_limit:
                return

            if len(current_elements) > len(best_found):
                best_found = list(current_elements)
                best_sums = current_sums

            if len(current_elements) >= max_size:
                return

            # Sort remaining candidates by some heuristic (larger first often works better)
            for i, x in enumerate(remaining_candidates):
                if time.time() > time_limit:
                    return

                new_sums, valid = can_add(x, current_elements, current_sums)
                if valid:
                    # Recurse with remaining candidates after x
                    remaining_after = remaining_candidates[i + 1:]
                    try_extend(current_elements + [x], new_sums, remaining_after)

        # Start with the initial set and try to extend
        if initial_set:
            current_elements = list(initial_set)
            current_sums = compute_sums(initial_set)
            if current_sums is not None:
                # Candidates are all numbers not in current_elements
                candidates = [x for x in range(256, 1, -1) if x not in current_elements]
                random.shuffle(candidates)
                try_extend(current_elements, current_sums, candidates)

        return best_found

    def simulated_annealing(initial_set, time_limit):
        """
        Simulated annealing to improve the set.
        Moves: add, remove, swap.
        """
        current = list(initial_set)
        current_sums = compute_sums(current) if current else set()
        if current_sums is None:
            current_sums = set()

        best = list(current)
        best_sums = current_sums

        def score_of(elements):
            """Score: prefer larger sets, then check validity."""
            if not elements:
                return 0
            sums = compute_sums(elements)
            if sums is None:
                return -1000
            if not check_divisibility(sums):
                return -500
            return len(elements)

        current_score = score_of(current)
        best_score = current_score

        T = 10.0
        cooling = 0.995
        min_T = 0.01

        while time.time() < time_limit and T > min_T:
            # Generate a move
            move_type = random.random()
            new_set = list(current)

            if move_type < 0.4:
                # Add a random element
                available = [x for x in range(1, 257) if x not in new_set]
                if available:
                    x = random.choice(available)
                    new_set.append(x)
            elif move_type < 0.7:
                # Remove a random element
                if new_set:
                    idx = random.randint(0, len(new_set) - 1)
                    new_set.pop(idx)
            else:
                # Swap: remove one, add another
                if new_set:
                    idx = random.randint(0, len(new_set) - 1)
                    removed = new_set.pop(idx)
                    available = [x for x in range(1, 257) if x not in new_set]
                    if available:
                        x = random.choice(available)
                        new_set.append(x)

            new_score = score_of(new_set)

            if new_score > current_score:
                current = new_set
                current_score = new_score
                if new_score > best_score:
                    best = list(current)
                    best_score = new_score
                    best_sums = compute_sums(current) if current else set()
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current = new_set
                        current_score = new_score
                # else: delta >= 0 but not > current_score? That can't happen.

            T *= cooling

        return best

    # Main search logic
    best_set = []

    # Phase 1: Greedy initialization
    if time.time() < deadline - 10:
        greedy_set, _ = greedy_init()
        best_set = greedy_set

    # Phase 2: Simulated annealing
    if time.time() < deadline - 10:
        sa_time = min(30.0, deadline - time.time() - 5)
        if sa_time > 5:
            improved = simulated_annealing(best_set, time.time() + sa_time)
            if len(improved) > len(best_set):
                best_set = improved

    # Phase 3: Branch-and-bound from the best found
    if time.time() < deadline - 5:
        bnb_time = min(20.0, deadline - time.time() - 3)
        if bnb_time > 3:
            bnb_result = branch_and_bound_search(best_set, time.time() + bnb_time)
            if len(bnb_result) > len(best_set):
                best_set = bnb_result

    # Phase 4: Try more random restarts with greedy + local search
    if time.time() < deadline - 3:
        restart_time = deadline - time.time() - 2
        restarts = int(restart_time / 2)
        for _ in range(restarts):
            if time.time() > deadline - 2:
                break
            g_set, _ = greedy_init()
            if len(g_set) > len(best_set):
                best_set = g_set
            # Quick local search: try to add more
            cur = list(g_set)
            cur_sums = compute_sums(cur) if cur else set()
            if cur_sums is not None:
                for trial in range(50):
                    if time.time() > deadline - 2:
                        break
                    candidates = [x for x in range(256, 1, -1) if x not in cur]
                    random.shuffle(candidates)
                    added = False
                    for x in candidates:
                        new_sums, valid = can_add(x, cur, cur_sums)
                        if valid:
                            cur.append(x)
                            cur_sums = new_sums
                            added = True
                            break
                    if not added:
                        break
                if len(cur) > len(best_set):
                    best_set = cur

    # Final validation and cleanup
    if best_set:
        if not is_valid(best_set):
            best_set = [100, 101]

    # Ensure types
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique

    return best_set
