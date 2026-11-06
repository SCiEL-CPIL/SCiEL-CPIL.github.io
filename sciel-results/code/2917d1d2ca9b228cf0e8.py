import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to S_set while keeping it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def greedy_build(S, order):
        """Build a sum-free set by processing elements in given order."""
        current = []
        current_set = set()
        for x in order:
            if can_add(x, current_set):
                current.append(x)
                current_set.add(x)
        return current

    def local_search(S, initial, time_limit):
        """
        Local search with add, 1-swap, and 2-swap moves.
        """
        current = list(initial)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        no_improve = 0
        max_no_improve = 80

        while time.time() < time_limit:
            improved = False

            # 1. Try to add any element
            candidates = [x for x in S if x not in current_set]
            random.shuffle(candidates)
            for add in candidates:
                if can_add(add, current_set):
                    current.append(add)
                    current_set.add(add)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    no_improve = 0
                    break

            if improved:
                continue

            # 2. Try 1-swap: remove one, add one
            rem_candidates = list(current)
            random.shuffle(rem_candidates)

            for rem in rem_candidates:
                if time.time() > time_limit:
                    break
                trial_set = current_set.copy()
                trial_set.discard(rem)

                add_candidates = [x for x in S if x not in trial_set]
                random.shuffle(add_candidates)

                for add in add_candidates:
                    if can_add(add, trial_set):
                        current = [x for x in current if x != rem]
                        current.append(add)
                        current_set = set(current)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break

            if not improved:
                # Try 2-swap: remove two, add two
                if len(current) >= 2:
                    rem_pairs = []
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            rem_pairs.append((current[i], current[j]))
                    random.shuffle(rem_pairs)
                    # Limit pairs to try
                    for rem1, rem2 in rem_pairs[:30]:
                        if time.time() > time_limit:
                            break
                        trial_set = current_set.copy()
                        trial_set.discard(rem1)
                        trial_set.discard(rem2)

                        add_candidates = [x for x in S if x not in trial_set]
                        random.shuffle(add_candidates)

                        added = []
                        for add in add_candidates:
                            if len(added) == 2:
                                break
                            if can_add(add, trial_set | set(added)):
                                added.append(add)

                        if len(added) == 2:
                            current = [x for x in current if x != rem1 and x != rem2]
                            current.extend(added)
                            current_set = set(current)
                            improved = True
                            no_improve = 0
                            break

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return best

    def conflict_degree(x, S_set):
        """Count how many elements in S conflict with x."""
        count = 0
        for y in S_set:
            if x == y:
                if 2 * x in S_set:
                    count += 1
            else:
                if x + y in S_set or x - y in S_set or y - x in S_set:
                    count += 1
        return count

    def solve_instance(S, time_limit):
        n = len(S)
        S_set_all = set(S)
        best = []
        best_size = 0

        # Structural priors
        # 1. Odds
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > best_size:
            best = list(odds)
            best_size = len(odds)

        # 2. Positives upper half
        pos = sorted([x for x in S if x > 0])
        if pos:
            half = len(pos) // 2
            pos_large = pos[half:]
            if len(pos_large) > best_size:
                best = list(pos_large)
                best_size = len(pos_large)

        # 3. Negatives lower half
        neg = sorted([x for x in S if x < 0])
        if neg:
            half = len(neg) // 2
            neg_small = neg[:half]
            if len(neg_small) > best_size:
                best = list(neg_small)
                best_size = len(neg_small)

        # 4. Greedy from multiple diverse orderings
        greedy_time = min(5.0, (time_limit - time.time()) * 0.15)
        greedy_deadline = time.time() + greedy_time

        while time.time() < greedy_deadline:
            # Different orderings
            strategies = [
                list(S),  # sorted
                list(reversed(S)),
                sorted(S, key=lambda x: abs(x)),
                sorted(S, key=lambda x: -abs(x)),
                sorted(S, key=lambda x: x),
                sorted(S, key=lambda x: -x),
            ]
            for strat in strategies:
                if time.time() > greedy_deadline:
                    break
                order = list(strat)
                random.shuffle(order)
                current = greedy_build(S, order)
                if len(current) > best_size:
                    best = current
                    best_size = len(current)

        # 5. Conflict-degree-based pruning and greedy on residual
        # Remove high-conflict elements, then greedy on the rest
        if time.time() < time_limit - 10:
            # Compute conflict degrees
            conflict_degs = {x: conflict_degree(x, S_set_all) for x in S}
            # Sort by conflict degree (ascending - low conflict first)
            low_conflict_order = sorted(S, key=lambda x: conflict_degs[x])
            current = greedy_build(S, low_conflict_order)
            if len(current) > best_size:
                best = current
                best_size = len(current)

            # High conflict first (removing them may unlock more)
            high_conflict_order = sorted(S, key=lambda x: -conflict_degs[x])
            current = greedy_build(S, high_conflict_order)
            if len(current) > best_size:
                best = current
                best_size = len(current)

        # 6. Local search from best and diverse starts
        remaining = time_limit - time.time()
        if remaining > 2.0:
            # Local search from best
            improved = local_search(S, best, time.time() + remaining * 0.4)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

            # Local search from odds
            time_left = time_limit - time.time()
            if time_left > 3.0 and len(odds) > 0:
                improved = local_search(S, odds, time.time() + time_left * 0.3)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)

            # Local search from pos_large
            time_left = time_limit - time.time()
            if time_left > 3.0 and pos:
                half = len(pos) // 2
                pos_large = pos[half:]
                improved = local_search(S, pos_large, time.time() + time_left * 0.3)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)

            # Additional local search from best with more time
            time_left = time_limit - time.time()
            if time_left > 2.0:
                improved = local_search(S, best, time.time() + time_left * 0.8)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)

        return best

    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]

    # Time allocation: proportional to size, with minimum
    fractions = [0.15, 0.30, 0.55]

    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed
        instance_time = start_time + remaining_total * fractions[i]
        # Ensure minimum
        if instance_time < time.time() + 3.0:
            instance_time = time.time() + 3.0
        # Ensure not past deadline
        if instance_time > start_time + DEADLINE - 2.0:
            instance_time = start_time + DEADLINE - 2.0

        result = solve_instance(S, instance_time)
        results.append(result)

        # Check time
        if time.time() > start_time + DEADLINE - 5.0:
            break

    # Fill missing with quick greedy (odds)
    while len(results) < len(instances):
        size, seed = instances[len(results)]
        S = instance(size, seed)
        result = [x for x in S if x % 2 != 0]
        results.append(result)

    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            S = instance(instances[i][0], instances[i][1])
            results[i] = [x for x in S if x % 2 != 0]

    return results
