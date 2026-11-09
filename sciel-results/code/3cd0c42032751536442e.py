import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    def compute_reachable(S):
        """Compute all subset sums <= 200 for set S using bitset. Returns None if 200 is reachable."""
        reachable = 1  # bit i set means sum i is reachable
        for x in S:
            new_sums = (reachable << x) & ((1 << 201) - 1)
            if new_sums & (1 << 200):
                return None
            reachable |= new_sums
        return reachable

    def is_valid(S):
        return compute_reachable(S) is not None

    def greedy(order):
        selected = []
        reachable = 1
        for x in order:
            new = (reachable << x) & ((1 << 201) - 1)
            if not (new & (1 << 200)):
                selected.append(x)
                reachable |= new
        return selected

    best = []
    for order in [list(range(100, 0, -1)), list(range(1, 101)), 
                  list(range(50, 0, -1)) + list(range(100, 50, -1)),
                  list(range(100, 50, -1)) + list(range(50, 0, -1))]:
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand

    best_set = set(best)
    best_size = len(best)

    def local_search_add(S):
        """Try to add any excluded element that keeps validity."""
        changed = True
        while changed:
            changed = False
            reach_S = compute_reachable(S)
            if reach_S is None:
                break
            for x in range(1, 101):
                if x in S:
                    continue
                new_sums = (reach_S << x) & ((1 << 201) - 1)
                if not (new_sums & (1 << 200)):
                    S = S | {x}
                    changed = True
                    break
        return S

    def try_swaps(S, time_limit):
        """Try removing one or two elements and adding two or three excluded ones."""
        excluded = [x for x in range(1, 101) if x not in S]
        if not excluded:
            return S

        reach_S = compute_reachable(S)
        if reach_S is None:
            return S

        best_trial = S
        best_trial_size = len(S)

        # 1-for-2 and 1-for-3 swaps
        for rem in list(S):
            if time.time() > time_limit:
                break
            reach_no_rem = compute_reachable(S - {rem})
            if reach_no_rem is None:
                continue

            trial = S - {rem}
            trial_reach = reach_no_rem
            added = 0
            for add in excluded:
                if time.time() > time_limit:
                    break
                if add in trial:
                    continue
                new_sums = (trial_reach << add) & ((1 << 201) - 1)
                if not (new_sums & (1 << 200)):
                    trial = trial | {add}
                    trial_reach |= new_sums
                    added += 1

            if len(trial) > best_trial_size:
                best_trial = trial
                best_trial_size = len(trial)

        # 2-for-3 swaps
        if time.time() < time_limit:
            s_list = list(S)
            for i in range(len(s_list)):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, len(s_list)):
                    if time.time() > time_limit:
                        break
                    rem_pair = {s_list[i], s_list[j]}
                    reach_no_rem = compute_reachable(S - rem_pair)
                    if reach_no_rem is None:
                        continue

                    trial = S - rem_pair
                    trial_reach = reach_no_rem
                    added = 0
                    for add in excluded:
                        if time.time() > time_limit:
                            break
                        if add in trial:
                            continue
                        new_sums = (trial_reach << add) & ((1 << 201) - 1)
                        if not (new_sums & (1 << 200)):
                            trial = trial | {add}
                            trial_reach |= new_sums
                            added += 1

                    if len(trial) > best_trial_size:
                        best_trial = trial
                        best_trial_size = len(trial)

        return best_trial

    def randomized_search(S, iterations, time_limit):
        rng = random.Random(42)
        current = set(S)
        current_size = len(current)

        for i in range(iterations):
            if time.time() > time_limit:
                break

            rem_candidates = list(current)
            if not rem_candidates:
                break
            rem = rng.choice(rem_candidates)

            excluded = [x for x in range(1, 101) if x not in current]
            if not excluded:
                break

            for num_add in [2, 3]:
                if len(excluded) < num_add:
                    continue
                adds = rng.sample(excluded, num_add)
                trial = current - {rem} | set(adds)
                if len(trial) > current_size:
                    if is_valid(trial):
                        current = trial
                        current_size = len(current)
                        break

        return current

    # Apply local search
    best_set = local_search_add(best_set)
    best_size = len(best_set)

    # Try swaps if time allows
    if time.time() < deadline - 25.0:
        swapped = try_swaps(best_set, deadline - 15.0)
        if len(swapped) > best_size:
            best_set = swapped
            best_size = len(best_set)
            best_set = local_search_add(best_set)
            best_size = len(best_set)

    # Randomized search if time allows
    if time.time() < deadline - 15.0:
        rand_result = randomized_search(best_set, 5000, deadline - 10.0)
        if len(rand_result) > best_size:
            best_set = rand_result
            best_size = len(best_set)
            best_set = local_search_add(best_set)
            best_size = len(best_set)

    # Final verification
    if not best_set:
        best_set = set(range(100, 50, -1))
        best_size = len(best_set)

    if not is_valid(best_set):
        best_set = set(range(67, 101))
        best_size = len(best_set)

    return list(best_set)
