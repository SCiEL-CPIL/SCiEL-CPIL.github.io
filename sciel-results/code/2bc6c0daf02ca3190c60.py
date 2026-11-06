"""Rewritten program: local search for large sum-free subsets."""
import time
import random

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def conflict_count(subset):
        s = set(subset)
        cnt = 0
        for a in subset:
            for b in subset:
                if a + b in s:
                    cnt += 1
        return cnt

    def solve_instance(values, time_budget):
        n = len(values)
        idx_to_val = list(range(n))
        val_to_idx = {v: i for i, v in enumerate(values)}

        # Greedy initial: sort by some heuristic, add if no conflict
        # Try multiple orderings
        best_subset = set()
        best_size = 0

        # Ordering 1: by absolute value (small first)
        orderings = [
            sorted(range(n), key=lambda i: abs(values[i])),
            sorted(range(n), key=lambda i: abs(values[i]), reverse=True),
            sorted(range(n), key=lambda i: values[i]),
            sorted(range(n), key=lambda i: values[i], reverse=True),
            list(range(n)),
        ]

        for order in orderings:
            current = set()
            for i in order:
                v = values[i]
                # Check if adding v creates conflict: v = a+b, or v+a in set, or v+b in set
                # Actually need: no a,b in current such that a+b == v, and no a in current with v+a in current
                ok = True
                for a in current:
                    if (v - a) in current:
                        ok = False
                        break
                    if (v + a) in current:
                        ok = False
                        break
                if ok:
                    current.add(v)
            if len(current) > best_size:
                best_size = len(current)
                best_subset = set(current)

        # Local search: try to add elements not in subset, remove elements that cause conflicts
        current = set(best_subset)

        # Build conflict map for efficiency
        def get_conflicts(subset):
            """Return set of elements in subset that are part of a conflict."""
            s = set(subset)
            conflicts = set()
            for a in subset:
                for b in subset:
                    c = a + b
                    if c in s:
                        conflicts.add(a)
                        conflicts.add(b)
                        conflicts.add(c)
            return conflicts

        # Iterative improvement
        improved = True
        while improved and time.time() < time_budget:
            improved = False

            # Try adding each element not in current
            candidates = [v for v in values if v not in current]
            random.shuffle(candidates)
            for v in candidates:
                trial = current | {v}
                if is_sum_free(trial):
                    current = trial
                    improved = True
                    break

            # If we can't add, try removing one element to allow adding another
            if not improved:
                confs = get_conflicts(current)
                if confs:
                    # Try removing each conflicting element
                    rem_list = list(confs)
                    random.shuffle(rem_list)
                    for rem in rem_list:
                        trial = current - {rem}
                        # Now try to add some candidate
                        candidates2 = [v for v in values if v not in trial]
                        random.shuffle(candidates2)
                        added = False
                        for v in candidates2:
                            trial2 = trial | {v}
                            if is_sum_free(trial2):
                                current = trial2
                                improved = True
                                added = True
                                break
                        if added:
                            break
                # If still no improvement, try random restart of local search
                if not improved:
                    # Simulated annealing style: accept worse moves occasionally
                    if random.random() < 0.1:
                        # Remove a random element
                        if current:
                            rem = random.choice(list(current))
                            current = current - {rem}
                            improved = True

        # Final cleanup: ensure sum-free
        while not is_sum_free(current):
            confs = get_conflicts(current)
            if not confs:
                break
            rem = random.choice(list(confs))
            current = current - {rem}

        return sorted(current)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        values = instance(size, seed)
        # Allocate time proportional to difficulty
        remaining = deadline - time.time()
        if remaining < 1:
            # Not enough time, use greedy only
            subset = [x for x in values if x % 2]
            results.append(subset)
            continue

        # Give more time to larger instances
        if size == 240:
            budget = time.time() + remaining * 0.5
        elif size == 120:
            budget = time.time() + remaining * 0.3
        else:
            budget = time.time() + remaining * 0.2

        subset = solve_instance(values, budget)
        results.append(subset)

    return results
