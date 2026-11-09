import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 82.0  # Leave 8s margin before 90s kill

    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N
    # For each element x, build list of "witness triples" - 3-element tuples
    # such that (x,) + tuple (in some order) forms a 4-AP.
    #
    # For a 4-AP (a, b, c, d) with step s:
    #   If x = a: witnesses (b, c, d)
    #   If x = b: witnesses (a, c, d)
    #   If x = c: witnesses (a, b, d)
    #   If x = d: witnesses (a, b, c)

    # Build conflict lists
    conflict = [[] for _ in range(N + 1)]  # 1-indexed
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflict[a].append((b, c, e))
            conflict[b].append((a, c, e))
            conflict[c].append((a, b, e))
            conflict[e].append((a, b, c))

    # Precompute numpy arrays for fast vectorized checking
    witness_arrays = [None] * (N + 1)
    for x in range(1, N + 1):
        if conflict[x]:
            witness_arrays[x] = np.array(conflict[x], dtype=np.int32)
        else:
            witness_arrays[x] = np.empty((0, 3), dtype=np.int32)

    def can_add(x, S_bool):
        """Check if x can be added to set represented by S_bool (numpy bool array, 1-indexed)."""
        wa = witness_arrays[x]
        if len(wa) == 0:
            return True
        # S_bool[wa] gives shape (k, 3) boolean array
        # Check if any row is all True (meaning all 3 witnesses are in S)
        return not np.any(np.all(S_bool[wa], axis=1))

    def greedy(order, S_bool):
        """Greedy: iterate through order, add x if possible."""
        added = 0
        for x in order:
            if not S_bool[x]:
                if can_add(x, S_bool):
                    S_bool[x] = True
                    added += 1
        return S_bool, added

    def count(S_bool):
        return int(S_bool[1:].sum())

    best_S = None
    best_count = 0

    def try_solution(S_bool):
        nonlocal best_S, best_count
        c = count(S_bool)
        if c > best_count:
            best_count = c
            best_S = S_bool.copy()

    # Strategy 1: Greedy with natural order
    S_bool = np.zeros(N + 1, dtype=bool)
    S_bool, _ = greedy(range(1, N + 1), S_bool)
    try_solution(S_bool)

    # Strategy 2: Greedy with reverse order
    S_bool = np.zeros(N + 1, dtype=bool)
    S_bool, _ = greedy(range(N, 0, -1), S_bool)
    try_solution(S_bool)

    # Strategy 3: Greedy with various random orders
    for seed in range(50):
        if time.time() > deadline:
            break
        S_bool = np.zeros(N + 1, dtype=bool)
        order = list(range(1, N + 1))
        random.Random(seed).shuffle(order)
        S_bool, _ = greedy(order, S_bool)
        try_solution(S_bool)

    # Strategy 4: Greedy with weighted order (prefer elements with fewer conflicts)
    if time.time() < deadline:
        # Count conflicts for each element
        conflict_counts = [len(conflict[x]) for x in range(N + 1)]
        # Sort by conflict count (ascending) - elements with fewer constraints first
        order = sorted(range(1, N + 1), key=lambda x: conflict_counts[x])
        S_bool = np.zeros(N + 1, dtype=bool)
        S_bool, _ = greedy(order, S_bool)
        try_solution(S_bool)

    # Strategy 5: Local search / hill climbing from best solution
    if best_S is not None:
        current = best_S.copy()
        current_count = best_count
        rng = random.Random(123)

        for iteration in range(200):
            if time.time() > deadline:
                break

            improvement_found = False

            # Try to add any missing element
            missing = [x for x in range(1, N + 1) if not current[x]]
            rng.shuffle(missing)
            for x in missing:
                if can_add(x, current):
                    current[x] = True
                    current_count += 1
                    improvement_found = True
                    if current_count > best_count:
                        best_count = current_count
                        best_S = current.copy()

            if improvement_found:
                continue

            # Try swaps: remove y, try to add x, then try to fill more
            present = [x for x in range(1, N + 1) if current[x]]
            if not missing or not present:
                break

            # Try multiple random swaps
            for _ in range(min(200, len(present) * len(missing))):
                if time.time() > deadline:
                    break

                y = rng.choice(present)
                x = rng.choice(missing)

                # Temporarily remove y
                current[y] = False
                # Try to add x
                if can_add(x, current):
                    current[x] = True
                    # Try to add more elements now
                    extra_added = 0
                    remaining_missing = [z for z in missing if z != x and not current[z]]
                    rng.shuffle(remaining_missing)
                    for z in remaining_missing:
                        if can_add(z, current):
                            current[z] = True
                            extra_added += 1

                    if extra_added > 0:
                        current_count += extra_added
                        improvement_found = True
                        if current_count > best_count:
                            best_count = current_count
                            best_S = current.copy()
                    else:
                        # Revert: remove x, restore y
                        current[x] = False
                        current[y] = True
                else:
                    # Revert
                    current[y] = True

            if not improvement_found and iteration > 50:
                # Perturb: randomly remove some elements and refill
                n_remove = rng.randint(3, 10)
                to_remove = rng.sample(present, min(n_remove, len(present)))
                for y in to_remove:
                    current[y] = False
                current_count = count(current)

                # Refill greedily
                missing = [x for x in range(1, N + 1) if not current[x]]
                rng.shuffle(missing)
                for x in missing:
                    if can_add(x, current):
                        current[x] = True
                        current_count += 1

                if current_count > best_count:
                    best_count = current_count
                    best_S = current.copy()

    # Strategy 6: Multiple restarts with perturbation
    if best_S is not None:
        base = best_S.copy()
        rng2 = random.Random(456)

        for pert in range(50):
            if time.time() > deadline:
                break

            current = base.copy()
            # Randomly remove 5-20 elements
            present = [x for x in range(1, N + 1) if current[x]]
            n_remove = min(len(present), rng2.randint(5, 20))
            to_remove = rng2.sample(present, n_remove)
            for y in to_remove:
                current[y] = False
            current_count = count(current)

            # Greedy fill with random order
            missing = [x for x in range(1, N + 1) if not current[x]]
            rng2.shuffle(missing)
            for x in missing:
                if can_add(x, current):
                    current[x] = True
                    current_count += 1

            try_solution(current)

            # Further local search
            improvement = True
            while improvement and time.time() < deadline:
                improvement = False
                missing = [x for x in range(1, N + 1) if not current[x]]
                rng2.shuffle(missing)
                for x in missing:
                    if can_add(x, current):
                        current[x] = True
                        current_count += 1
                        improvement = True
                        if current_count > best_count:
                            best_count = current_count
                            best_S = current.copy()

    # Strategy 7: Try different orderings based on modular arithmetic
    if time.time() < deadline:
        for mod in [3, 5, 7, 11, 13, 16, 17, 32]:
            if time.time() > deadline:
                break
            order = list(range(1, N + 1))
            order.sort(key=lambda x: (x % mod, x))
            S_bool = np.zeros(N + 1, dtype=bool)
            S_bool, _ = greedy(order, S_bool)
            try_solution(S_bool)

    # Final verification and return
    if best_S is None:
        return [1]

    result = [x for x in range(1, N + 1) if best_S[x]]
    return sorted(result)

# EVOLVE-BLOCK-END
