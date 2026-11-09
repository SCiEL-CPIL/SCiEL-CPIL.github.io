import time
import random
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin before 90s kill

    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element x, build list of "witness triples" - 3-element tuples
    # such that x together with the triple forms a 4-AP.
    conflict = [[] for _ in range(N + 1)]  # 1-indexed
    for (a, b, c, d) in aps:
        conflict[a].append((b, c, d))
        conflict[b].append((a, c, d))
        conflict[c].append((a, b, d))
        conflict[d].append((a, b, c))

    # Precompute witness arrays for fast numpy-based checking
    witness_arrays = [None] * (N + 1)
    for x in range(1, N + 1):
        if conflict[x]:
            witness_arrays[x] = np.array(conflict[x], dtype=np.int32)
        else:
            witness_arrays[x] = np.empty((0, 3), dtype=np.int32)

    def can_add_np(x, S_bool):
        """Check if x can be added, using numpy for the witness check."""
        wa = witness_arrays[x]
        if len(wa) == 0:
            return True
        return not np.any(np.all(S_bool[wa], axis=1))

    def greedy(order, S_bool):
        """Greedy: iterate through order, add x if possible."""
        added = 0
        for x in order:
            if not S_bool[x]:
                if can_add_np(x, S_bool):
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

    # Strategy 3: Greedy with random orders
    for seed in range(30):
        if time.time() > deadline:
            break
        S_bool = np.zeros(N + 1, dtype=bool)
        order = list(range(1, N + 1))
        random.Random(seed).shuffle(order)
        S_bool, _ = greedy(order, S_bool)
        try_solution(S_bool)

    # Strategy 4: Local search with add-only and swap moves
    if best_S is not None:
        current = best_S.copy()
        current_count = best_count
        rng = random.Random(123)
        max_rounds = 50
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improvement_found = False

            # Try to add any missing element
            missing = [x for x in range(1, N + 1) if not current[x]]
            rng.shuffle(missing)
            for x in missing:
                if can_add_np(x, current):
                    current[x] = True
                    current_count += 1
                    improvement_found = True
                    if current_count > best_count:
                        best_count = current_count
                        best_S = current.copy()
            if improvement_found:
                continue

            # Try 2-opt style swaps: remove y, add x, then try to add more
            present = [x for x in range(1, N + 1) if current[x]]
            missing = [x for x in range(1, N + 1) if not current[x]]
            if not missing:
                break

            # Try systematic swaps: for each missing x, find best y to remove
            tried_swaps = 0
            max_swaps = 200
            while tried_swaps < max_swaps and time.time() < deadline:
                y = rng.choice(present)
                x = rng.choice(missing)
                tried_swaps += 1

                current[y] = False
                if can_add_np(x, current):
                    current[x] = True
                    # Try to greedily add more
                    added_more = 0
                    for z in missing:
                        if z != x and not current[z] and can_add_np(z, current):
                            current[z] = True
                            added_more += 1
                    if added_more > 0:
                        current_count += added_more
                        improvement_found = True
                        if current_count > best_count:
                            best_count = current_count
                            best_S = current.copy()
                    else:
                        # Revert if no improvement
                        current[x] = False
                        current[y] = True
                else:
                    current[y] = True

            if not improvement_found and rnd >= 5:
                # No improvement after several rounds, break
                break

        try_solution(current)

    # Strategy 5: Perturbation + greedy fill + local search
    if best_S is not None:
        base = best_S.copy()
        rng = random.Random(456)
        for pert in range(40):
            if time.time() > deadline:
                break
            current = base.copy()
            present = [x for x in range(1, N + 1) if current[x]]
            n_remove = min(len(present), rng.randint(5, 20))
            to_remove = rng.sample(present, n_remove)
            for y in to_remove:
                current[y] = False
            current_count = count(current)

            # Greedy fill with random order
            missing = [x for x in range(1, N + 1) if not current[x]]
            rng.shuffle(missing)
            for x in missing:
                if can_add_np(x, current):
                    current[x] = True
                    current_count += 1

            try_solution(current)

            # Local search: add-only
            improvement = True
            while improvement and time.time() < deadline:
                improvement = False
                missing = [x for x in range(1, N + 1) if not current[x]]
                for x in missing:
                    if can_add_np(x, current):
                        current[x] = True
                        current_count += 1
                        improvement = True
                        if current_count > best_count:
                            best_count = current_count
                            best_S = current.copy()

    # Convert to list of integers
    if best_S is None:
        return [1]

    result = [x for x in range(1, N + 1) if best_S[x]]
    return sorted(result)

# EVOLVE-BLOCK-END
