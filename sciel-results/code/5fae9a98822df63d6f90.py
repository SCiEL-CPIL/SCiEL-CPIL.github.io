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

    # Strategy 3: Bit-reversal order (good for AP-free structures)
    S_bool = np.zeros(N + 1, dtype=bool)
    bit_rev_order = []
    for x in range(1, N + 1):
        # Reverse the bits of (x-1) and add 1 to get 1-indexed bit-reversed position
        x0 = x - 1
        rev = 0
        for i in range(9):  # 512 = 2^9
            if x0 & (1 << i):
                rev |= (1 << (8 - i))
        bit_rev_order.append(rev + 1)
    S_bool, _ = greedy(bit_rev_order, S_bool)
    try_solution(S_bool)

    # Strategy 4: Modular pattern orders
    for mod in [7, 11, 13, 17]:
        if time.time() > deadline:
            break
        S_bool = np.zeros(N + 1, dtype=bool)
        mod_order = []
        for r in range(mod):
            for x in range(1, N + 1):
                if (x - 1) % mod == r:
                    mod_order.append(x)
        S_bool, _ = greedy(mod_order, S_bool)
        try_solution(S_bool)

    # Strategy 5: Greedy with random orders (increased count)
    for seed in range(50):
        if time.time() > deadline:
            break
        S_bool = np.zeros(N + 1, dtype=bool)
        order = list(range(1, N + 1))
        random.Random(seed).shuffle(order)
        S_bool, _ = greedy(order, S_bool)
        try_solution(S_bool)

    # Strategy 6: Local search with add-only and swap moves
    if best_S is not None:
        current = best_S.copy()
        current_count = best_count
        rng = random.Random(123)
        max_rounds = 100
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improvement_found = False

            # Try to add any missing element (full scan)
            missing = [x for x in range(1, N + 1) if not current[x]]
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

            # Try systematic swaps with priority on high-conflict removals
            tried_swaps = 0
            max_swaps = 500
            while tried_swaps < max_swaps and time.time() < deadline:
                # Prioritize removing elements that are in many APs
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

            if not improvement_found and rnd >= 10:
                # No improvement after several rounds, break
                break

        try_solution(current)

    # Strategy 7: Perturbation + greedy fill + local search (increased iterations)
    if best_S is not None:
        base = best_S.copy()
        rng = random.Random(456)
        for pert in range(80):
            if time.time() > deadline:
                break
            current = base.copy()
            present = [x for x in range(1, N + 1) if current[x]]
            n_remove = min(len(present), rng.randint(5, 25))
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

            # Local search: add-only with full scan
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

    # Strategy 8: Final polishing pass - try all pairs of removals
    if best_S is not None:
        final = best_S.copy()
        final_count = best_count
        present = [x for x in range(1, N + 1) if final[x]]
        missing = [x for x in range(1, N + 1) if not final[x]]
        
        if missing:
            # Try removing 2 elements and adding back
            for i in range(len(present)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(present)):
                    y1, y2 = present[i], present[j]
                    final[y1] = False
                    final[y2] = False
                    added = 0
                    for x in missing:
                        if can_add_np(x, final):
                            final[x] = True
                            added += 1
                    if added > 2:
                        final_count += added - 2
                        if final_count > best_count:
                            best_count = final_count
                            best_S = final.copy()
                    else:
                        # Revert
                        for x in missing:
                            final[x] = False
                        final[y1] = True
                        final[y2] = True
                        final_count = best_count
                    final[y1] = True
                    final[y2] = True

    # Convert to list of integers
    if best_S is None:
        return [1]

    result = [x for x in range(1, N + 1) if best_S[x]]
    return sorted(result)

# EVOLVE-BLOCK-END
