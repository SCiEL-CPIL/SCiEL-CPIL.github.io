import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)
    best_family = []
    best_size = 0

    def greedy_build(order, current_best_size=0):
        """
        Greedily build a family from the given order of set indices.
        Uses an efficient data structure to detect sunflowers:
        For each set i in F, inter_maps[i] maps intersection_mask -> list of indices j
        such that F[i] & F[j] == intersection_mask.

        A new set m conflicts with existing F[i], F[j] if there exists i, j in F
        such that m & F[i] == m & F[j] == F[i] & F[j].
        """
        F = []
        inter_maps = []

        for pos, idx in enumerate(order):
            if time.time() > deadline:
                break

            m = all_sets[idx]
            conflict = False

            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})

                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)

                if len(F) >= max_target:
                    break
                # Early termination heuristic
                if len(F) + (len(order) - pos) <= current_best_size:
                    break

        return F

    def local_search_improve(family, max_iterations=300):
        """
        Try to improve the family by swapping out one set for another.
        Uses a more systematic approach: try to find a set that can be
        added by removing one existing set.
        """
        F = list(family)
        in_family = set(F)
        inter_maps = []

        # Rebuild inter_maps
        for idx, mask in enumerate(F):
            inter_maps.append({})

        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                s = F[i] & F[j]
                if s not in inter_maps[i]:
                    inter_maps[i][s] = []
                inter_maps[i][s].append(j)
                if s not in inter_maps[j]:
                    inter_maps[j][s] = []
                inter_maps[j][s].append(i)

        # Pre-compute which sets are NOT in family
        available = [m for m in all_sets if m not in in_family]

        improved = False
        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            if not available:
                break

            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break

                # Pick a random set to remove
                remove_idx = random.randrange(len(F))
                remove_mask = F[remove_idx]

                # Pick a random set to add
                add_mask = random.choice(available)

                # Check if add_mask can replace remove_mask
                # Build F without remove_mask, then check if add_mask fits
                temp_F = F[:remove_idx] + F[remove_idx + 1:]
                conflict = False

                # Check add_mask against all pairs in temp_F
                for i in range(len(temp_F)):
                    s1 = add_mask & temp_F[i]
                    for j in range(i + 1, len(temp_F)):
                        if (add_mask & temp_F[j]) == s1 and (temp_F[i] & temp_F[j]) == s1:
                            conflict = True
                            break
                    if conflict:
                        break

                if not conflict:
                    # Perform swap
                    F[remove_idx] = add_mask
                    in_family.remove(remove_mask)
                    in_family.add(add_mask)
                    available.remove(add_mask)
                    available.append(remove_mask)
                    improved = True
                    break

        return F, improved

    # 1. Deterministic pass (natural order)
    if time.time() < deadline:
        order = list(range(total))
        F = greedy_build(order, best_size)
        if len(F) > best_size:
            best_family = F
            best_size = len(F)

    # 2. Random restarts with different seeds/orders
    rng = random.Random(12345)

    while time.time() < deadline - 5.0:  # Leave 5s for cleanup/return
        order = list(range(total))
        rng.shuffle(order)

        F = greedy_build(order, best_size)

        if len(F) > best_size:
            # Try local search improvement
            F_improved, improved = local_search_improve(F, max_iterations=200)
            if improved and len(F_improved) > best_size:
                best_family = F_improved
                best_size = len(F_improved)
            else:
                best_family = F
                best_size = len(F)

            if best_size >= max_target:
                break

    # Final local search on best family
    if time.time() < deadline - 2.0 and best_family:
        F_final, _ = local_search_improve(best_family, max_iterations=400)
        if len(F_final) > best_size:
            best_family = F_final
            best_size = len(F_final)

    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
