"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill

    # Generate all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_masks.append(mask)

    n = len(all_masks)  # C(16,4) = 1820

    # Precompute pairwise intersections as bitmasks
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = all_masks[i]
        for j in range(i + 1, n):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij

    # For each pair (i,j), precompute the set of k that form a sunflower with i,j
    # A triple (i,j,k) is a sunflower iff inter[i][j] == inter[i][k] == inter[j][k]
    # So for pair (i,j) with val = inter[i][j], k is a partner iff
    # inter[i][k] == val AND inter[j][k] == val
    #
    # Build: for each i, a dict mapping intersection value -> list of indices
    inter_dict = [{} for _ in range(n)]
    for i in range(n):
        d = {}
        mi = all_masks[i]
        for j in range(n):
            if i == j:
                continue
            v = mi & all_masks[j]
            if v not in d:
                d[v] = []
            d[v].append(j)
        inter_dict[i] = d

    # Precompute sunflower_partners[i][j] = frozenset of k (k != i,j) such that (i,j,k) is sunflower
    # This is O(n^2 * n) in worst case but with the dict lookup it's faster
    sunflower_partners = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = inter[i][j]
            # k must be in inter_dict[i][val] AND inter_dict[j][val], excluding i and j
            if val not in inter_dict[i] or val not in inter_dict[j]:
                sunflower_partners[(i, j)] = frozenset()
                continue
            candidates = set(inter_dict[i][val]) & set(inter_dict[j][val])
            candidates.discard(i)
            candidates.discard(j)
            sunflower_partners[(i, j)] = frozenset(candidates)

    # Greedy with conflict counting
    # conflict_count[x] = number of pairs (i,j) in chosen such that x is in sunflower_partners[(i,j)]
    # When we add c to chosen, for each existing i in chosen:
    #   new conflicts for x: x is in sunflower_partners[(i,c)]
    # We maintain this incrementally.

    def greedy_with_conflict_order(order, deadline):
        chosen = []
        chosen_set = set()
        conflict_count = [0] * n  # for each candidate, how many conflict pairs with chosen

        for x in order:
            if time.time() > deadline:
                break
            if x in chosen_set:
                continue
            if conflict_count[x] == 0:
                # Add x to chosen
                chosen.append(x)
                chosen_set.add(x)
                # Update conflict counts for all remaining candidates
                for i in chosen[:-1]:  # all previously chosen
                    partners = sunflower_partners.get((min(i, x), max(i, x)), frozenset())
                    for k in partners:
                        if k not in chosen_set:
                            conflict_count[k] += 1

        return chosen

    # Strategy 1: Greedy with natural order
    best = greedy_with_conflict_order(list(range(n)), deadline)
    best_size = len(best)

    if time.time() > deadline:
        result = []
        for x in best:
            m = all_masks[x]
            elems = [i for i in range(1, 17) if (m >> i) & 1]
            result.append(elems)
        return result

    # Strategy 2: Greedy with random order (multiple seeds)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline:
            break
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        fam = greedy_with_conflict_order(order, deadline)
        if len(fam) > best_size:
            best = fam
            best_size = len(fam)

    if time.time() > deadline:
        result = []
        for x in best:
            m = all_masks[x]
            elems = [i for i in range(1, 17) if (m >> i) & 1]
            result.append(elems)
        return result

    # Strategy 3: Local search - try single swaps
    current = list(best)
    current_set = set(best)

    # Precompute for current: for each pair in current, the set of candidates that conflict
    # conflict_with_pair[i][j] = set of x not in current such that x in sunflower_partners[(i,j)]
    # Actually, we need: for each candidate x not in current, which pairs (i,j) in current
    # have x in sunflower_partners[(i,j)]?
    # That's exactly what conflict_count tracks.

    # For local search, we try: remove r, add a. The new set is valid iff
    # no sunflower triple exists in (current \ {r}) ∪ {a}
    # A sunflower triple in the new set must include a (since current \ {r} is valid)
    # So: for all pairs (i,j) in current \ {r}, a must NOT be in sunflower_partners[(i,j)]

    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 50:
        improved = False
        iterations += 1

        # Compute conflict counts for all candidates not in current
        conflict_count = [0] * n
        for idx_i in range(len(current)):
            i = current[idx_i]
            for idx_j in range(idx_i + 1, len(current)):
                j = current[idx_j]
                partners = sunflower_partners.get((min(i, j), max(i, j)), frozenset())
                for k in partners:
                    if k not in current_set:
                        conflict_count[k] += 1

        # Try swaps: for each r in current, for each a not in current with conflict_count[a] == 1
        # (meaning a conflicts with exactly one pair, which must include r)
        # Actually, let's just try all swaps where a has low conflict count
        for r_idx in range(len(current)):
            if time.time() > deadline:
                break
            r = current[r_idx]

            # Find candidates a not in current that could replace r
            # a is valid to add if for all pairs (i,j) in current \ {r}, a not in sunflower_partners[(i,j)]
            # Equivalently, the only pair that might conflict with a must include r
            for a in range(n):
                if a in current_set:
                    continue
                # Check if a can replace r
                # a conflicts with pair (i,j) if a in sunflower_partners[(i,j)]
                # We need: no pair (i,j) in current \ {r} has a in sunflower_partners[(i,j)]
                valid = True
                for i_idx in range(len(current)):
                    if i_idx == r_idx:
                        continue
                    i = current[i_idx]
                    for j_idx in range(i_idx + 1, len(current)):
                        if j_idx == r_idx:
                            continue
                        j = current[j_idx]
                        key = (min(i, j), max(i, j))
                        if a in sunflower_partners.get(key, frozenset()):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Perform the swap
                    current[r_idx] = a
                    current_set.discard(r)
                    current_set.add(a)
                    improved = True
                    break
            if improved:
                break

    best_size = len(current)
    best = current

    # Convert back to sorted lists of elements
    result = []
    for x in best:
        m = all_masks[x]
        elems = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(elems)

    return result
# EVOLVE-BLOCK-END
