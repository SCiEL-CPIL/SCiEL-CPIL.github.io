# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy with multiple orderings + local search (swap/add/remove) within
    a time budget.
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build conflict matrix: conflict[i][j] = True if (i-j) mod m or (j-i) mod m is forbidden
    # i.e., i and j cannot both be in the set
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflict[i, j] = True

    def is_valid(s):
        """Check if set s has no forbidden pairs."""
        s_list = list(s)
        for idx, a in enumerate(s_list):
            for b in s_list[idx+1:]:
                if conflict[a, b]:
                    return False
        return True

    def greedy_random(order, forbidden_set):
        """Greedy: process candidates in given order, add if no conflict with selected."""
        selected = set()
        for cand in order:
            ok = True
            for s in selected:
                if conflict[cand, s]:
                    ok = False
                    break
            if ok:
                selected.add(cand)
        return selected

    def local_search(s, forbidden_set, max_iters=5000):
        """Improve set s via swap, add, and remove operations."""
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            s_list = sorted(s)
            # Try adding
            for cand in range(m):
                if cand not in s:
                    if not any(conflict[cand, x] for x in s):
                        s.add(cand)
                        improved = True
                        break
            if improved:
                continue
            # Try swap: remove one, add another
            for rem in s_list:
                if time.time() > deadline:
                    break
                s_temp = set(s)
                s_temp.discard(rem)
                # Find best addition
                for cand in range(m):
                    if cand not in s_temp:
                        if not any(conflict[cand, x] for x in s_temp):
                            s_temp.add(cand)
                            if len(s_temp) > len(s):
                                s = s_temp
                                improved = True
                                break
                if improved:
                    break
            if improved:
                continue
            # Try single swap (remove one, add one, same size) to enable future improvements
            for rem in s_list:
                if time.time() > deadline:
                    break
                s_without = set(s)
                s_without.discard(rem)
                best_cand = -1
                best_count = -1
                for cand in range(m):
                    if cand not in s_without:
                        if not any(conflict[cand, x] for x in s_without):
                            # Count how many more we could add after this
                            s2 = set(s_without)
                            s2.add(cand)
                            count = 0
                            for c2 in range(m):
                                if c2 not in s2:
                                    if not any(conflict[c2, x] for x in s2):
                                        count += 1
                            if count > best_count:
                                best_count = count
                                best_cand = cand
                if best_cand >= 0 and best_count > 0:
                    s = set(s_without)
                    s.add(best_cand)
                    improved = True
                    break
        return s

    # Strategy 1: deterministic greedy (original)
    best_set = set()
    for cand in range(m):
        if not any(conflict[cand, s] for s in best_set):
            best_set.add(cand)
    best_set = local_search(best_set, forbidden)
    best_size = len(best_set)

    # Strategy 2: randomized greedy with multiple seeds
    rng = random.Random(42)
    for trial in range(20000):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        s = greedy_random(order, forbidden)
        s = local_search(s, forbidden, max_iters=2000)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 3: start from subsets of best_set, perturb, and search
    for trial in range(5000):
        if time.time() > deadline:
            break
        s = set(best_set)
        # Randomly remove 1-3 elements
        n_remove = min(len(s), rng.randint(1, 3))
        to_remove = rng.sample(sorted(s), n_remove)
        for r in to_remove:
            s.discard(r)
        s = local_search(s, forbidden, max_iters=3000)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 4: random restarts with different initial seeds
    for trial in range(10000):
        if time.time() > deadline:
            break
        # Start with a random small seed set
        seed_size = rng.randint(1, 5)
        seed = set(rng.sample(range(m), seed_size))
        if not is_valid(seed):
            # Make valid
            seed = set()
            for x in sorted(rng.sample(range(m), seed_size)):
                if not any(conflict[x, y] for y in seed):
                    seed.add(x)
        s = greedy_random(list(range(m)), forbidden)
        # Merge with seed preference
        s2 = set(seed)
        for cand in range(m):
            if cand not in s2 and not any(conflict[cand, x] for x in s2):
                s2.add(cand)
        s2 = local_search(s2, forbidden, max_iters=1500)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = s2

    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
