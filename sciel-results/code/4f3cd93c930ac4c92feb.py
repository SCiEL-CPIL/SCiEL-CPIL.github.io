# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Construct a large family of 4-element subsets of [1,16] such that no three distinct sets
    form a sunflower (including empty-core). Uses a greedy randomized independent-set approach
    on the conflict graph, with local improvement.
    """
    deadline = time.time() + 85.0  # safety margin
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n + 1), k))
    m = len(all_sets)  # 2184

    # Convert to frozensets and index
    set_map = {i: frozenset(s) for i, s in enumerate(all_sets)}

    # Precompute pairwise intersections as frozensets
    # inter[i][j] = frozenset of intersection
    inter = [[None] * m for _ in range(m)]
    for i in range(m):
        si = set_map[i]
        inter[i][i] = frozenset()
        for j in range(i + 1, m):
            sj = set_map[j]
            inter[i][j] = frozenset(si & sj)
            inter[j][i] = inter[i][j]

    def is_sunflower(i, j, l):
        """Check if sets i, j, l form a sunflower: all three pairwise intersections equal."""
        return inter[i][j] == inter[i][l] == inter[j][l]

    def greedy_pass(order, deadline):
        """Greedy pass: add set if it doesn't form a sunflower with any two already chosen."""
        chosen = []
        chosen_set = set()
        for idx in order:
            if time.time() > deadline:
                break
            if idx in chosen_set:
                continue
            ok = True
            c = len(chosen)
            for a in range(c):
                if not ok:
                    break
                ia = chosen[a]
                for b in range(a + 1, c):
                    ib = chosen[b]
                    if is_sunflower(ia, ib, idx):
                        ok = False
                        break
            if ok:
                chosen.append(idx)
                chosen_set.add(idx)
        return chosen

    # Initial greedy passes
    best = []

    # Pass 1: natural order
    order = list(range(m))
    result = greedy_pass(order, deadline)
    best = list(result)

    # Pass 2: reverse order
    order = list(range(m - 1, -1, -1))
    result = greedy_pass(order, deadline)
    if len(result) > len(best):
        best = list(result)

    # Pass 3+: random shuffles
    rng = random.Random(42)
    num_random = 20
    for trial in range(num_random):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        result = greedy_pass(order, deadline)
        if len(result) > len(best):
            best = list(result)

    # Local improvement: try to add any unchosen set that doesn't create a sunflower
    if time.time() < deadline - 3.0:
        improved = True
        best_set = set(best)
        while improved and time.time() < deadline - 1.0:
            improved = False
            for idx in range(m):
                if time.time() > deadline:
                    break
                if idx in best_set:
                    continue
                ok = True
                c = len(best)
                for a in range(c):
                    if not ok:
                        break
                    ia = best[a]
                    for b in range(a + 1, c):
                        ib = best[b]
                        if is_sunflower(ia, ib, idx):
                            ok = False
                            break
                if ok:
                    best.append(idx)
                    best_set.add(idx)
                    improved = True

    # Final swap-based local search: try replacing one chosen set with one unchosen
    # to see if we can gain size (this is complex, so just do a simple one-pass)
    if time.time() < deadline - 2.0:
        best_set = set(best)
        for idx in range(m):
            if time.time() > deadline:
                break
            if idx in best_set:
                continue
            # Try adding idx; if it conflicts, see if we can remove one conflicting set
            # For simplicity, skip this for now
            pass

    return [sorted(all_sets[i]) for i in best]
# EVOLVE-BLOCK-END
