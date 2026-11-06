import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 84  # safety margin before 90s kill
    N = 192

    # --- Precompute pair-sum residues for fast incremental checks ---
    # For a given a, the "forbidden" pairs (b,c) for adding x are those where (x+b+c)%a==0.
    # We precompute, for each a in 1..192, a dict: residue -> list of (b,c) pairs from [1..192]
    # Actually simpler: for each a, for each x, the forbidden (b,c) are those with (b+c) % a == (-x) % a.
    # Precompute pair_sums_by_residue[a][r] = list of pairs (b,c), b<c, 1<=b,c<=192, with (b+c)%a == r
    # But that's huge. Instead, for the incremental check with a fixed S, we just iterate pairs in S.
    # The real speedup: use a set of (b,c) present in S for O(1) lookup.

    def is_valid_with_incremental(S, x):
        """Check if adding x to valid set S keeps it valid."""
        S_set = set(S)
        # 1) x as tested element: does x divide sum of any 3 distinct in S?
        if len(S) >= 3:
            for t in combinations(S, 3):
                if (t[0] + t[1] + t[2]) % x == 0:
                    return False
        # 2) For each a in S: check new triples {x, b, c} with b,c in S\{a}, b!=c
        #    Need (x + b + c) % a == 0 for some pair (b,c) in S, b != c, b != a, c != a.
        for a in S:
            target = (-x) % a  # need (b+c) % a == target
            # iterate pairs in S excluding a
            for i in range(len(S)):
                b = S[i]
                if b == a:
                    continue
                for j in range(i + 1, len(S)):
                    c = S[j]
                    if c == a:
                        continue
                    if (b + c) % a == target:
                        return False
        return True

    def is_valid_full(S):
        """Full validation."""
        n = len(S)
        if n < 4:
            return True
        for i, a in enumerate(S):
            others = [v for j, v in enumerate(S) if j != i]
            for t in combinations(others, 3):
                if (t[0] + t[1] + t[2]) % a == 0:
                    return False
        return True

    def greedy_build(order, deadline):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_with_incremental(selected, x):
                selected.append(x)
        return selected

    # --- Phase 1: Greedy with multiple orderings ---
    best = []
    # Descending (larger numbers less likely to divide sums)
    cand = greedy_build(range(N, 0, -1), deadline)
    if len(cand) > len(best):
        best = cand
    # Ascending
    cand = greedy_build(range(1, N + 1), deadline)
    if len(cand) > len(best):
        best = cand
    # Random restarts
    rng = random.Random(12345)
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_build(order, deadline)
        if len(cand) > len(best):
            best = cand

    best = list(best)
    best_len = len(best)

    # --- Phase 2: Local search ---
    all_vals = set(range(1, N + 1))

    def try_add_swap(current, remove_idx, add_val, deadline):
        if add_val in current:
            return None
        candidate = [v for i, v in enumerate(current) if i != remove_idx]
        candidate.append(add_val)
        if is_valid_full(candidate):
            return candidate
        return None

    while time.time() < deadline - 3:
        improved = False
        cur = list(best)
        # Try 1-for-1 swaps to increase size (only useful if we can add without removing, but set is "maxed")
        # Actually try to ADD an element first
        for x in range(N, 0, -1):
            if x in cur:
                continue
            if is_valid_with_incremental(cur, x):
                cur.append(x)
                best = cur
                best_len += 1
                improved = True
                break
        if improved:
            continue

        # Try 1-for-1 swaps (replace one element, might allow adding more later)
        for i in range(len(cur)):
            if time.time() > deadline - 3:
                break
            removed = cur[i]
            for x in range(N, 0, -1):
                if x in cur or x == removed:
                    continue
                candidate = try_add_swap(cur, i, x, deadline)
                if candidate is not None:
                    # Same size but different composition; try to then add
                    # Only accept if it leads to a bigger set after an add attempt
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Try 2-for-1: remove two, add one (to free up constraints)
        found = False
        for i in range(len(cur)):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, len(cur)):
                if time.time() > deadline - 3:
                    break
                candidate = [v for k, v in enumerate(cur) if k != i and k != j]
                # Try adding the largest available
                remaining = [x for x in range(N, 0, -1) if x not in candidate]
                for x in remaining[:30]:
                    if is_valid_with_incremental(candidate, x):
                        best = candidate + [x]
                        best_len = len(best)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            # Try 1-for-2: remove one, add two
            found2 = False
            for i in range(len(cur)):
                if time.time() > deadline - 3:
                    break
                candidate = [v for k, v in enumerate(cur) if k != i]
                remaining = [x for x in range(N, 0, -1) if x not in candidate]
                top = remaining[:15]
                for a1 in top:
                    if time.time() > deadline - 3:
                        break
                    cand1 = candidate + [a1]
                    for a2 in top:
                        if a2 == a1:
                            continue
                        if is_valid_with_incremental(cand1, a2):
                            best = cand1 + [a2]
                            best_len = len(best)
                            found2 = True
                            break
                    if found2:
                        break
                if found2:
                    break
            if not found2:
                break

    return best
