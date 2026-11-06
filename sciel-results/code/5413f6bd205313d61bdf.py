import time
import math
import random
import sys
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """All (a,b,c) with distinct a,b,c in [1,N] and 1/a = 1/b + 1/c."""
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180

    triples = _precompute_triples(N)
    T = len(triples)

    # element (1..N) -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    # Precompute triple conflict count for quick feasibility checks
    # triple_members[t] = set of elements in triple t
    triple_members = [set() for _ in range(T)]
    for idx, (a, b, c) in enumerate(triples):
        triple_members[idx] = {a, b, c}

    def is_feasible(S, x):
        """Check if adding x to S is feasible (no triple fully contained)."""
        for t_idx in elem_triples[x]:
            a, b, c = triples[t_idx]
            if (a in S and b in S and c in S):
                return False
            # More precisely: check if all three members of this triple are in S∪{x}
            # Since x is one of them, check if the other two are in S
            others = [v for v in (a, b, c) if v != x]
            if others[0] in S and others[1] in S:
                return False
        return True

    def greedy_seed(order=None, random_seed=None):
        """Greedy: repeatedly add the element that creates the fewest new conflicts.
        order: list of elements to consider, or None for default."""
        S = set()
        counts = [0] * T  # counts[t] = number of chosen elements in triple t
        in_S = [False] * (N + 1)
        
        if order is None:
            order = list(range(1, N + 1))
        
        # For each step, pick the element from remaining that has min cost
        remaining = set(order)
        while remaining:
            if time.time() > deadline - 3.0:
                break
            best_x = -1
            best_cost = float('inf')
            for x in remaining:
                if in_S[x]:
                    continue
                cost = 0
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        cost = 10**9
                        break
                    else:
                        cost += 1
                if cost < best_cost or (cost == best_cost and (best_x == -1 or x < best_x)):
                    best_cost = cost
                    best_x = x
            if best_x == -1:
                break
            S.add(best_x)
            in_S[best_x] = True
            remaining.discard(best_x)
            for t_idx in elem_triples[best_x]:
                counts[t_idx] += 1
        return S, counts

    def local_search_add(S, counts, random_seed, max_rounds=10):
        """Try to add back any element that doesn't violate a triple."""
        random.seed(random_seed)
        S = set(S)
        counts = list(counts)
        removed = [x for x in range(1, N + 1) if x not in S]
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            random.shuffle(removed)
            added_any = False
            new_removed = []
            for x in removed:
                if x in S:
                    continue
                ok = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                    added_any = True
                else:
                    new_removed.append(x)
            removed = new_removed
            if not added_any:
                break
        return S, counts

    def swap_local_search(S, counts, random_seed, max_rounds=20):
        """Try swapping: remove one element, then greedily add removed elements."""
        random.seed(random_seed)
        S = set(S)
        counts = list(counts)
        removed = [x for x in range(1, N + 1) if x not in S]
        best_S = set(S)
        best_counts = list(counts)
        best_size = len(S)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            if not S:
                break
            x_remove = random.choice(list(S))
            S.discard(x_remove)
            for t_idx in elem_triples[x_remove]:
                counts[t_idx] -= 1
            # Now try to add as many removed elements as possible
            random.shuffle(removed)
            for y in removed:
                if y in S:
                    continue
                ok = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                best_counts = list(counts)
                removed = [x for x in range(1, N + 1) if x not in best_S]
            else:
                # Revert
                S = set(best_S)
                counts = list(best_counts)
                removed = [x for x in range(1, N + 1) if x not in S]
        return best_S, best_counts

    def branch_and_bound_with_lp(S, counts, remaining, depth, max_depth=25):
        """Branch and bound with LP relaxation upper bound."""
        nonlocal best_size, best_S, best_counts
        if time.time() > deadline - 3.0:
            return
        if depth > max_depth:
            return
        if not remaining:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                best_counts = list(counts)
            return

        # Upper bound: |S| + |remaining|
        if len(S) + len(remaining) <= best_size:
            return

        # LP relaxation bound: solve fractional packing on remaining
        # For each remaining element, it can contribute at most 1
        # But triples constrain: for each triple, sum of y_i <= 1
        # Cheap bound: count active conflicts
        active_conflicts = 0
        for x in remaining:
            for t_idx in elem_triples[x]:
                if counts[t_idx] >= 2:
                    active_conflicts += 1
                    break
        # Each element with active conflict can't be added directly
        # Simple bound: |S| + (number of remaining with no active conflict)
        free_count = sum(1 for x in remaining if all(counts[t_idx] < 2 for t_idx in elem_triples[x]))
        if len(S) + free_count <= best_size:
            return

        x = remaining[0]
        rest = remaining[1:]

        # Option 1: include x (if feasible)
        feasible = True
        for t_idx in elem_triples[x]:
            if counts[t_idx] >= 2:
                feasible = False
                break
        if feasible:
            counts2 = list(counts)
            for t_idx in elem_triples[x]:
                counts2[t_idx] += 1
            branch_and_bound_with_lp(S | {x}, counts2, rest, depth + 1, max_depth)
            if time.time() > deadline - 3.0:
                return

        # Option 2: exclude x
        branch_and_bound_with_lp(S, counts, rest, depth + 1, max_depth)

    # Generate diverse greedy seeds
    best_S = set()
    best_counts = [0] * T
    best_size = 0

    # Seed 1: ascending
    S1, C1 = greedy_seed(order=list(range(1, N + 1)))
    if len(S1) > best_size:
        best_size = len(S1)
        best_S = set(S1)
        best_counts = list(C1)

    # Seed 2: descending
    S2, C2 = greedy_seed(order=list(range(N, 0, -1)))
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = set(S2)
        best_counts = list(C2)

    # Seed 3: random order
    random.seed(42)
    order3 = list(range(1, N + 1))
    random.shuffle(order3)
    S3, C3 = greedy_seed(order=order3)
    if len(S3) > best_size:
        best_size = len(S3)
        best_S = set(S3)
        best_counts = list(C3)

    # Seed 4: by conflict degree (elements in fewer triples first)
    conflict_deg = {x: len(elem_triples[x]) for x in range(1, N + 1)}
    order4 = sorted(range(1, N + 1), key=lambda x: (conflict_deg[x], x))
    S4, C4 = greedy_seed(order=order4)
    if len(S4) > best_size:
        best_size = len(S4)
        best_S = set(S4)
        best_counts = list(C4)

    # Seed 5: reverse conflict degree
    order5 = sorted(range(1, N + 1), key=lambda x: (-conflict_deg[x], x))
    S5, C5 = greedy_seed(order=order5)
    if len(S5) > best_size:
        best_size = len(S5)
        best_S = set(S5)
        best_counts = list(C5)

    # Local search on best seed
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        S_l, c_l = local_search_add(best_S, best_counts, seed)
        if len(S_l) > best_size:
            best_size = len(S_l)
            best_S = set(S_l)
            best_counts = list(c_l)

    # Swap local search
    for seed in range(30):
        if time.time() > deadline - 5.0:
            break
        S_s, c_s = swap_local_search(best_S, best_counts, seed + 200, max_rounds=15)
        if len(S_s) > best_size:
            best_size = len(S_s)
            best_S = set(S_s)
            best_counts = list(c_s)

    # B&B on a reduced problem
    removed = [x for x in range(1, N + 1) if x not in best_S]
    def conflict_degree(x):
        d = 0
        for t_idx in elem_triples[x]:
            if best_counts[t_idx] >= 2:
                d += 1
        return d
    removed.sort(key=lambda x: (-conflict_degree(x), x))

    # Branch on up to 15 most constrained removed elements
    K = 15
    branch_set = removed[:K]

    def bb_with_greedy(S, counts, branch_elems, depth=0):
        nonlocal best_size, best_S, best_counts
        if time.time() > deadline - 3.0:
            return
        if depth > 20:
            return
        if not branch_elems:
            # Greedily add all remaining feasible elements
            S2 = set(S)
            counts2 = list(counts)
            for x in range(1, N + 1):
                if x in S2:
                    continue
                ok = True
                for t_idx in elem_triples[x]:
                    if counts2[t_idx] >= 2:
                        ok = False
                        break
                if ok:
                    S2.add(x)
                    for t_idx in elem_triples[x]:
                        counts2[t_idx] += 1
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = set(S2)
                best_counts = list(counts2)
            return

        x = branch_elems[0]
        rest = branch_elems[1:]

        # Include x if feasible
        feasible = True
        for t_idx in elem_triples[x]:
            if counts[t_idx] >= 2:
                feasible = False
                break
        if feasible:
            counts2 = list(counts)
            for t_idx in elem_triples[x]:
                counts2[t_idx] += 1
            bb_with_greedy(S | {x}, counts2, rest, depth + 1)
            if time.time() > deadline - 3.0:
                return

        # Exclude x
        bb_with_greedy(S, counts, rest, depth + 1)

    bb_with_greedy(best_S, best_counts, branch_set)

    # More local search iterations after B&B
    for seed in range(100):
        if time.time() > deadline - 3.0:
            break
        S_l, c_l = local_search_add(best_S, best_counts, seed + 500)
        if len(S_l) > best_size:
            best_size = len(S_l)
            best_S = set(S_l)
            best_counts = list(c_l)
        S_s, c_s = swap_local_search(best_S, best_counts, seed + 700, max_rounds=10)
        if len(S_s) > best_size:
            best_size = len(S_s)
            best_S = set(S_s)
            best_counts = list(c_s)

    # Final verification: remove elements to break any remaining triples
    changed = True
    while changed:
        changed = False
        for (a, b, c) in triples:
            if a in best_S and b in best_S and c in best_S:
                # Remove the element with highest conflict degree to minimize damage
                degs = [conflict_degree(a), conflict_degree(b), conflict_degree(c)]
                max_idx = degs.index(max(degs))
                to_remove = [a, b, c][max_idx]
                best_S.discard(to_remove)
                changed = True
                break

    if not best_S:
        best_S = set(range(90, 181))

    return list(best_S)

# EVOLVE-BLOCK-END
