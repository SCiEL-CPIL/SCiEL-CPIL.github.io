import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 256

    # Precompute canonical 3-APs and 4-APs
    canonical_aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            if c < n:
                canonical_aps3.append((a, b, c))

    canonical_aps4 = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                canonical_aps4.append((a, b, c, e))

    # Precompute for each element v:
    # partners3[v]: list of (u, w) such that {v, u, w} forms a 3-AP (v is one of the three)
    # partners4[v]: list of (u, w, x) such that {v, u, w, x} forms a 4-AP
    partners3 = [[] for _ in range(n)]
    for a, b, c in canonical_aps3:
        partners3[a].append((b, c))
        partners3[b].append((a, c))
        partners3[c].append((a, b))

    partners4 = [[] for _ in range(n)]
    for a, b, c, d in canonical_aps4:
        partners4[a].append((b, c, d))
        partners4[b].append((a, c, d))
        partners4[c].append((a, b, d))
        partners4[d].append((a, b, c))

    # Precompute bitmask-based fast checks
    # For 3-AP gain when adding v: count pairs (u,w) in partners3[v] where both u,w are in A
    # We can precompute for each v a list, but for speed use bit operations
    # Actually, let's precompute a "partner mask" approach:
    # For each v, create a bitmask pm3[v] where bit i is set if i is a partner of v in some 3-AP
    # But we need to count PAIRS, not individual partners.
    # Better: for each v, store the list and iterate (lists are short, ~O(n) per element max ~50)

    # Precompute degree-like scores for greedy ordering
    # Score of element v = number of 3-APs it participates in (total, not just with existing set)
    deg3 = [len(partners3[v]) for v in range(n)]
    deg4 = [len(partners4[v]) for v in range(n)]

    def count_3aps_mask(A):
        count = 0
        for a, b, c in canonical_aps3:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1:
                count += 1
        return count

    def count_4aps_mask(A):
        count = 0
        for a, b, c, d in canonical_aps4:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1 and (A >> d) & 1:
                count += 1
        return count

    def gain_add(A, v):
        """Number of new 3-APs created by adding v to A."""
        gain = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                gain += 1
        return gain

    def loss_remove(A, v):
        """Number of 3-APs destroyed by removing v from A."""
        loss = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                loss += 1
        return loss

    def can_add(A, v):
        """Check if adding v to A creates no 4-AP."""
        for u, w, x in partners4[v]:
            if ((A >> u) & 1) and ((A >> w) & 1) and ((A >> x) & 1):
                return False
        return True

    def greedy_construct(order):
        """Greedy: add elements in given order if no 4-AP is created."""
        A = 0
        for v in order:
            if can_add(A, v):
                A |= (1 << v)
        return A

    def hill_climb(A, time_limit):
        """Iterative improvement: try all add/remove/swap moves, take best improving move."""
        current_score = count_3aps_mask(A)
        best_score = current_score
        best_A = A

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_move = None  # (type, v, u) where type in {'add','remove','swap'}

            # Try all adds
            for v in range(n):
                if (A >> v) & 1:
                    continue
                if not can_add(A, v):
                    continue
                g = gain_add(A, v)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', v, -1)

            # Try all removes
            for v in range(n):
                if not ((A >> v) & 1):
                    continue
                l = loss_remove(A, v)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', v, -1)

            # Try all swaps
            for u in range(n):
                if not ((A >> u) & 1):
                    continue
                for v in range(n):
                    if (A >> v) & 1:
                        continue
                    if u == v:
                        continue
                    # Check if swap is valid (no 4-AP after removing u, adding v)
                    # A' = A \ {u} U {v}
                    # Check v doesn't complete 4-AP with A\{u}
                    valid = True
                    for w, x, y in partners4[v]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and ((A >> y) & 1):
                            if w != u and x != u and y != u:
                                valid = False
                                break
                    if not valid:
                        continue

                    # Compute delta
                    loss_u = 0
                    for w, x in partners3[u]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and w != v and x != v:
                            loss_u += 1
                    # Note: if w or x is v, that AP doesn't exist yet (v not in A)
                    # Actually v is not in A, so (A >> v) & 1 is 0, so the condition handles it

                    gain_v = 0
                    for w, x in partners3[v]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and w != u and x != u:
                            gain_v += 1

                    delta = gain_v - loss_u
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', v, u)

            if best_move is not None and best_delta > 0:
                mtype, v, u = best_move
                if mtype == 'add':
                    A |= (1 << v)
                elif mtype == 'remove':
                    A &= ~(1 << v)
                else:  # swap
                    A &= ~(1 << u)
                    A |= (1 << v)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
                improved = True

        return best_A, best_score

    def local_search_with_restarts(A, time_limit):
        """Apply hill climbing, then if stuck, try random perturbation and re-climb."""
        A, score = hill_climb(A, time_limit)
        return A, score

    # Main search: multiple greedy constructions with different orderings, then hill-climb each
    best_A_global = 0
    best_score_global = 0

    # Generate different orderings
    orderings = []

    # Ordering 1: by decreasing 3-AP degree
    order1 = sorted(range(n), key=lambda v: (-deg3[v], v))
    orderings.append(order1)

    # Ordering 2: by decreasing (3-AP degree - 4-AP degree)
    order2 = sorted(range(n), key=lambda v: (-(deg3[v] - deg4[v]), v))
    orderings.append(order2)

    # Ordering 3: natural order
    orderings.append(list(range(n)))

    # Ordering 4: reverse natural
    orderings.append(list(range(n - 1, -1, -1)))

    # Ordering 5-8: random shuffles with fixed seeds
    for s in range(4):
        rng = random.Random(s)
        o = list(range(n))
        rng.shuffle(o)
        orderings.append(o)

    # Ordering 9: by increasing 4-AP degree (fewer conflicts first)
    order9 = sorted(range(n), key=lambda v: (deg4[v], -deg3[v]))
    orderings.append(order9)

    # Ordering 10: by decreasing 3-AP degree, tie-break by increasing 4-AP degree
    order10 = sorted(range(n), key=lambda v: (-deg3[v], deg4[v]))
    orderings.append(order10)

    num_orderings = len(orderings)
    time_per_ordering = (deadline - start_time - 2.0) / num_orderings

    for i, order in enumerate(orderings):
        if time.time() > deadline - 3.0:
            break

        A = greedy_construct(order)
        score = count_3aps_mask(A)

        # Hill climb
        time_limit = min(deadline - 2.0, start_time + (i + 1) * time_per_ordering + 1.0)
        A, score = hill_climb(A, time_limit)

        if score > best_score_global:
            best_score_global = score
            best_A_global = A

    # If we have time, do a final round of random perturbations on the best solution
    remaining = deadline - 3.0 - time.time()
    if remaining > 2.0:
        rng = random.Random(42)
        for attempt in range(20):
            if time.time() > deadline - 3.0:
                break
            # Perturb: randomly flip 5-15 bits
            A_pert = best_A_global
            num_flips = rng.randint(5, 15)
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                if (A_pert >> v) & 1:
                    A_pert &= ~(1 << v)
                else:
                    # Only add if no 4-AP
                    if can_add(A_pert, v):
                        A_pert |= (1 << v)

            # Ensure validity: remove any elements that create 4-APs
            # Simple fix: greedily rebuild
            A_fix = 0
            elems = [v for v in range(n) if (A_pert >> v) & 1]
            rng.shuffle(elems)
            for v in elems:
                if can_add(A_fix, v):
                    A_fix |= (1 << v)

            # Hill climb the perturbed solution
            time_limit = min(deadline - 2.0, time.time() + remaining / 20 - 0.5)
            A_hc, score_hc = hill_climb(A_fix, time_limit)

            if score_hc > best_score_global:
                best_score_global = score_hc
                best_A_global = A_hc

    # Convert to list
    best_set = [i + 1 for i in range(n) if (best_A_global >> i) & 1]

    # Final verification
    if count_4aps_mask(best_A_global) != 0:
        # Fallback: greedy from scratch
        A = 0
        for v in range(n):
            if can_add(A, v):
                A |= (1 << v)
        best_set = [i + 1 for i in range(n) if (A >> i) & 1]

    return sorted(best_set)
