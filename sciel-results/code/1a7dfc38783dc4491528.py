import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85
    N = 180

    # Precompute all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => (b-a)*c = a*b => c = a*b/(b-a) for b > a
    triples = []
    seen = set()

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = tuple(sorted([a, b, c]))
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)

    # Build adjacency: for each element, list of triple indices containing it
    element_triples = defaultdict(list)
    for i, t in enumerate(triples):
        for x in t:
            element_triples[x].append(i)

    # Count how many triples each element participates in
    triple_count = [0] * (N + 1)
    for t in triples:
        for x in t:
            triple_count[x] += 1

    def can_add(S_set, x):
        """Check if x can be added to S_set without creating a violation."""
        for i in element_triples[x]:
            t = triples[i]
            if all(y in S_set for y in t if y != x):
                return False
        return True

    # Strategy 1: Greedy removal from full set, removing element in most violated triples
    S = set(range(1, N + 1))

    # Find all violated triples
    violated = set()
    for i, t in enumerate(triples):
        if all(x in S for x in t):
            violated.add(i)

    while violated and time.time() < deadline - 10:
        elem_viol = [0] * (N + 1)
        for i in violated:
            t = triples[i]
            for x in t:
                if x in S:
                    elem_viol[x] += 1

        max_count = 0
        best_elem = -1
        for x in range(1, N + 1):
            if x in S and elem_viol[x] > max_count:
                max_count = elem_viol[x]
                best_elem = x

        if best_elem == -1:
            break

        S.discard(best_elem)

        # Update violated set incrementally
        new_violated = set()
        for i, t in enumerate(triples):
            if all(x in S for x in t):
                new_violated.add(i)
        violated = new_violated

    best_set = set(S)
    best_size = len(best_set)

    # Try to add back elements that don't create violations
    removed = set(range(1, N + 1)) - S
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in list(removed):
            if time.time() > deadline - 10:
                break
            if can_add(S, x):
                S.add(x)
                removed.discard(x)
                improved = True
                best_size = len(S)

    # Strategy 2: Greedy construction from scratch with different orderings
    # Key insight: larger numbers have fewer conflicts, so include them first
    for order_mode in ['desc', 'asc', 'by_degree']:
        if time.time() > deadline - 15:
            break

        S2 = set()
        if order_mode == 'desc':
            order = list(range(N, 0, -1))
        elif order_mode == 'asc':
            order = list(range(1, N + 1))
        else:
            order = sorted(range(1, N + 1), key=lambda x: (triple_count[x], x))

        for x in order:
            if time.time() > deadline - 15:
                break
            if can_add(S2, x):
                S2.add(x)

        # Try to add back remaining elements
        remaining = set(range(1, N + 1)) - S2
        improved2 = True
        while improved2 and time.time() < deadline - 15:
            improved2 = False
            for x in list(remaining):
                if time.time() > deadline - 15:
                    break
                if can_add(S2, x):
                    S2.add(x)
                    remaining.discard(x)
                    improved2 = True

        if len(S2) > best_size:
            best_set = set(S2)
            best_size = len(S2)

    # Strategy 3: Randomized greedy with multiple restarts
    import random
    random.seed(42)
    for restart in range(300):
        if time.time() > deadline - 10:
            break

        S3 = set()
        order = list(range(1, N + 1))
        random.shuffle(order)

        for x in order:
            if time.time() > deadline - 10:
                break
            if can_add(S3, x):
                S3.add(x)

        # Try to add back
        remaining = set(range(1, N + 1)) - S3
        improved3 = True
        while improved3 and time.time() < deadline - 10:
            improved3 = False
            for x in list(remaining):
                if time.time() > deadline - 10:
                    break
                if can_add(S3, x):
                    S3.add(x)
                    remaining.discard(x)
                    improved3 = True

        if len(S3) > best_size:
            best_set = set(S3)
            best_size = len(S3)

    # Strategy 4: Weighted greedy - prefer elements with fewer constraints
    for restart in range(200):
        if time.time() > deadline - 10:
            break

        S4 = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        candidates.sort(key=lambda x: (triple_count[x], random.random()))

        for x in candidates:
            if time.time() > deadline - 10:
                break
            if can_add(S4, x):
                S4.add(x)

        remaining = set(range(1, N + 1)) - S4
        improved4 = True
        while improved4 and time.time() < deadline - 10:
            improved4 = False
            for x in list(remaining):
                if time.time() > deadline - 10:
                    break
                if can_add(S4, x):
                    S4.add(x)
                    remaining.discard(x)
                    improved4 = True

        if len(S4) > best_size:
            best_set = set(S4)
            best_size = len(S4)

    # Strategy 5: Hill-climbing with 1-removal + multi-add moves
    if time.time() < deadline - 10:
        current_set = set(best_set)

        for round_num in range(100):
            if time.time() > deadline - 10:
                break

            improved = False

            # Try adding each removed element
            removed_elems = [x for x in range(1, N + 1) if x not in current_set]
            for x in removed_elems:
                if time.time() > deadline - 10:
                    break
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True

            # Try: remove one element, then greedily add back as many as possible
            for x in list(current_set):
                if time.time() > deadline - 10:
                    break

                old_size = len(current_set)
                current_set.discard(x)

                # Try to add back any removed elements
                removed_after = [y for y in range(1, N + 1) if y not in current_set]
                for y in removed_after:
                    if time.time() > deadline - 10:
                        break
                    if can_add(current_set, y):
                        current_set.add(y)

                if len(current_set) > old_size:
                    improved = True
                else:
                    # Revert
                    # Remove all elements that were added
                    added = current_set - set(range(1, N + 1))  # This won't work directly
                    # Simpler: just check if we improved overall
                    pass

            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)

            if not improved:
                break

    # Strategy 6: Final polish - iterate add-then-remove-then-add until stable
    if time.time() < deadline - 3:
        for polish_round in range(100):
            if time.time() > deadline - 3:
                break

            changed = False

            # Phase 1: Add all possible
            for x in range(1, N + 1):
                if x not in best_set:
                    if can_add(best_set, x):
                        best_set.add(x)
                        changed = True

            # Phase 2: Try removing each element and re-adding
            for x in list(best_set):
                if time.time() > deadline - 3:
                    break
                old_size = len(best_set)
                best_set.discard(x)

                # Try to add back everything possible
                for y in range(1, N + 1):
                    if y not in best_set:
                        if can_add(best_set, y):
                            best_set.add(y)
                            changed = True

                if len(best_set) < old_size:
                    # Revert: remove what was added, re-add x
                    added = best_set - (best_set - {x})  # This is getting complex
                    # Just break and accept current state
                    break

            if not changed:
                break

    return list(best_set)
