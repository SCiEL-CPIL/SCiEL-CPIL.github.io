"""Maximum Independent Set on conflict graph for reciprocal Schur triple avoidance."""
import random
import time
import math

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = b*c/(b+c), so (b+c) | b*c. Since 1/a > 1/b and 1/a > 1/c, a < b and a < c.
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))

    # Build conflict graph: edge between b and c if they form a triple with some a
    # This is the key structural change: we need max independent set in this graph
    conflict = {x: set() for x in range(1, N + 1)}
    for a, b, c in triples:
        conflict[b].add(c)
        conflict[c].add(b)

    degree = {x: len(conflict[x]) for x in range(1, N + 1)}

    def is_independent(S):
        """Check if S is an independent set in the conflict graph."""
        for x in S:
            for y in conflict[x]:
                if y in S:
                    return False
        return True

    def greedy_independent(order):
        """Greedy: add vertices in order if they don't conflict with already-added."""
        S = set()
        for x in order:
            if not any(y in S for y in conflict[x]):
                S.add(x)
        return S

    # Phase 1: Multiple greedy initializations
    best = set()
    best_size = 0

    # Degree-based ordering (low degree first - they're "safer")
    degree_order = sorted(range(1, N + 1), key=lambda x: degree[x])
    for order in [degree_order, list(reversed(degree_order)), 
                  list(range(N, 0, -1)), list(range(1, N + 1))]:
        cand = greedy_independent(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Random orderings
    for _ in range(200):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_independent(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Phase 2: Local search with 1-flips (add/remove) and 2-flips (swap)
    # Energy: -|S| + penalty * (# edges in S)
    # We want to maximize |S| subject to # edges in S = 0

    def edge_count(S):
        """Count edges within S in the conflict graph."""
        count = 0
        for x in S:
            for y in conflict[x]:
                if y in S and x < y:
                    count += 1
        return count

    def energy(S):
        return -len(S) + 10.0 * edge_count(S)

    current_S = set(best)
    current_e = energy(current_S)
    best_S = set(current_S)
    best_e = current_e
    best_size = len(current_S)

    T = 3.0
    T_end = 0.01
    cooling = 0.99998
    no_improve = 0
    step = 0

    while T > T_end and time.time() < deadline:
        step += 1

        # Choose move type
        r = rng.random()
        if r < 0.5:
            # 1-flip: add or remove
            if rng.random() < 0.6:
                # Add
                not_in = [x for x in range(1, N + 1) if x not in current_S]
                if not not_in:
                    T *= cooling
                    continue
                x = rng.choice(not_in)
                new_S = current_S | {x}
            else:
                # Remove
                if not current_S:
                    T *= cooling
                    continue
                x = rng.choice(list(current_S))
                new_S = current_S - {x}
        elif r < 0.85:
            # 1-for-1 swap
            in_S = list(current_S)
            not_in = [x for x in range(1, N + 1) if x not in current_S]
            if not in_S or not not_in:
                T *= cooling
                continue
            y = rng.choice(in_S)
            x = rng.choice(not_in)
            new_S = (current_S - {y}) | {x}
        else:
            # 2-for-2 swap
            in_S = list(current_S)
            not_in = [x for x in range(1, N + 1) if x not in current_S]
            if len(in_S) < 2 or len(not_in) < 2:
                T *= cooling
                continue
            y1, y2 = rng.sample(in_S, 2)
            x1, x2 = rng.sample(not_in, 2)
            new_S = (current_S - {y1, y2}) | {x1, x2}

        new_e = energy(new_S)
        delta = new_e - current_e

        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_S = new_S
            current_e = new_e

            if edge_count(current_S) == 0 and len(current_S) > best_size:
                best_S = set(current_S)
                best_size = len(current_S)
                best_e = current_e
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling

        # Periodic polish: add any non-conflicting elements
        if step % 200 == 0 and edge_count(current_S) == 0:
            for x in range(1, N + 1):
                if x not in current_S and not any(y in current_S for y in conflict[x]):
                    current_S.add(x)
                    current_e = energy(current_S)
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_size = len(current_S)

        if no_improve > 30000:
            break

    # Phase 3: Final polish on best solution
    final_S = set(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in final_S:
                continue
            if not any(y in final_S for y in conflict[x]):
                final_S.add(x)
                improved = True
                break

    # Phase 4: Try 1-for-1 swaps to potentially enable more additions
    for _ in range(10000):
        if time.time() >= deadline:
            break
        swapped = False
        for x in range(1, N + 1):
            if x in final_S:
                continue
            # Check if x conflicts with any in final_S
            conflicts_with = [y for y in conflict[x] if y in final_S]
            if not conflicts_with:
                continue
            # Try removing one conflicting element
            for y in conflicts_with:
                trial = (final_S - {y}) | {x}
                if is_independent(trial):
                    # Check if this enables adding more
                    final_S = trial
                    swapped = True
                    break
            if swapped:
                break
        if not swapped:
            break

    # Final polish after swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in final_S:
                continue
            if not any(y in final_S for y in conflict[x]):
                final_S.add(x)
                improved = True
                break

    # Verification
    if not is_independent(final_S):
        # Fallback: use the upper half
        final_S = set(range(91, N + 1))

    return list(final_S)
