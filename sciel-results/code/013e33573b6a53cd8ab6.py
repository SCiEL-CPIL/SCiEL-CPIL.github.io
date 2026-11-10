import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 128

    # Pre-compute forbidden hyperedges
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            max_c = min(N - b, N - a)
            if max_c <= b:
                continue
            for c in range(b + 1, max_c + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                v = frozenset({a, b, c, s1, s2, s3})
                edges.append(v)

    unique_edges = list(set(edges))

    # Build index: element -> list of edge indices containing it
    elem_to_edges = {i: [] for i in range(1, N + 1)}
    for i, e in enumerate(unique_edges):
        for v in e:
            elem_to_edges[v].append(i)

    def is_valid(S_set):
        for e in unique_edges:
            if e.issubset(S_set):
                return False
        return True

    def check_add(S_set, x):
        for idx in elem_to_edges[x]:
            e = unique_edges[idx]
            if e.difference({x}).issubset(S_set):
                return False
        return True

    def greedy_complete(S, rng):
        """Add elements greedily if valid."""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x not in S and check_add(S, x):
                    S.add(x)
                    improved = True
        return S

    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)

    rng = random.Random(42)

    for restart in range(40):
        if time.time() > deadline:
            break

        if restart == 0:
            S = set(range(N // 2 + 1, N + 1))
        elif restart == 1:
            S = set(range(int(N * 0.4) + 1, N + 1))
        elif restart == 2:
            S = set(range(int(N * 0.45) + 1, N + 1))
        elif restart == 3:
            S = set(range(int(N * 0.35) + 1, N + 1))
        elif restart == 4:
            S = set(range(int(N * 0.3) + 1, N + 1))
        else:
            size = rng.randint(55, 75)
            S = set(rng.sample(range(1, N + 1), size))

        # Make valid
        for _ in range(200):
            if time.time() > deadline:
                break
            found = False
            for e in unique_edges:
                if e.issubset(S):
                    victim = rng.choice(list(e))
                    S.discard(victim)
                    found = True
                    break
            if not found:
                break

        # Greedy complete
        S = greedy_complete(S, rng)

        if len(S) > best_size and is_valid(S):
            best_size = len(S)
            best_set = set(S)

        # Simulated annealing with single flips and swaps
        T = 3.0
        cooling = 0.9995
        T_min = 0.01
        max_steps = 30000

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if T < T_min:
                break

            move = rng.random()
            accepted = False

            if move < 0.5:
                # Single flip
                if rng.random() < 0.5:
                    # Add
                    candidates = [x for x in range(1, N + 1) if x not in S]
                    if candidates:
                        x = rng.choice(candidates)
                        if check_add(S, x):
                            S.add(x)
                            accepted = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = set(S)
                        else:
                            # Reject (or accept with prob exp(-1/T))
                            if rng.random() < pow(2.718, -1.0 / T):
                                S.add(x)
                                accepted = True
                else:
                    # Remove
                    if S:
                        x = rng.choice(list(S))
                        S.discard(x)
                        accepted = True  # Always accept removals
            else:
                # Swap
                if S:
                    x = rng.choice(list(S))
                    candidates = [z for z in range(1, N + 1) if z not in S]
                    if candidates:
                        y = rng.choice(candidates)
                        S.discard(x)
                        if check_add(S, y):
                            S.add(y)
                            accepted = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = set(S)
                        else:
                            if rng.random() < pow(2.718, -1.0 / T):
                                S.add(y)
                                accepted = True
                            else:
                                S.add(x)

            T *= cooling

        # After SA, clean up violations and greedy complete
        for _ in range(100):
            if time.time() > deadline:
                break
            found = False
            for e in unique_edges:
                if e.issubset(S):
                    victim = rng.choice(list(e))
                    S.discard(victim)
                    found = True
                    break
            if not found:
                break

        S = greedy_complete(S, rng)

        if len(S) > best_size and is_valid(S):
            best_size = len(S)
            best_set = set(S)

    if not is_valid(best_set):
        best_set = set(range(N // 2 + 1, N + 1))
        best_size = len(best_set)

    return [int(x) for x in best_set]
