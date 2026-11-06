import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Base cycle edges
    base_edges = [(i, (i + 1) % N) for i in range(N)]

    required = set(range(3, N + 1))

    # Chord s (0,s) for s in 2..38
    # Single chord covers lengths: s+1 and N-s+1
    # Pair of chords a < b covers length: b - a + 2

    chord_singles = {}
    for s in range(2, N - 1):
        lengths = set()
        l1 = s + 1
        if 3 <= l1 <= N:
            lengths.add(l1)
        l2 = N - s + 1
        if 3 <= l2 <= N:
            lengths.add(l2)
        chord_singles[s] = lengths

    # Precompute pair lengths for all pairs
    pair_len = {}
    for a in range(2, N - 1):
        for b in range(a + 1, N - 1):
            l = b - a + 2
            if 3 <= l <= N:
                pair_len[(a, b)] = l

    def compute_covered(S):
        """Compute covered lengths given chord set S (frozenset or set)."""
        cov = {N}
        S_list = sorted(S)
        for s in S_list:
            cov.update(chord_singles[s])
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                key = (a, b)
                if key in pair_len:
                    cov.add(pair_len[key])
        return cov

    candidates = list(range(2, N - 1))  # 2..38

    # Try to find a small set of chords covering all lengths
    # Use a more aggressive search: start from known good sets, then optimize

    best_S = None
    best_size = float('inf')

    # Known good starting points
    starting_sets = [
        set([2, 5, 8, 11, 14, 17, 20, 23]),
        set([2, 4, 6, 8, 10, 12, 14, 16, 18]),
        set([2, 3, 5, 7, 9, 11, 13, 15]),
        set([2, 5, 9, 13, 17, 21, 25]),
        set([2, 4, 7, 10, 13, 16, 19, 22]),
    ]

    # Add random starting sets
    rng = random.Random(42)
    for _ in range(20):
        size = rng.randint(6, 12)
        s = set(rng.sample(candidates, size))
        starting_sets.append(s)

    # Greedy improvement from each starting set
    for S0 in starting_sets:
        if time.time() > deadline - 20.0:
            break

        S = set(S0)
        covered = compute_covered(S)

        # If not covering all, try to add chords
        while not required.issubset(covered):
            best_gain = -1
            best_s = None
            for s in candidates:
                if s in S:
                    continue
                new_cov = compute_covered(S | {s})
                gain = len(new_cov & required - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s is None or best_gain <= 0:
                break
            S.add(best_s)
            covered = compute_covered(S)

        if required.issubset(covered) and len(S) < best_size:
            best_size = len(S)
            best_S = S.copy()
            if best_size <= 7:
                break

    # If we found a good set, try local search to minimize
    if best_S is not None:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            S_list = sorted(best_S)

            # Try removals
            for i in range(len(S_list)):
                if time.time() > deadline - 10.0:
                    break
                trial_S = best_S - {S_list[i]}
                if required.issubset(compute_covered(trial_S)):
                    best_S = trial_S
                    best_size = len(best_S)
                    improved = True
                    break

            # Try swaps: remove one, add another
            if not improved and len(S_list) >= 2:
                for i in range(len(S_list)):
                    if time.time() > deadline - 10.0:
                        break
                    for s in candidates:
                        if s in best_S:
                            continue
                        trial_S = (best_S - {S_list[i]}) | {s}
                        if len(trial_S) == len(best_S) and required.issubset(compute_covered(trial_S)):
                            best_S = trial_S
                            improved = True
                            break
                    if improved:
                        break

    # If we still don't have a good set, use a fallback
    if best_S is None:
        best_S = set(range(2, 39, 3))

    S = best_S
    S_sorted = sorted(S)

    # Build the edge list
    edges = list(base_edges)
    for s in S:
        edges.append((0, s))

    # Remove duplicate edges
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges

    # Build cycles for each length 3..N
    cycles_by_len = {}
    cycles_by_len[N] = list(range(N))

    # From single chords
    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))

        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))

    # From pairs of chords
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            l = b - a + 2
            if 3 <= l <= N and l not in cycles_by_len:
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))

    # Check if all lengths are covered
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            # Fallback: use a valid cycle construction
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L <= 39:
                k = 41 - L
                cycles_by_len[L] = [0] + list(range(39, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))

    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]

    # Verify all cycles are valid (edges exist)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    valid = True
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to a safe construction with more chords
        S_fallback = set(range(2, 21))
        edges_fallback = list(base_edges)
        for s in S_fallback:
            edges_fallback.append((0, s))
        unique_edges_fallback = []
        seen_fallback = set()
        for u, v in edges_fallback:
            if u > v:
                u, v = v, u
            if (u, v) not in seen_fallback:
                seen_fallback.add((u, v))
                unique_edges_fallback.append((u, v))

        cycles_fallback = []
        for L in range(3, N + 1):
            if L <= 21:
                cycles_fallback.append(list(range(L)))
            elif L <= 39:
                k = 41 - L
                cycles_fallback.append([0] + list(range(39, k - 1, -1)))
            else:
                cycles_fallback.append(list(range(N)))

        return {'edges': unique_edges_fallback, 'cycles': cycles_fallback}

    return {'edges': edges, 'cycles': final_cycles}
