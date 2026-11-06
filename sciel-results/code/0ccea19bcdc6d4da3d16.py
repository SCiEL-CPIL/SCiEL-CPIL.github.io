import random
import time

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses a conflict-graph / maximum-independent-set heuristic combined with
    incremental DP verification and local search.
    """
    N = 512
    MAX_SIZE = 48
    MAX_SUM = N * MAX_SIZE  # 24576

    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    # Precompute a fast validity check using bitsets
    def check_validity(S_list):
        m = len(S_list)
        if m < 3:
            return True
        for idx, a in enumerate(S_list):
            others = [x for i, x in enumerate(S_list) if i != idx]
            m_o = len(others)
            dp = [0] * (m_o + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_o, 0, -1):
                    dp[k] |= (dp[k - 1] << x)
            for k in range(2, m_o + 1):
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def find_violation(S_list):
        m = len(S_list)
        if m < 3:
            return None, None
        for idx, a in enumerate(S_list):
            others = [x for i, x in enumerate(S_list) if i != idx]
            m_o = len(others)
            # k=2
            for i in range(m_o):
                for j in range(i + 1, m_o):
                    if others[i] + others[j] == 2 * a:
                        return a, [others[i], others[j]]
            # k=3
            others_set = set(others)
            for i in range(m_o):
                for j in range(i + 1, m_o):
                    rem = 3 * a - others[i] - others[j]
                    if rem > others[j] and rem in others_set and rem != others[i] and rem != others[j]:
                        return a, [others[i], others[j], rem]
            # k>=4 via DP
            dp = [0] * (m_o + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_o, 0, -1):
                    dp[k] |= (dp[k - 1] << x)
            for k in range(4, m_o + 1):
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return a, None
        return None, None

    # Conflict graph heuristic: an element x "conflicts" with a if x and some y in S
    # form a 3-term AP centered at a, i.e., x + y == 2a. More generally, we build
    # a graph where vertices are candidate elements and an edge means they cannot
    # coexist in a valid set (based on 3-term AP conflicts with some third element).
    # We use a greedy independent set on this graph as an initial selection, then
    # refine with DP-based local search.

    def build_conflict_graph(candidates):
        """
        Build a conflict graph among candidates where an edge (u, v) exists if
        there exists w in candidates such that {u, v, w} contains a 3-term AP
        (i.e., one is the mean of the other two). This is a necessary condition
        for invalidity but not sufficient; we use it as a heuristic.
        """
        cand_set = set(candidates)
        conflicts = {x: set() for x in candidates}
        for i, u in enumerate(candidates):
            for v in candidates[i + 1:]:
                # Check if u and v can be part of a 3-term AP with some w in candidates
                # w = 2u - v, w = 2v - u, or u = (v + w)/2 => w = 2u - v
                # We need w in candidates, w != u, w != v
                w1 = 2 * u - v
                w2 = 2 * v - u
                if (w1 in cand_set and w1 != u and w1 != v) or (w2 in cand_set and w2 != u and w2 != v):
                    conflicts[u].add(v)
                    conflicts[v].add(u)
        return conflicts

    def greedy_independent_set(conflicts, order):
        """Greedy: select elements in 'order' if they have no conflict with already selected."""
        selected = []
        selected_set = set()
        for x in order:
            if any(y in selected_set for y in conflicts[x]):
                continue
            selected.append(x)
            selected_set.add(x)
        return selected

    def local_search(S, max_iters=200):
        """
        Improve set S by trying to add elements or swap.
        """
        S = set(S)
        all_vals = set(range(1, N + 1))

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            # Try to add an element
            added = False
            candidates_to_add = [x for x in all_vals if x not in S]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add[:40]:
                new_S = list(S | {x})
                if check_validity(new_S):
                    S = S | {x}
                    added = True
                    break

            if added:
                continue

            # Try a swap
            if S:
                a = random.choice(list(S))
                candidates_to_swap = [x for x in all_vals if x not in S]
                random.shuffle(candidates_to_swap)
                for x in candidates_to_swap[:40]:
                    new_S = list((S - {a}) | {x})
                    if check_validity(new_S):
                        S = (S - {a}) | {x}
                        break

        return S

    def repair_set(S, max_steps=500):
        """Repair an invalid set by removing elements causing violations."""
        S = set(S)
        for _ in range(max_steps):
            if time.time() > deadline:
                break

            S_list = list(S)
            if len(S_list) < 3:
                break

            a, T = find_violation(S_list)
            if a is None:
                break

            r = random.random()
            if T is not None and len(T) > 0 and r < 0.5:
                victim = random.choice(T)
                S.discard(victim)
            elif r < 0.8:
                S.discard(a)
            else:
                others = [x for x in S if x != a]
                if others:
                    S.discard(random.choice(others))
                else:
                    S.discard(a)

        return S

    best_set = set()
    best_size = 0
    candidates = list(range(1, N + 1))

    # Build conflict graph once
    conflicts = build_conflict_graph(candidates)

    # Strategy 1: Greedy independent set on conflict graph with various orderings
    orderings = []

    # Large-first
    orderings.append(list(range(N, 0, -1)))
    # Small-first
    orderings.append(list(range(1, N + 1)))
    # Random
    for _ in range(5):
        o = candidates[:]
        random.shuffle(o)
        orderings.append(o)
    # Mixed
    mixed = []
    for i in range(N // 2):
        mixed.append(N - i)
        mixed.append(i + 1)
    orderings.append(mixed)
    # Powers of 2 first
    powers = [2 ** i for i in range(1, 10) if 2 ** i <= N]
    rest = [x for x in candidates if x not in set(powers)]
    random.shuffle(rest)
    orderings.append(powers + rest)

    for order in orderings:
        if time.time() > deadline:
            break

        S_list = greedy_independent_set(conflicts, order)
        S = set(S_list)

        # Verify
        if S:
            if check_validity(list(S)):
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()

        # Local search to improve
        if len(S) < MAX_SIZE:
            S = local_search(S, max_iters=150)
            S = set(S)
            if S:
                if check_validity(list(S)):
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()

        if best_size == MAX_SIZE:
            break

    # If we have a good set, try more local search
    if best_set and len(best_set) < MAX_SIZE:
        best_set = local_search(best_set, max_iters=300)
        best_set = set(best_set)
        if best_set:
            if check_validity(list(best_set)):
                best_size = len(best_set)
            else:
                best_set = set()
                best_size = 0

    # If still not at MAX_SIZE, try repair-based approach
    if best_size < MAX_SIZE and time.time() < deadline - 5:
        for restart in range(15):
            if time.time() > deadline:
                break

            S = set(random.sample(candidates, MAX_SIZE))
            S = repair_set(S, max_steps=400)

            if S:
                if check_validity(list(S)):
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()

                    if len(S) < MAX_SIZE:
                        S = local_search(S, max_iters=100)
                        S = set(S)
                        if S:
                            if check_validity(list(S)):
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_set = S.copy()

            if best_size == MAX_SIZE:
                break

    # Final verification
    if best_set:
        if check_validity(list(best_set)):
            return list(best_set)

    # Fallback
    return [1, 3, 7, 15, 31, 63, 127, 255, 511]
