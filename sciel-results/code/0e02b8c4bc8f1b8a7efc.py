"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    deadline = time.time() + 72  # safety margin under 90s kill
    N = 128

    # Precompute GCD matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad_pairs: bad_pairs[v] = list of (a,b) such that (v,a,b) is a bad triple
    bad_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd(a+1, b+1)
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd(a+1, c+1) == ga and gcd(b+1, c+1) == ga:
                    bad_pairs[c].append((a,b))
                    bad_pairs[a].append((b,c))
                    bad_pairs[b].append((a,c))

    # Precompute lookup: bad_pair_lookup[v][u] = list of w such that (u,w) in bad_pairs[v]
    bad_pair_lookup = [{} for _ in range(N)]
    for v in range(N):
        for (a, b) in bad_pairs[v]:
            if a not in bad_pair_lookup[v]:
                bad_pair_lookup[v][a] = []
            bad_pair_lookup[v][a].append(b)
            if b not in bad_pair_lookup[v]:
                bad_pair_lookup[v][b] = []
            bad_pair_lookup[v][b].append(a)

    # Precompute degree (number of bad pairs involving v)
    degree = [len(bad_pairs[v]) for v in range(N)]

    # Precompute bad_partner_counts: for each v, a list of u such that (u, w) in bad_pairs[v] for some w
    # i.e., bad_partner[v] = set of u where there exists w with (u,w) in bad_pairs[v]
    bad_partner = [set() for _ in range(N)]
    for v in range(N):
        for (a, b) in bad_pairs[v]:
            bad_partner[v].add(a)
            bad_partner[v].add(b)

    def compute_blocked(selected_set):
        """Compute blocked counts for all vertices not in selected_set."""
        blocked = [0] * N
        for v in selected_set:
            for u in range(N):
                if u == v or u in selected_set:
                    continue
                if v in bad_pair_lookup[u]:
                    for w in bad_pair_lookup[u][v]:
                        if w in selected_set:
                            blocked[u] += 1
        return blocked

    def fast_local_improve(subset):
        """
        Fast local improvement using a partner-count array.
        partner_count[v] = number of selected vertices u such that (u, w) is a bad pair for v for some selected w.
        v is addable iff partner_count[v] == 0.
        When we add vertex x, for each v not in s, partner_count[v] += |{w in s : (x, w) in bad_pairs[v]}|
        But more precisely: partner_count[v] counts the number of selected u where there exists selected w with (u,w) in bad_pairs[v].
        Actually, blocked[v] = number of pairs (a,b) in s such that (a,b) in bad_pairs[v].
        So partner_count is not quite right. Let me use a different approach:
        
        For each v, define conf[v] = number of selected pairs (a,b) with (a,b) in bad_pairs[v].
        v is addable iff conf[v] == 0.
        When adding x: for each v not in s, conf[v] += number of a in s such that (x,a) in bad_pairs[v].
        """
        s = set(subset)
        # Compute initial conf
        conf = [0] * N
        for v in s:
            for u in range(N):
                if u in s:
                    continue
                if v in bad_pair_lookup[u]:
                    for w in bad_pair_lookup[u][v]:
                        if w in s:
                            conf[u] += 1
        
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in s:
                    continue
                if conf[v] == 0:
                    s.add(v)
                    improved = True
                    # Update conf for all u not in s
                    for u in range(N):
                        if u in s:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s:
                                    conf[u] += 1
        return list(s)

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid, using blocked counts."""
        selected = []
        selected_set = set()
        blocked = [0] * N
        for v in order:
            if v in selected_set:
                continue
            if blocked[v] == 0:
                selected.append(v)
                selected_set.add(v)
                for u in range(N):
                    if u == v or u in selected_set:
                        continue
                    if v in bad_pair_lookup[u]:
                        for w in bad_pair_lookup[u][v]:
                            if w in selected_set:
                                blocked[u] += 1
        return selected

    def swap_improve(subset, rng, max_swaps=20):
        """Try swap: remove one element, then greedily re-add. Limit swaps for speed."""
        s = set(subset)
        if not s:
            return list(s)
        best_size = len(s)
        best_set = None
        elements = list(s)
        rng.shuffle(elements)
        for rem in elements[:max_swaps]:
            if time.time() > deadline:
                break
            s2 = s - {rem}
            blocked2 = compute_blocked(s2)
            for v in range(N):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = s2
        if best_set is not None:
            return list(best_set)
        return list(s)

    def two_swap_improve(subset, rng, max_pairs=20):
        """Try 2-swap: remove two elements, then greedily re-add."""
        s = set(subset)
        if len(s) < 2:
            return list(s)
        best_size = len(s)
        best_set = None
        elements = list(s)
        rng.shuffle(elements)
        n = len(elements)
        for i in range(min(n, max_pairs)):
            if time.time() > deadline:
                break
            for j in range(i+1, min(n, i + max_pairs)):
                if time.time() > deadline:
                    break
                s2 = s - {elements[i], elements[j]}
                blocked2 = compute_blocked(s2)
                for v in range(N):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                if len(s2) > best_size:
                    best_size = len(s2)
                    best_set = s2
        if best_set is not None:
            return list(best_set)
        return list(s)

    def targeted_polish(subset, deadline):
        """
        Try targeted removals: remove pairs of vertices that are 'conflicting'
        in the sense that removing both might allow adding more than 2 others.
        """
        s = set(subset)
        blocked = compute_blocked(s)
        cur_size = len(s)
        best_size = cur_size
        best_s = set(s)
        
        # Find vertices that are blocked
        blocked_vertices = [v for v in range(N) if v not in s and blocked[v] > 0]
        
        # For each blocked vertex, find which pairs in s are blocking it
        # Try removing those pairs and see if we gain
        rng_local = random.Random(77)
        rng_local.shuffle(blocked_vertices)
        
        for bv in blocked_vertices[:30]:  # limit for speed
            if time.time() > deadline:
                break
            # Find which pairs in s block bv
            blocking_pairs = []
            for (a, b) in bad_pairs[bv]:
                if a in s and b in s:
                    blocking_pairs.append((a, b))
            
            if not blocking_pairs:
                continue
            
            # Try removing one element from each blocking pair
            for (a, b) in blocking_pairs[:5]:
                if time.time() > deadline:
                    break
                # Try removing a
                s2 = s - {a}
                blocked2 = compute_blocked(s2)
                for v in range(N):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                if len(s2) > best_size:
                    best_size = len(s2)
                    best_s = s2
                # Try removing b
                s2 = s - {b}
                blocked2 = compute_blocked(s2)
                for v in range(N):
                    if v in s2:
                        continue
                    if blocked2[v] == 0:
                        s2.add(v)
                        for u in range(N):
                            if u == v or u in s2:
                                continue
                            if v in bad_pair_lookup[u]:
                                for w in bad_pair_lookup[u][v]:
                                    if w in s2:
                                        blocked2[u] += 1
                if len(s2) > best_size:
                    best_size = len(s2)
                    best_s = s2
        
        return list(best_s), best_size

    # Main search
    random.seed(42)
    rng = random.Random(12345)
    best = []
    best_size = 0

    # Phase 1: Multiple greedy + fast local_improve
    for trial in range(300):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = fast_local_improve(candidate)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    # Sorted orderings
    for order in [list(range(N)), list(range(N-1, -1, -1)),
                  sorted(range(N), key=lambda x: degree[x]),
                  sorted(range(N), key=lambda x: -degree[x])]:
        if time.time() > deadline:
            break
        candidate = greedy_construction(order)
        candidate = fast_local_improve(candidate)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    # Phase 2: Swap improvement
    for trial in range(30):
        if time.time() > deadline:
            break
        if not best:
            break
        improved = swap_improve(best, rng, max_swaps=15)
        if len(improved) > best_size:
            best_size = len(improved)
            best = improved
        else:
            break

    # Phase 2b: 2-swap improvement
    for trial in range(15):
        if time.time() > deadline:
            break
        if not best:
            break
        improved = two_swap_improve(best, rng, max_pairs=10)
        if len(improved) > best_size:
            best_size = len(improved)
            best = improved
        else:
            break

    # Phase 3: Targeted polish
    if best and time.time() < deadline - 3:
        improved, new_size = targeted_polish(best, deadline - 1)
        if new_size > best_size:
            best_size = new_size
            best = improved

    # Phase 4: More randomized restarts with perturbation
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = fast_local_improve(candidate)
        for _ in range(2):
            if time.time() > deadline:
                break
            if not candidate:
                break
            candidate = swap_improve(candidate, rng, max_swaps=10)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    if not best:
        best = greedy_construction(list(range(N)))
        best = fast_local_improve(best)

    return [v + 1 for v in sorted(best)]
# EVOLVE-BLOCK-END
