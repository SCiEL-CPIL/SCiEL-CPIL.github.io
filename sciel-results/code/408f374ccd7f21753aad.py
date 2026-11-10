# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    t0 = time.time()
    deadline = t0 + 590.0  # safety margin before 630s wall-clock kill

    # Forbidden differences F = {x^2 mod 205 : x=1..204} \ {0}
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Symmetrize: D = F ∪ (-F). Since -1 is a square mod 205, D == F,
    # but compute explicitly to be safe.
    D = set()
    for d in forbidden:
        D.add(d)
        D.add((-d) % m)
    D.discard(0)
    D = list(D)

    # adj[x] bitmask: bit s set iff (x - s) % m in D  (i.e. x conflicts with s)
    adj = [0] * m
    for x in range(m):
        mask = 0
        for d in D:
            s = (x - d) % m
            mask |= (1 << s)
        adj[x] = mask

    # conflict_list[x]: list of neighbors of x (for fast 2-2 checks)
    conflict_list = [[] for _ in range(m)]
    for x in range(m):
        lst = []
        for s in range(m):
            if (adj[x] >> s) & 1:
                lst.append(s)
        conflict_list[x] = lst

    all_mask = (1 << m) - 1

    def is_addable(M_S, x):
        return (M_S & adj[x]) == 0

    def greedy(order, M_S=0, S=None):
        if S is None:
            S = []
        for x in order:
            if not ((M_S >> x) & 1) and (M_S & adj[x]) == 0:
                S.append(x)
                M_S |= (1 << x)
        return S, M_S

    def greedy_most_free(order):
        # Greedy but at each step pick the candidate with fewest conflicts
        # with the current set (tie broken by original order).
        S = []
        M_S = 0
        remaining = set(order)
        while remaining:
            best_x = None
            best_conf = None
            # iterate in original order for deterministic tie-break
            for x in order:
                if x in remaining:
                    c = (M_S & adj[x]).bit_count()
                    if best_conf is None or c < best_conf:
                        best_conf = c
                        best_x = x
                        if c == 0:
                            break
            if best_x is None:
                break
            if best_conf == 0:
                S.append(best_x)
                M_S |= (1 << best_x)
                remaining.discard(best_x)
            else:
                # No zero-conflict candidate; take the least-conflict one
                # (still valid since we only add if it doesn't conflict
                #  with existing set? No - best_conf>0 means it conflicts.
                # So we cannot add it. Stop.
                break
        return S, M_S

    def add_only_polish(M_S, S):
        improved = True
        while improved:
            improved = False
            for x in range(m):
                if not ((M_S >> x) & 1) and (M_S & adj[x]) == 0:
                    S.append(x)
                    M_S |= (1 << x)
                    improved = True
                    break
        return S, M_S

    def one_one_swap(M_S, S):
        # Try to find a valid 1-1 swap; return (found, new_M_S, new_S, u, v)
        n = len(S)
        # shuffle u order
        u_order = list(range(n))
        random.shuffle(u_order)
        for ui in u_order:
            u = S[ui]
            M_temp = M_S ^ (1 << u)
            # candidates: x not in M_temp and (M_temp & adj[x]) == 0
            # iterate x in random order
            v_candidates = list(range(m))
            random.shuffle(v_candidates)
            for v in v_candidates:
                if (M_temp >> v) & 1:
                    continue
                if (M_temp & adj[v]) == 0:
                    # valid swap
                    S2 = S[:]
                    S2[ui] = v
                    return True, M_temp | (1 << v), S2, u, v
        return False, M_S, S, None, None

    def two_two_kick(M_S, S):
        # Try a 2-2 swap: remove u1,u2, add v1,v2.
        # Returns (found, new_M_S, new_S) or (False, M_S, S)
        n = len(S)
        if n < 2:
            return False, M_S, S
        # pick a random pair (u1, u2)
        u1_idx, u2_idx = random.sample(range(n), 2)
        u1 = S[u1_idx]
        u2 = S[u2_idx]
        M_base = M_S ^ (1 << u1) ^ (1 << u2)
        # candidate pool: not in M_base
        # We'll try random v1, v2
        pool = [x for x in range(m) if not ((M_base >> x) & 1)]
        random.shuffle(pool)
        # Limit pool size for speed
        pool = pool[:120]
        for i in range(len(pool)):
            v1 = pool[i]
            if (M_base & adj[v1]) != 0:
                continue
            M1 = M_base | (1 << v1)
            for j in range(len(pool)):
                if i == j:
                    continue
                v2 = pool[j]
                if (M1 >> v2) & 1:
                    continue
                if (M1 & adj[v2]) == 0:
                    # valid
                    S2 = S[:]
                    S2[u1_idx] = v1
                    S2[u2_idx] = v2
                    return True, M1 | (1 << v2), S2
        return False, M_S, S

    def local_search(S, M_S):
        # Add-only polish
        S, M_S = add_only_polish(M_S, S)
        improved = True
        stall = 0
        while improved:
            improved = False
            # 1-1 swap
            found, M_S2, S2, u, v = one_one_swap(M_S, S)
            if found:
                S, M_S = S2, M_S2
                S, M_S = add_only_polish(M_S, S)
                improved = True
                stall = 0
                continue
            # 2-2 kick
            found, M_S2, S2 = two_two_kick(M_S, S)
            if found:
                S, M_S = S2, M_S2
                S, M_S = add_only_polish(M_S, S)
                improved = True
                stall = 0
                continue
            break
        return S, M_S

    best_set = []
    best_size = 0

    # Initial: a few deterministic + random starts
    starts = []
    # natural order
    starts.append(list(range(m)))
    # reverse
    starts.append(list(range(m - 1, -1, -1)))
    # random
    for _ in range(3):
        o = list(range(m))
        random.shuffle(o)
        starts.append(o)

    for order in starts:
        S, M_S = greedy(order)
        S, M_S = local_search(S, M_S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        if time.time() > deadline - 60:
            break

    # Main loop: random restarts with local search
    while time.time() < deadline:
        order = list(range(m))
        random.shuffle(order)
        S, M_S = greedy(order)
        # Occasionally use most-free greedy
        if random.random() < 0.3:
            S2, M_S2 = greedy_most_free(order)
            if len(S2) > len(S):
                S, M_S = S2, M_S2
        S, M_S = local_search(S, M_S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            # If we get a large set, we can afford to be more aggressive
            # but keep the loop going to try to improve further.

    if not best_set:
        best_set = [0]

    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
