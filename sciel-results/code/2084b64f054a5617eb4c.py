import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin under 90s kill

    n = 256

    # ---- Precompute 3-APs and 4-APs (0-indexed) and partner lists ----
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

    deg3 = [len(partners3[v]) for v in range(n)]
    deg4 = [len(partners4[v]) for v in range(n)]

    # ---- Fast scoring / delta helpers (bitmask A, 0-indexed bits) ----
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
        gain = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                gain += 1
        return gain

    def loss_remove(A, v):
        loss = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                loss += 1
        return loss

    def can_add(A, v):
        for u, w, x in partners4[v]:
            if ((A >> u) & 1) and ((A >> w) & 1) and ((A >> x) & 1):
                return False
        return True

    def greedy_construct(order):
        A = 0
        for v in order:
            if can_add(A, v):
                A |= (1 << v)
        return A

    def hill_climb(A, time_limit):
        current_score = count_3aps_mask(A)
        best_score = current_score
        best_A = A

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_move = None

            # Adds
            for v in range(n):
                if (A >> v) & 1:
                    continue
                if not can_add(A, v):
                    continue
                g = gain_add(A, v)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', v, -1)

            # Removes
            for v in range(n):
                if not ((A >> v) & 1):
                    continue
                l = loss_remove(A, v)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', v, -1)

            # Swaps
            for u in range(n):
                if not ((A >> u) & 1):
                    continue
                for v in range(n):
                    if (A >> v) & 1 or u == v:
                        continue
                    valid = True
                    for w, x, y in partners4[v]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and ((A >> y) & 1):
                            if w != u and x != u and y != u:
                                valid = False
                                break
                    if not valid:
                        continue

                    loss_u = 0
                    for w, x in partners3[u]:
                        if ((A >> w) & 1) and ((A >> x) & 1) and w != v and x != v:
                            loss_u += 1
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
                else:
                    A &= ~(1 << u)
                    A |= (1 << v)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
                improved = True

        return best_A, best_score

    def sa_polish(A, time_limit):
        rng = random.Random(12345)
        A_set = set(v for v in range(n) if (A >> v) & 1)
        A_mask = A
        current_score = count_3aps_mask(A_mask)
        best_score = current_score
        best_mask = A_mask

        steps = 60000
        T_start = 4.0
        T_end = 0.05
        for i in range(steps):
            if time.time() > time_limit:
                break
            if not A_set:
                break
            u = rng.choice(list(A_set))
            while True:
                v = rng.randint(0, n - 1)
                if v not in A_set:
                    break

            bit_v = 1 << v
            bit_u = 1 << u
            valid = True
            for w, x, y in partners4[v]:
                if ((A_mask >> w) & 1) and ((A_mask >> x) & 1) and ((A_mask >> y) & 1):
                    if w != u and x != u and y != u:
                        valid = False
                        break
            if not valid:
                continue

            loss_u = 0
            for w, x in partners3[u]:
                if ((A_mask >> w) & 1) and ((A_mask >> x) & 1) and w != v and x != v:
                    loss_u += 1
            gain_v = 0
            for w, x in partners3[v]:
                if ((A_mask >> w) & 1) and ((A_mask >> x) & 1) and w != u and x != u:
                    gain_v += 1
            delta = gain_v - loss_u

            if delta > 0:
                accept = True
            else:
                T = T_start * (T_end / T_start) ** (i / steps)
                if rng.random() < math.exp(delta / T):
                    accept = True
                else:
                    accept = False

            if accept:
                A_set.remove(u)
                A_set.add(v)
                A_mask ^= bit_u
                A_mask |= bit_v
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask

        return best_mask, best_score

    # ---- Build diverse initial orderings ----
    orderings = []
    orderings.append(sorted(range(n), key=lambda v: (-deg3[v], v)))
    orderings.append(sorted(range(n), key=lambda v: (-(deg3[v] - deg4[v]), v)))
    orderings.append(list(range(n)))
    orderings.append(list(range(n - 1, -1, -1)))
    for s in range(6):
        rng = random.Random(100 + s)
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)

    # 4-AP-avoidance ordering: prefer elements that complete many 3-APs but
    # few 4-APs, with a tie-break that delays dense 4-AP hubs.
    orderings.append(sorted(range(n), key=lambda v: (deg4[v], -deg3[v])))

    best_A_global = 0
    best_score_global = 0

    for i, order in enumerate(orderings):
        if time.time() > deadline - 5.0:
            break
        A = greedy_construct(order)
        remaining = deadline - 5.0 - time.time()
        budget = max(1.0, remaining / (len(orderings) - i))
        A, score = hill_climb(A, time.time() + budget)
        # short SA polish on the local optimum
        A, score = sa_polish(A, time.time() + min(1.5, max(0.5, deadline - 5.0 - time.time())))
        if score > best_score_global:
            best_score_global = score
            best_A_global = A

    # ---- Perturbation restarts from the global best ----
    if time.time() < deadline - 8.0:
        rng = random.Random(777)
        for _ in range(6):
            if time.time() > deadline - 8.0:
                break
            A = best_A_global
            bits = [v for v in range(n) if (A >> v) & 1]
            for _ in range(4):
                if not bits:
                    break
                v = rng.choice(bits)
                A &= ~(1 << v)
                bits.remove(v)
            for v in range(n):
                if not ((A >> v) & 1) and can_add(A, v):
                    A |= (1 << v)
            remaining = deadline - 8.0 - time.time()
            A, score = hill_climb(A, time.time() + max(0.5, remaining / 2))
            A, score = sa_polish(A, time.time() + min(1.0, max(0.3, deadline - 8.0 - time.time())))
            if score > best_score_global:
                best_score_global = score
                best_A_global = A

    # ---- Final verification / fallback ----
    if count_4aps_mask(best_A_global) != 0 or best_A_global == 0:
        rng = random.Random(0)
        A = 0
        for v in range(n):
            if can_add(A, v):
                A |= (1 << v)
        best_A_global = A
        best_score_global = count_3aps_mask(best_A_global)

    result = sorted(v + 1 for v in range(n) if (best_A_global >> v) & 1)
    if not result:
        result = [1]
    return result
