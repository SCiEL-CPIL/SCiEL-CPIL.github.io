"""Kissing number in dimension 11: exact maximum independent set on the full
norm-4 integer-vector conflict graph via bitset branch-and-bound (Tomita/BBMC),
warm-started by a greedy + simulated-annealing pass on the exact graph."""
import numpy as np
import itertools
import time
import sys

D = 11

# EVOLVE-BLOCK-START

def _build_candidates():
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    return v4_list, v1_list


def _build_conflict_bits(v4_list, v1_list):
    n4 = len(v4_list)
    n1 = len(v1_list)
    n = n4 + n1
    v4 = np.array(v4_list, dtype=np.int64)
    bits = [0] * n
    # W4-W4: conflict iff dot > 2
    for i in range(n4):
        for j in range(i + 1, n4):
            if int(v4[i] @ v4[j]) > 2:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    # W4-W1: dot in {-1,0,1}, never > 2 -> no conflicts
    # W1-W1: conflict only if same position, opposite sign
    for a in range(n1):
        pa = a // 2
        for c in range(a + 1, n1):
            if c // 2 == pa:
                bits[n4 + a] |= (1 << (n4 + c))
                bits[n4 + c] |= (1 << (n4 + a))
    return bits, n


def _greedy_color(R, N):
    """Tomita greedy coloring of induced subgraph on bitset R.
    Returns list of color-class bitsets (ascending color)."""
    color_classes = []
    remaining = R
    while remaining:
        C = 0
        cand = remaining
        while cand:
            lsb = cand & (-cand)
            v = lsb.bit_length() - 1
            C |= lsb
            cand ^= lsb
            cand &= ~N[v]
        color_classes.append(C)
        remaining &= ~C
    return color_classes


def _max_independent_set(bits, deadline, warm_bits):
    n = len(bits)
    N = bits
    best = [bin(warm_bits).count('1') if warm_bits else 0]
    best_set = [warm_bits]
    sys.setrecursionlimit(1000000)

    def rec(R, size, S):
        if time.time() > deadline:
            return
        if R == 0:
            if size > best[0]:
                best[0] = size
                best_set[0] = S
            return
        color_classes = _greedy_color(R, N)
        c = len(color_classes)
        if size + c <= best[0]:
            return
        for idx in range(c - 1, -1, -1):
            if time.time() > deadline:
                return
            if size + idx + 1 <= best[0]:
                break
            C = color_classes[idx]
            lsb = C & (-C)
            v = lsb.bit_length() - 1
            rec((R & ~C) & N[v], size + 1, S | lsb)
            R ^= lsb
    rec((1 << n) - 1, 0, 0)
    return best[0], best_set[0]


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    v4_list, v1_list = _build_candidates()
    bits, n = _build_conflict_bits(v4_list, v1_list)
    n4 = len(v4_list)

    rng = np.random.default_rng(42)

    # ---- Warm start: greedy min-degree independent set on exact graph ----
    deg = [bin(bits[i]).count('1') for i in range(n)]
    in_set = [False] * n
    alive = [True] * n
    S = 0
    while any(alive):
        if time.time() > deadline - 60:
            break
        best_v = -1
        best_d = 10 ** 9
        for i in range(n):
            if alive[i] and deg[i] < best_d:
                best_d = deg[i]
                best_v = i
        if best_v < 0:
            break
        in_set[best_v] = True
        S |= (1 << best_v)
        alive[best_v] = False
        for j in range(n):
            if alive[j] and (bits[best_v] >> j) & 1:
                alive[j] = False
                deg[j] -= 1

    # ---- Short SA on exact graph to strengthen incumbent ----
    count_conf = [0] * n
    for v in range(n):
        if in_set[v]:
            for w in range(n):
                if (bits[v] >> w) & 1:
                    count_conf[w] += 1

    T = 2.0
    sa_deadline = min(deadline - 70, start_time + 20)
    it = 0
    while time.time() < sa_deadline:
        not_in = [v for v in range(n) if not in_set[v]]
        if not not_in:
            break
        if rng.random() < 0.7:
            s = min(not_in, key=lambda v: count_conf[v])
            k = count_conf[s]
            delta = 1 - k
            if delta > 0 or (delta == 0 and rng.random() < 0.5) or (delta < 0 and rng.random() < np.exp(delta / T)):
                for u in range(n):
                    if in_set[u] and (bits[s] >> u) & 1:
                        in_set[u] = False
                        S &= ~(1 << u)
                        for w in range(n):
                            if (bits[u] >> w) & 1:
                                count_conf[w] -= 1
                in_set[s] = True
                S |= (1 << s)
                for w in range(n):
                    if (bits[s] >> w) & 1:
                        count_conf[w] += 1
        else:
            cand = not_in[rng.integers(len(not_in))]
            conflicts = [u for u in range(n) if in_set[u] and (bits[cand] >> u) & 1]
            if len(conflicts) >= 2:
                delta = 1 - len(conflicts)
                if delta >= 0 or rng.random() < np.exp(delta / T):
                    for u in conflicts:
                        in_set[u] = False
                        S &= ~(1 << u)
                        for w in range(n):
                            if (bits[u] >> w) & 1:
                                count_conf[w] -= 1
                    in_set[cand] = True
                    S |= (1 << cand)
                    for w in range(n):
                        if (bits[cand] >> w) & 1:
                            count_conf[w] += 1
        T = max(0.05, T * 0.9998)
        it += 1
        if it % 30000 == 0:
            T = max(T, 1.5)
    warm_bits = S

    # ---- Exact B&B (bitset Tomita) ----
    bb_deadline = deadline - 5
    best_size, best_bits = _max_independent_set(bits, bb_deadline, warm_bits)

    # ---- Build result ----
    final = []
    for v in range(n):
        if (best_bits >> v) & 1:
            final.append(v4_list[v] if v < n4 else v1_list[v - n4])
    result = np.array(final, dtype=np.int64)

    # ---- Verify min pairwise distance >= max norm ----
    if len(result) > 1:
        norms_sq = np.sum(result ** 2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        dots = result @ result.T
        offdiag = dots - np.diag(np.diag(dots))
        min_dist_sq = int(np.min(2 * max_norm_sq - offdiag))
        if min_dist_sq < max_norm_sq:
            final = []
            for v in range(n):
                if (warm_bits >> v) & 1:
                    final.append(v4_list[v] if v < n4 else v1_list[v - n4])
            result = np.array(final, dtype=np.int64)

    if len(result) == 0:
        result = np.array(v1_list, dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
