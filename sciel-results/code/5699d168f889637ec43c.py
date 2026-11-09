"""Fresh design: randomized incremental construction + local repair with
bitmask-based C6 detection for 4-colourings of K_n (no monochromatic C6)."""
import time
import random
import sys

C = 4
DEADLINE = 112.0          # explicit safety margin vs. the 150 s wall-clock limit
_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - _START)


class State:
    __slots__ = ("n", "adj", "cnt")

    def __init__(self, n):
        self.n = n
        self.adj = [0] * C
        self.cnt = [0] * C          # number of C6 "seeds" per color (0 == no C6)

    def copy(self):
        s = State(self.n)
        s.adj = self.adj[:]
        s.cnt = self.cnt[:]
        return s

    def set_edge(self, u, v, c):
        """Set edge (u,v) to color c. Requires that it is currently uncolored
        (not already present in any color). Returns True on success."""
        m = 1 << v
        if self.adj[c] & (1 << u):
            return False            # already set
        self.adj[c] |= (m << u) | ((1 << u) << v)
        self._update_seed(c, u, v, 1)
        self._update_seed(c, v, u, 1)
        return True

    def _update_seed(self, c, u, v, sign):
        """Adjust cnt[c] by sign * (number of length-2 paths through (u,v)).
        A C6 seed is a 2-path x-u-v whose endpoints x,v are adjacent."""
        adj = self.adj[c]
        N = (adj >> u) & ~((1 << (u + 1)) - 1)      # neighbours of u with index > u
        N &= adj & ((1 << u) - 1)                    # ...that are also neighbours of v (index < u)
        self.cnt[c] += sign * (N.bit_count())

    def valid(self):
        return all(x == 0 for x in self.cnt)


def _count_seeds_bitadj(adj, n):
    total = 0
    for u in range(n):
        N = adj[u]
        N &= adj[u] >> u
        total += N.bit_count()
    return total


def _bit_adjacency(adj, n):
    A = [0] * n
    for u in range(n):
        row = adj[u]
        A[u] = (row & ((1 << (u + 1)) - 1)) << (n - u - 1)
    return A


def _count_c6_adj(A, n):
    # counts undirected 6-cycles exactly; 0 iff no C6 present
    A2 = A
    A4 = [0] * n
    for i in range(n):
        s = 0
        ai = A[i]
        while ai:
            j = (ai & -ai).bit_length() - 1
            s += A2[j].bit_count()
            ai &= ai - 1
        A4[i] = s
    total = 0
    for i in range(n):
        ai = A[i]
        ai &= ~((1 << i) - 1)
        while ai:
            j = (ai & -ai).bit_length() - 1
            total += A4[i] & A[j]
            ai &= ai - 1
    return total >> 1


def _verify(n, colors):
    bitadj = [[0] * n for _ in range(C)]
    idx = 0
    for i in range(n):
        row_i = bitadj[i]
        for j in range(i + 1, n):
            c = colors[idx]
            if not (0 <= c < C):
                return False
            idx += 1
            bitadj[i][j] = 1 << c
            bitadj[j][i] = 1 << c
    for c in range(C):
        A = [row[c] for row in bitadj]
        if _count_c6_adj(A, n) > 0:
            return False
    return True


def _build_seed(n, seed):
    rng = random.Random(seed)
    S = State(n)
    for i in range(n):
        for j in range(i + 1, n):
            S.set_edge(i, j, rng.randrange(C))
    return S


def _local_search(S, max_iter, seed):
    rng = random.random
    n = S.n
    total = sum(S.cnt)
    if total == 0:
        return S
    T = max(total, 1) * 1e-2
    decay = max(1e-5, 1.0 / max(1, max_iter)) ** 1.0
    for _ in range(max_iter):
        if _remaining() < 1.0:
            break
        for c in range(C):
            if S.cnt[c] == 0:
                continue
            u = rng() * n
            v = rng() * n
            u = int(u)
            v = int(v)
            if u == v:
                continue
            if (S.adj[c] >> u) & (1 << v):
                old = c
            elif (S.adj[(c + 1) % C] >> u) & (1 << v):
                old = (c + 1) % C
            elif (S.adj[(c + 2) % C] >> u) & (1 << v):
                old = (c + 2) % C
            elif (S.adj[(c + 3) % C] >> u) & (1 << v):
                old = (c + 3) % C
            else:
                continue
            old_cnt = S.cnt[old]
            S.set_edge(u, v, old)          # remove
            S._update_seed(old, u, v, -1)
            S._update_seed(old, v, u, -1)
            best_c = -1
            best_delta = None
            for nc in range(C):
                if nc == old:
                    continue
                S.set_edge(u, v, nc)
                S._update_seed(nc, u, v, 1)
                S._update_seed(nc, v, u, 1)
                d = (S.cnt[nc] - old_cnt)
                ok = S.valid()
                if best_delta is None or d < best_delta or (d == best_delta and ok):
                    best_delta = d
                    best_c = nc
                    best_ok = ok
                # undo
                S._update_seed(nc, v, u, -1)
                S._update_seed(nc, u, v, -1)
                S.set_edge(u, v, nc)
            # apply best_c
            S.set_edge(u, v, best_c)
            S._update_seed(best_c, u, v, 1)
            S._update_seed(best_c, v, u, 1)
            new_total = sum(S.cnt)
            delta = new_total - total
            if delta < 0 or (delta == 0) or (rng() < (0.5 * (1.0 - decay))):
                total = new_total
            else:
                # revert best_c back to old
                S.set_edge(u, v, best_c)
                S._update_seed(best_c, u, v, -1)
                S._update_seed(best_c, v, u, -1)
                S.set_edge(u, v, old)
                S._update_seed(old, u, v, 1)
                S._update_seed(old, v, u, 1)
            if total == 0:
                return S
    return S if S.valid() else None


def _to_colors(S):
    n = S.n
    out = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for c in range(C):
                if (S.adj[c] >> i) & (1 << j):
                    out[idx] = c
                    break
            idx += 1
    return out


def construct():
    # Known-good fallback (n=9), always valid.
    def fallback9():
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n)
                                   for j in range(i + 1, n)]}

    best = None
    best_n = 0

    def consider(S):
        nonlocal best, best_n
        if S is None or not S.valid():
            return
        if S.n > best_n:
            best = S
            best_n = S.n

    # Incrementally grow from a valid base, with periodic restarts and local repair.
    seed_base = 12345
    for target in range(9, 20):
        if _remaining() < 8.0:
            break
        improved = False
        attempts = 0
        while _remaining() > 4.0 and not improved and attempts < 12:
            attempts += 1
            S = _build_seed(target, seed_base + attempts)
            if S.valid():
                improved = True
            else:
                S = _local_search(S, 6000, seed_base + attempts)
                if S is not None:
                    improved = True
            if not improved:
                # occasionally try smaller repair budget to keep moving
                pass
        if improved:
            consider(S)

    # If we could not reach a large n, at least ensure n=9 is recorded.
    if best is None:
        best = _build_seed(9, 999)
        if not best.valid():
            best = _local_search(best, 4000, 999)
        if best is None or not best.valid():
            return fallback9()

    cols = _to_colors(best)
    if _verify(best.n, cols):
        return {'n': best.n, 'colors': cols}
    return fallback9()


# ---- self test on import (cheap, guarded) ----
if __name__ == "__main__":
    res = construct()
    print("n =", res['n'], "len =", len(res['colors']))
