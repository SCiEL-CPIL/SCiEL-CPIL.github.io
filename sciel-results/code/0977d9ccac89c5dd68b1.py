import time
from math import gcd
from functools import reduce
from collections import deque

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 80.0

    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)  # 30030

    def hopcroft_karp(candidates, W):
        """Bipartite matching: divisors (left, n nodes) to positions (right, W nodes).
        candidates[i] = list of positions valid for divisor i.
        Returns list of positions (one per divisor) or None."""
        n_left = len(candidates)
        pair_u = [-1] * n_left
        pair_v = [-1] * W
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in candidates[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in candidates[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1

        if matching != n_left:
            return None
        return pair_u

    def check_window(L, W):
        """Check if window [L, L+W-1] admits a valid assignment.
        Returns list of 10 values or None."""
        candidates = []
        for d in divisors:
            r = (-L) % d
            pos = []
            p = r
            while p < W:
                pos.append(p)
                p += d
            if not pos:
                return None
            candidates.append(pos)

        result = hopcroft_karp(candidates, W)
        if result is None:
            return None

        assignment = [L + result[i] for i in range(n)]
        return assignment

    # Search W from minimum upward
    for W in range(n - 1, 300):
        if time.time() > deadline:
            break

        # The pattern of valid positions depends on L mod lcm_all.
        # We need to try all residue classes. But we can limit L range.
        # For each residue r in [0, lcm_all-1], pick L = r+1 if r+1+W-1 <= MAX_V,
        # else L = r+1 + k*lcm_all for smallest k making L+W-1 <= MAX_V.
        # Actually simplest: try L = 1, 2, ..., lcm_all (covers all residues).
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            if L + W - 1 > MAX_V:
                continue

            result = check_window(L, W)
            if result is not None:
                # Verify
                if len(set(result)) != n:
                    continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                return [int(v) for v in result]

    # Fallback: return divisors themselves (valid but not optimal)
    return [int(d) for d in divisors]
