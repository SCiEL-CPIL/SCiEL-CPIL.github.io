"""Fast sum-free subset search via conflict-graph + bitsets, with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = 112.0  # safety margin under the 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve(S0):
        n = len(S0)
        if n == 0:
            return []
        # Bitsets over the n elements.
        sum_to = [None] * n
        dbl = [None] * n
        conflict = [None] * n
        for i in range(n):
            sum_to[i] = np.zeros(n, dtype=bool)
            dbl[i] = np.zeros(n, dtype=bool)
            conflict[i] = np.zeros(n, dtype=bool)
        # Precompute pairwise sums.
        for i in range(n):
            for j in range(i, n):
                s = S0[i] + S0[j]
                # find k with S0[k] == s
                # use dict for speed
                pass
        # Build a value->index map.
        idx = {}
        for i, v in enumerate(S0):
            idx[v] = i
        for i in range(n):
            for j in range(i, n):
                s = S0[i] + S0[j]
                if s in idx:
                    k = idx[s]
                    sum_to[k][i] = True
                    sum_to[k][j] = True
        for i in range(n):
            s = 2 * S0[i]
            if s in idx:
                k = idx[s]
                dbl[k][i] = True
        # Conflict graph: i,j conflict if i+j in S0 or 2i in S0 (as the "c").
        # i and j cannot both be in a sum-free set iff (i+j) is in S0 OR (2i) in S0 OR (2j) in S0.
        for i in range(n):
            for j in range(i, n):
                # conflict if there exists k in S0 with i+j==k, or 2i==k, or 2j==k
                if (i + j) in idx or (2 * S0[i]) in idx or (2 * S0[j]) in idx:
                    conflict[i][j] = True
                    conflict[j][i] = True
        # Also a self-conflict: 2i in S0 means i cannot be in set (i+i=2i in S0).
        self_conflict = np.zeros(n, dtype=bool)
        for i in range(n):
            if (2 * S0[i]) in idx:
                self_conflict[i] = True

        def feasible(S):
            # S: bool array of length n. Check sum-free: for any i,j in S, i+j not in S.
            if not S.any():
                return True
            # For each i in S, check sum_to[i] & S == 0 and dbl[i] & S == 0
            for i in np.nonzero(S)[0]:
                if (sum_to[i] & S).any() or (dbl[i] & S).any():
                    return False
            return True

        def can_add(S, i):
            # Can we add element i to S (i not in S) and stay sum-free?
            # Need: no j in S with i+j in S  -> sum_to[i] & S == 0
            # and no j in S with 2i ... actually 2i in S means adding i gives i+i=2i in S -> conflict.
            # Also for each j in S, j+j=2j in S is already enforced by S being sum-free.
            # Adding i: new triples involving i: i+i=2i (need 2i not in S), i+j (need i+j not in S for all j in S).
            if S[i]:
                return False
            if (sum_to[i] & S).any():
                return False
            if (dbl[i] & S).any():
                return False
            return True

        def can_remove_add(S, r, a):
            # Remove r (in S), add a (not in S). Check result is sum-free.
            # Only need to check triples involving a (others unchanged).
            S2 = S.copy()
            S2[r] = False
            S2[a] = True
            # check triples with a
            if (sum_to[a] & S2).any():
                return False
            if (dbl[a] & S2).any():
                return False
            return True

        best = np.zeros(n, dtype=bool)
        best_size = 0

        # ---- Multiple restarts: greedy (random order) + local search ----
        rng = random.Random(12345)
        max_restarts = 4000

        def local_search(S):
            # 1-swap: remove one, add one. Iterate to local optimum.
            improved = True
            passes = 0
            while improved and passes < 200:
                improved = False
                passes += 1
                if time.time() > DEADLINE:
                    break
                # Try all (remove r, add a) with net +1
                S_idx = np.nonzero(S)[0]
                notS_idx = np.nonzero(~S)[0]
                for r in S_idx:
                    for a in notS_idx:
                        if can_remove_add(S, r, a):
                            S = S.copy()
                            S[r] = False
                            S[a] = True
                            improved = True
                            break
                    if improved:
                        break
                # Try pure add
                if not improved:
                    for a in notS_idx:
                        if can_add(S, a):
                            S = S.copy()
                            S[a] = True
                            improved = True
                            break
            return S

        for restart in range(max_restarts):
            if time.time() > DEADLINE:
                break
            # Random greedy: pick random order, add if feasible.
            order = list(range(n))
            rng.shuffle(order)
            S = np.zeros(n, dtype=bool)
            for i in order:
                if can_add(S, i):
                    S[i] = True
            # Local search
            S = local_search(S)
            if S.sum() > best_size:
                best_size = S.sum()
                best = S.copy()

        # ---- 2-opt swaps for polish on best ----
        def polish(S):
            improved = True
            passes = 0
            while improved and passes < 100:
                improved = False
                passes += 1
                if time.time() > DEADLINE:
                    break
                S_idx = np.nonzero(S)[0]
                notS_idx = np.nonzero(~S)[0]
                # 2-opt: remove r1,r2 add a1,a2
                for r1 in S_idx:
                    for r2 in S_idx:
                        if r1 == r2:
                            continue
                        for a1 in notS_idx:
                            for a2 in notS_idx:
                                if a1 == a2:
                                    continue
                                if a1 in (r1, r2) or a2 in (r1, r2):
                                    continue
                                S2 = S.copy()
                                S2[r1] = False
                                S2[r2] = False
                                S2[a1] = True
                                S2[a2] = True
                                ok = True
                                for a in (a1, a2):
                                    if (sum_to[a] & S2).any() or (dbl[a] & S2).any():
                                        ok = False
                                        break
                                if ok:
                                    S = S2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            return S

        best = polish(best)
        return [S0[i] for i in np.nonzero(best)[0]]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S0 = instance(size, seed)
        # deadline-aware: solve
        ans = solve(S0)
        # Final verification: must be sum-free
        ans_set = set(ans)
        valid = True
        for a in ans:
            for b in ans:
                if a + b in ans_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to odd subset (guaranteed sum-free)
            ans = [x for x in S0 if x % 2]
        results.append(sorted(ans))
    return results
# EVOLVE-BLOCK-END
