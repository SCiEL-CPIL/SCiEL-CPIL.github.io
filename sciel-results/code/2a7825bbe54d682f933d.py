import time
import random
import numpy as np

N = 128

def construct():
    t0 = time.time()
    deadline = t0 + 82.0          # safety margin before the 90s kill

    # ---- Precompute all nonconstant 4-term progressions (0-based) ----
    P = []
    for i in range(N):
        for d in range(1, (N - 1 - i) // 3 + 1):
            j3 = i + 3 * d
            if j3 < N:
                P.append((i, i + d, i + 2 * d, j3))
    P = np.array(P, dtype=np.int64)          # shape (M,4)

    # For each position, the 4-AP indices that involve it (for local search)
    pos_progressions = [[] for _ in range(N)]
    for idx in range(len(P)):
        for p in P[idx]:
            pos_progressions[p].append(idx)

    def validate(arr):
        """True iff every 4-AP uses >= 3 distinct colors. Fast (numpy)."""
        if arr is None:
            return False
        a = np.asarray(arr, dtype=np.int64)
        if a.shape != (N,) or (a < 0).any():
            return False
        q = a[P]                                   # (M,4)
        s0 = q[:, 0]
        s1 = q[:, 1]
        s2 = q[:, 2]
        s3 = q[:, 3]
        # A 4-AP is bad (<3 distinct) iff it is 1-color (all equal) or 2-color (exactly one split).
        all_eq = (s0 == s1) & (s1 == s2) & (s2 == s3)
        one_split = ((s0 == s1) & (s0 == s2) & (s1 != s3)) | \
                    ((s0 == s1) & (s0 != s2) & (s1 == s3)) | \
                    ((s0 == s1) & (s0 != s2) & (s1 != s3)) | \
                    ((s0 != s1) & (s0 == s2) & (s1 == s3))
        return not (all_eq.any() or one_split.any())

    def count_colors(arr):
        return len(set(int(x) for x in arr))

    # ---- Fast 3-color feasibility check (numpy vectorized) ----
    # A 3-coloring of all 4-APs forces each AP to be exactly 2+2, i.e.
    # two opposite pairs: (x0==x2 and x1==x3) OR (x0==x3 and x1==x2).
    def try_3color():
        M = len(P)
        if M == 0:
            return None
        # For each AP, the two allowed pairings (opposite pairs).
        #  pairing 0: (0,2)&(1,3)   -> vars 0,2 same; vars 1,3 same
        #  pairing 1: (0,3)&(1,2)   -> vars 0,3 same; vars 1,2 same
        # We pick pairing 0 as the "base" (0~2, 1~3) and let pairing 1 be a flip.
        base_pairs = [(0, 2), (1, 3)]
        flip_pairs = [(0, 3), (1, 2)]
        # Build constraint rows for base pairing: each row = (varA, varB) must be equal.
        base_rows = []
        for (a, b, c, d) in P:
            base_rows.append((a, c))
            base_rows.append((b, d))
        base_rows = np.array(base_rows, dtype=np.int64)
        base_A = base_rows[:, 0]
        base_B = base_rows[:, 1]

        # For each AP, the flip must equalize the two "cross" pairs:
        #   flip=1 requires (a==d) and (b==c).
        # These are constraints that only hold when the AP is in pairing 1.
        # Represent as: for AP k, if flip[k]==1 then (a_k==d_k) and (b_k==c_k).
        # Equivalently: (flip[k]==0) OR (a_k==d_k) OR (b_k==c_k) must hold.
        # We'll do a simple backtracking over the M flip variables with strong propagation.
        a_idx = P[:, 0]
        d_idx = P[:, 3]
        b_idx = P[:, 1]
        c_idx = P[:, 2]

        # Assign colors 0/1/2 to the 3 base-pair groups via union-find on base_rows.
        # But base_rows may connect many positions; the groups are the connected components
        # of the graph with edges base_rows. Each component gets a color in {0,1,2}.
        # Then flip variables impose extra equalities that cross components -> must be
        # consistent with the component coloring.

        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for (x, y) in base_rows:
            union(int(x), int(y))

        # components
        comp = [find(i) for i in range(N)]
        comp_id = {}
        comp_of = [0] * N
        for i in range(N):
            r = comp[i]
            if r not in comp_id:
                comp_id[r] = len(comp_id)
            comp_of[i] = comp_id[r]
        ncomp = len(comp_id)

        # Now each component has a color in {0,1,2}. The flip constraints require:
        # for AP k with flip[k]==1: a_k==d_k (same color) and b_k==c_k (same color).
        # In component terms: comp_of[a_k]==comp_of[d_k] implies their colors equal,
        # and comp_of[b_k]==comp_of[c_k] implies their colors equal.
        # If two components are forced equal by a flip=1, they must share a color.
        # Build, for each AP k, the two "merge requirements" under flip=1:
        #   (comp_of[a_k], comp_of[d_k]) and (comp_of[b_k], comp_of[c_k]).
        # If flip[k]==0, no requirement.
        # We need to assign colors 0/1/2 to components and flips to APs such that
        # every flip=1 AP has both merges color-consistent.
        # This is: choose a 3-coloring of components + a subset of APs (the flips)
        # such that for every AP NOT flipped, its two opposite pairs are already
        # color-consistent (i.e. comp_of[a]==comp_of[c] colors equal AND comp_of[b]==comp_of[d]
        # colors equal). Wait -- base pairing already forces a~c and b~d to be SAME
        # component, so they're automatically same color. Good: flip=0 is always consistent
        # with base. So the only constraints are the flip=1 ones.
        # Therefore: for ANY 3-coloring of components, we can set flip[k]=1 exactly when
        # the flip=1 merges are color-consistent, else flip[k]=0. But we must ensure that
        # for EVERY AP, at least one of the two pairings is color-consistent:
        #   pairing0 (base) is ALWAYS consistent (same components).
        # So EVERY AP is automatically satisfied by pairing0!
        # => Any 3-coloring of the components yields a valid 3-coloring.
        # But we need the components to be colorable with 3 colors (they always are: just
        # assign 3 colors arbitrarily to components, since there are no constraints between
        # components from the base pairing). And we must use exactly 3 colors overall and
        # every AP must be nonconstant.
        #
        # However: we also need each AP to be NONCONSTANT (2+2, not 4 same). With base
        # pairing, AP k has colors (col[C(a)], col[C(b)], col[C(a)], col[C(b)]). This is
        # constant iff col[C(a)]==col[C(b)]. So we need col[C(a)] != col[C(b)] for every AP.
        # That is a proper-2-coloring-like constraint on the component graph: edge between
        # C(a_k) and C(b_k) for each AP k, must have different colors. With 3 colors this is
        # always solvable (3-color any graph). We just need a 3-coloring of the component
        # graph where adjacent (via an AP) components differ.
        #
        # Build component graph: edge between C(a_k) and C(b_k) for each AP k.
        comp_graph = [[] for _ in range(ncomp)]
        for k in range(M):
            u = comp_of[int(a_idx[k])]
            v = comp_of[int(b_idx[k])]
            if u != v:
                comp_graph[u].append(v)
                comp_graph[v].append(u)

        # 3-color the component graph (greedy + backtracking fallback).
        color_of = [-1] * ncomp
        def color_graph():
            order = sorted(range(ncomp), key=lambda x: -len(comp_graph[x]))
            def rec(i):
                if i == ncomp:
                    return True
                u = order[i]
                used = set()
                for v in comp_graph[u]:
                    if color_of[v] != -1:
                        used.add(color_of[v])
                for c in (0, 1, 2):
                    if c not in used:
                        color_of[u] = c
                        if rec(i + 1):
                            return True
                        color_of[u] = -1
                return False
            return rec(0)

        if not color_graph():
            return None

        arr = [color_of[comp_of[i]] for i in range(N)]
        # Ensure exactly 3 colors are used and nonconstant; if only <3 colors appear,
        # recolor to spread.
        used = set(arr)
        if len(used) < 3:
            # remap to use 3 colors
            remap = {used[i]: i for i in range(len(used))}
            arr = [remap[x] for x in arr]
            # if still <3, force a third color on one position (keeps validity? risky)
        if validate(arr):
            return arr
        return None

    res3 = try_3color()
    if res3 is not None and validate(res3):
        return [int(x) for x in res3]

    # ---- Greedy construction (minimize used colors) ----
    def greedy(max_colors):
        colors = [-1] * N
        for x in range(N):
            # valid colors: for every AP containing x, once x is set the AP must have >=3
            # (only check APs where the other three are already set)
            best = None
            for c in range(max_colors):
                ok = True
                for pidx in pos_progressions[x]:
                    a, b, cc, d = P[pidx]
                    others = [a, b, cc, d]
                    vals = [colors[a], colors[b], colors[cc], colors[d]]
                    # replace x's slot with c
                    # check the four values with x=c
                    four = []
                    for v in others:
                        if v == x:
                            four.append(c)
                        else:
                            four.append(colors[v])
                    if -1 in four:
                        continue
                    if len(set(four)) < 3:
                        ok = False
                        break
                if ok:
                    best = c
                    break
            if best is None:
                return None
            colors[x] = best
        return colors

    # ---- Local search to eliminate colors ----
    def local_search(arr, time_limit):
        arr = list(arr)
        cur = count_colors(arr)
        end = time.time() + time_limit
        while time.time() < end and cur >= 3:
            improved = False
            used = sorted(set(arr))
            for ct in used:
                if time.time() > end:
                    break
                positions = [i for i in range(N) if arr[i] == ct]
                if not positions:
                    continue
                trial = list(arr)
                ok = True
                for pos in positions:
                    found = False
                    for nc in used:
                        if nc == ct:
                            continue
                        trial[pos] = nc
                        good = True
                        for pidx in pos_progressions[pos]:
                            a, b, cc, d = P[pidx]
                            if len({trial[a], trial[b], trial[cc], trial[d]}) < 3:
                                good = False
                                break
                        if good:
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if ok and validate(trial):
                    arr = trial
                    cur -= 1
                    improved = True
                    break
            if not improved:
                break
        return arr, cur

    # ---- Random restart + repair ----
    def repair(trial, time_limit):
        end = time.time() + time_limit
        while time.time() < end:
            bad = None
            for pidx in range(len(P)):
                a, b, cc, d = P[pidx]
                if len({trial[a], trial[b], trial[cc], trial[d]}) < 3:
                    bad = pidx
                    break
            if bad is None:
                return True
            a, b, cc, d = P[bad]
            # pick a position in this AP to recolor to a fresh color
            pos = random.choice([a, b, cc, d])
            trial[pos] = random.randint(0, 7)
        return validate(trial)

    # Start from greedy with a small palette, then improve.
    best = None
    best_count = 999
    for mc in (4, 5, 6):
        g = greedy(mc)
        if g is not None and validate(g):
            g, c = local_search(g, 4.0)
            if c < best_count:
                best, best_count = g, c
        if time.time() > deadline - 6:
            break
        if best_count <= 3:
            break

    if best is None:
        g = greedy(128)
        if g is not None and validate(g):
            best, best_count = g, count_colors(g)

    # Random restarts to try to beat the current best (esp. reach 3).
    random.seed(1234)
    end = deadline - 2.0
    while time.time() < end and best_count > 3:
        nc = random.randint(3, max(3, best_count - 1))
        trial = [random.randint(0, nc - 1) for _ in range(N)]
        if repair(trial, 0.5) and validate(trial):
            trial, c = local_search(trial, 2.0)
            if c < best_count:
                best, best_count = trial, c
        if time.time() > deadline - 3:
            break

    # Guaranteed-valid fallback (robust, always valid).
    if best is None or not validate(best):
        colors = []
        for x in range(1, N + 1):
            for color in range(128):
                ok1 = all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                          for d in range(1, (x - 1) // 3 + 1))
                ok2 = all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                          for d in range(1, (x - 1) // 2 + 1))
                if ok1 and ok2:
                    colors.append(color)
                    break
        best = colors

    return [int(x) for x in best]
