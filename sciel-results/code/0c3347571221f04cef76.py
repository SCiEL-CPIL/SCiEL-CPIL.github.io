import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s kill, keep 5s margin

    n = 128

    # Precompute all 4-term APs (0-indexed): (a, a+d, a+2d, a+3d)
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))

    # pos -> list of ap indices
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    ap_count = [len(pos_to_aps[i]) for i in range(n)]

    def count_violations(colors_arr):
        v = 0
        for a, b, c, d in ap_list:
            if len((colors_arr[a], colors_arr[b], colors_arr[c], colors_arr[d])) < 3:
                v += 1
        return v

    def is_valid_full(colors_arr):
        for a, b, c, d in ap_list:
            if len((colors_arr[a], colors_arr[b], colors_arr[c], colors_arr[d])) < 3:
                return False
        return True

    # ---------- Fast greedy (deterministic) to get a valid baseline ----------
    def greedy_build(k_colors, seed=None):
        rng = random.Random(seed)
        colors = [-1] * n
        # order: most constrained first
        order = sorted(range(n), key=lambda i: -ap_count[i])
        for pos in order:
            cand = list(range(k_colors))
            rng.shuffle(cand)
            chosen = None
            for col in cand:
                ok = True
                for aidx in pos_to_aps[pos]:
                    a, b, c, d = ap_list[aidx]
                    vals = [colors[a], colors[b], colors[c], colors[d]]
                    # place col at pos
                    for i, p in enumerate((a, b, c, d)):
                        if p == pos:
                            vals[i] = col
                    # check: if all assigned, must be >=3 distinct; if 3 assigned and all same -> impossible
                    asg = [x for x in vals if x != -1]
                    if len(asg) == 4:
                        if len(set(asg)) < 3:
                            ok = False
                            break
                    elif len(asg) == 3:
                        if len(set(asg)) == 1:
                            ok = False
                            break
                if ok:
                    chosen = col
                    break
            if chosen is None:
                # fallback: pick first that doesn't immediately break a full AP
                for col in cand:
                    colors[pos] = col
                    bad = False
                    for aidx in pos_to_aps[pos]:
                        a, b, c, d = ap_list[aidx]
                        vals = [colors[a], colors[b], colors[c], colors[d]]
                        if all(x != -1 for x in vals) and len(set(vals)) < 3:
                            bad = True
                            break
                    colors[pos] = -1
                    if not bad:
                        chosen = col
                        break
            if chosen is None:
                chosen = 0
            colors[pos] = chosen
        return colors

    # ---------- Local search: minimize distinct colors while staying valid ----------
    def local_search_reduce(colors_arr, k, iters):
        # colors_arr uses labels 0..k-1
        cur = list(colors_arr)
        used = set(cur)
        cur_colors = len(used)
        best = cur
        best_colors = cur_colors
        rng = random.Random(12345)
        for it in range(iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for pos in range(n):
                if time.time() > deadline - 5:
                    break
                old = cur[pos]
                # try to change to another used color to possibly drop a color
                for newc in range(k):
                    if newc == old:
                        continue
                    cur[pos] = newc
                    if count_violations(cur) == 0:
                        nu = set(cur)
                        if len(nu) < best_colors:
                            best = cur[:]
                            best_colors = len(nu)
                            improved = True
                            break
                        # keep change even if same count (helps future reductions)
                        break
                    else:
                        cur[pos] = old
                if improved:
                    break
            if not improved:
                # try a random perturbation
                for _ in range(5):
                    p = rng.randrange(n)
                    cur[p] = rng.randrange(k)
                if count_violations(cur) == 0:
                    continue
                else:
                    cur = best[:]
        return best, best_colors

    # ---------- Backtracking with forward checking (MRV) for exact K colors ----------
    def backtrack_k(k, seed=0):
        rng = random.Random(seed)
        colors = [-1] * n
        # domain per position
        domains = [set(range(k)) for _ in range(n)]
        assigned = [False] * n
        assigned_count = [0]

        def check_pos(pos, col):
            for aidx in pos_to_aps[pos]:
                a, b, c, d = ap_list[aidx]
                vals = [colors[a], colors[b], colors[c], colors[d]]
                for i, p in enumerate((a, b, c, d)):
                    if p == pos:
                        vals[i] = col
                asg = [x for x in vals if x != -1]
                if len(asg) == 4:
                    if len(set(asg)) < 3:
                        return False
                elif len(asg) == 3:
                    if len(set(asg)) == 1:
                        return False
            return True

        def forward_check(pos):
            # after assigning pos, reduce domains of unassigned neighbors
            for aidx in pos_to_aps[pos]:
                a, b, c, d = ap_list[aidx]
                members = (a, b, c, d)
                unassigned = [p for p in members if not assigned[p]]
                if len(unassigned) == 0:
                    vals = [colors[p] for p in members]
                    if len(set(vals)) < 3:
                        return False
                elif len(unassigned) == 1:
                    p = unassigned[0]
                    assigned_vals = [colors[q] for q in members if q != p]
                    # p must make total distinct >=3
                    # remove colors that would make it invalid
                    new_dom = set()
                    for col in domains[p]:
                        trial = assigned_vals + [col]
                        if len(set(trial)) >= 3:
                            new_dom.add(col)
                    if not new_dom:
                        return False
                    domains[p] = new_dom
            return True

        def backtrack():
            if time.time() > deadline:
                return None
            if assigned_count[0] == n:
                return colors[:]
            # MRV: pick unassigned with smallest domain
            best_pos = -1
            best_size = 10**9
            for pos in range(n):
                if assigned[pos]:
                    continue
                sz = len(domains[pos])
                if sz == 0:
                    return None
                if sz < best_size:
                    best_size = sz
                    best_pos = pos
                    if sz == 1:
                        break
            if best_pos == -1:
                return None
            cols = list(domains[best_pos])
            rng.shuffle(cols)
            for col in cols:
                colors[best_pos] = col
                assigned[best_pos] = True
                assigned_count[0] += 1
                # snapshot domains
                dom_snap = [set(d) for d in domains]
                if forward_check(best_pos):
                    res = backtrack()
                    if res is not None:
                        return res
                # restore
                for i in range(n):
                    domains[i] = dom_snap[i]
                assigned[best_pos] = False
                assigned_count[0] -= 1
                colors[best_pos] = -1
            return None

        return backtrack()

    # ---------- Try to find a valid 3-coloring ----------
    best_solution = None
    best_color_count = 10**9

    # 1) Quick greedy with 3 colors + local search
    for seed in [0, 1, 2]:
        if time.time() > deadline - 20:
            break
        g = greedy_build(3, seed=seed)
        if is_valid_full(g):
            red, cc = local_search_reduce(g, 3, 200)
            if cc < best_color_count:
                best_color_count = cc
                best_solution = red
            if best_color_count <= 3:
                return best_solution

    # 2) Backtracking with 3 colors (a few randomized restarts)
    for seed in range(8):
        if time.time() > deadline - 10:
            break
        res = backtrack_k(3, seed=seed)
        if res is not None and is_valid_full(res):
            best_color_count = len(set(res))
            best_solution = res
            if best_color_count <= 3:
                return best_solution

    # 3) Fallback: greedy with more colors, then reduce
    for k in [4, 5, 6]:
        if time.time() > deadline - 5:
            break
        for seed in [0, 1]:
            if time.time() > deadline - 5:
                break
            g = greedy_build(k, seed=seed)
            if is_valid_full(g):
                red, cc = local_search_reduce(g, k, 300)
                if cc < best_color_count:
                    best_color_count = cc
                    best_solution = red
                if best_color_count <= 3:
                    return best_solution

    if best_solution is None:
        # last resort: a guaranteed-valid coloring using many colors (identity-like)
        # Use color = index mod 128 ensures every AP has 4 distinct? Not always.
        # Use color = position (0..127): any 4-term AP has 4 distinct positions -> 4 distinct colors. Valid.
        best_solution = list(range(n))
        best_color_count = n

    # Ensure labels in 0..127 and relabel to compact if beneficial
    if best_solution is not None:
        # relabel to 0..m-1 preserving distinctness (doesn't change count)
        mapping = {}
        out = []
        for c in best_solution:
            if c not in mapping:
                mapping[c] = len(mapping)
            out.append(mapping[c])
        best_solution = out

    return best_solution
