import time
import random
import math

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 135  # safety margin under 150s kill

    # Precompute all edges (u < v, differ in one bit)
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # Precompute all 2-faces (672) and their 4 edges
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    dv = x ^ (1 << i) ^ (1 << j)
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, dv), max(b, dv))]
                    e3 = edge_idx[(min(dv, c), max(dv, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
    n_faces = len(face_edges)  # 672

    # Each edge is incident to exactly 2 faces (a 4-cube edge lies in 2 squares)
    # Precompute the two face indices per edge for fast updates
    ef = edge_faces  # alias

    def greedy_hit_set():
        """Greedy: pick edge covering most uncovered faces."""
        R = [False] * m
        fc = [0] * n_faces
        unc = [True] * n_faces
        n_unc = n_faces
        while n_unc > 0:
            best_e = -1
            best_g = -1
            for e in range(m):
                if R[e]:
                    continue
                g = 0
                for f in ef[e]:
                    if unc[f]:
                        g += 1
                if g > best_g:
                    best_g = g
                    best_e = e
            if best_e < 0:
                break
            R[best_e] = True
            for f in ef[best_e]:
                if unc[f]:
                    unc[f] = False
                    n_unc -= 1
                fc[f] += 1
        return R, fc

    def recompute_fc(R):
        fc = [0] * n_faces
        for e in range(m):
            if R[e]:
                for f in ef[e]:
                    fc[f] += 1
        return fc

    # Smooth surrogate: log-sum-exp of (uncovered penalty) for faces with fc==0.
    # We approximate the "pressure" from uncovered faces with a soft count.
    # When n_unc == 0 the surrogate is 0; otherwise it grows with the number
    # of uncovered faces but is differentiable-ish w.r.t. single-edge moves.
    def smooth_uncovered(n_unc, scale=1.0):
        if n_unc <= 0:
            return 0.0
        # soft-min style: scale * log(1 + exp(n_unc / scale))
        x = n_unc / scale
        if x > 50:
            return scale * (x + math.log(1.0))
        return scale * math.log1p(math.exp(x))

    LAMBDA = 6.0  # penalty weight for uncovered faces

    def sa_minimize(R_init, fc_init, budget_sec, T0=3.0, T_min=0.01, cool=0.9998):
        """Simulated annealing to minimize |R| (hitting set size).
        R[e]=True means edge e is REMOVED (in hitting set).
        Selected edges = those with R[e]=False.
        """
        R = R_init[:]
        fc = fc_init[:]
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]

        n_unc = sum(1 for c in fc if c == 0)
        best_unc = n_unc

        T = T0
        t_end = time.time() + budget_sec

        while time.time() < t_end:
            e = random.randint(0, m - 1)
            if R[e]:
                # Try removing edge e from hitting set (add it back to selection)
                newly_unc = 0
                for f in ef[e]:
                    if fc[f] == 1:
                        newly_unc += 1
                new_n_unc = n_unc + newly_unc
                new_size = cur_size - 1

                # Apply
                for f in ef[e]:
                    fc[f] -= 1
                R[e] = False
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                # Acceptance
                if n_unc == 0:
                    accept = True
                else:
                    # Smooth surrogate cost
                    old_cost = cur_size + 1 + LAMBDA * smooth_uncovered(n_unc - newly_unc)
                    new_cost = cur_size + LAMBDA * smooth_uncovered(n_unc)
                    delta = new_cost - old_cost
                    accept = (random.random() < math.exp(-delta / T))

                if not accept:
                    # Revert
                    R[e] = True
                    for f in ef[e]:
                        fc[f] += 1
                    cur_size += 1
                    n_unc -= newly_unc
            else:
                # Try adding edge e to hitting set (remove from selection)
                dec_unc = 0
                for f in ef[e]:
                    if fc[f] == 0:
                        dec_unc += 1
                new_n_unc = n_unc - dec_unc
                new_size = cur_size + 1

                # Apply
                R[e] = True
                for f in ef[e]:
                    fc[f] += 1
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                # Acceptance
                if dec_unc > 0:
                    accept = True  # reduces uncovered, always good
                else:
                    old_cost = cur_size - 1 + LAMBDA * smooth_uncovered(n_unc + dec_unc)
                    new_cost = cur_size + LAMBDA * smooth_uncovered(n_unc)
                    delta = new_cost - old_cost
                    accept = (random.random() < math.exp(-delta / T))

                if not accept:
                    # Revert
                    R[e] = False
                    for f in ef[e]:
                        fc[f] -= 1
                    cur_size -= 1
                    n_unc += dec_unc

            T *= cool
            if T < T_min:
                T = T_min

        return best_R, best_size

    def greedy_reduce(R):
        """Remove redundant edges from hitting set (fc[f] > 1 for all faces of e)."""
        changed = True
        while changed:
            changed = False
            fc = recompute_fc(R)
            for e in range(m):
                if R[e]:
                    ok = True
                    for f in ef[e]:
                        if fc[f] <= 1:
                            ok = False
                            break
                    if ok:
                        R[e] = False
                        changed = True
                        break
        return R

    def two_swap_polish(R, budget_sec):
        """Try to remove two edges at once from hitting set if validity holds."""
        fc = recompute_fc(R)
        n_unc = sum(1 for c in fc if c == 0)
        if n_unc > 0:
            return R
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]
        t_end = time.time() + budget_sec
        # List of currently-in-hitting-set edges
        in_hs = [e for e in range(m) if R[e]]
        random.shuffle(in_hs)
        while time.time() < t_end:
            # pick two random edges from hitting set
            if len(in_hs) < 2:
                break
            i = random.randint(0, len(in_hs) - 1)
            j = random.randint(0, len(in_hs) - 1)
            if i == j:
                continue
            e1 = in_hs[i]
            e2 = in_hs[j]
            # Check if removing both keeps all faces covered
            ok = True
            for f in ef[e1]:
                if fc[f] <= 1:
                    # face f is only covered by e1 (or e2 also?)
                    # check if e2 also covers f
                    if f not in ef[e2]:
                        ok = False
                        break
            if ok:
                for f in ef[e2]:
                    if fc[f] <= 1:
                        if f not in ef[e1]:
                            ok = False
                            break
            if ok:
                # Both removable
                R[e1] = False
                R[e2] = False
                for f in ef[e1]:
                    fc[f] -= 1
                for f in ef[e2]:
                    fc[f] -= 1
                in_hs = [e for e in in_hs if e != e1 and e != e2]
                cur_size -= 2
                if cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]
                # Continue greedy reduce
                R = greedy_reduce(R)
                fc = recompute_fc(R)
                cur_size = sum(R)
                if cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]
                in_hs = [e for e in range(m) if R[e]]
                random.shuffle(in_hs)
        return best_R

    # --- Main search ---
    R0, fc0 = greedy_hit_set()
    initial_size = sum(R0)

    best_R = R0[:]
    best_size = initial_size

    # Greedy reduce the initial solution
    best_R = greedy_reduce(best_R)
    best_size = sum(best_R)

    # Multi-restart SA
    num_restarts = 6
    base_time = 16.0
    for r in range(num_restarts):
        if time.time() > deadline - 25:
            break
        if r == 0:
            R_r = best_R[:]
            fc_r = recompute_fc(R_r)
        else:
            # Perturb best solution
            R_r = best_R[:]
            k = random.randint(4, 10)
            for _ in range(k):
                e = random.randint(0, m - 1)
                R_r[e] = not R_r[e]
            fc_r = recompute_fc(R_r)

        per_time = base_time if time.time() < deadline - 40 else max(2.0, deadline - 25 - time.time())
        R_new, size_new = sa_minimize(R_r, fc_r, per_time)
        # Greedy reduce the SA result
        R_new = greedy_reduce(R_new)
        size_new = sum(R_new)
        if size_new < best_size:
            best_size = size_new
            best_R = R_new[:]

    # Final polish: very low temperature SA
    if time.time() < deadline - 15:
        fc_cur = recompute_fc(best_R)
        R_pol, size_pol = sa_minimize(best_R, fc_cur, min(10.0, deadline - 15 - time.time()),
                                       T0=0.5, T_min=0.005, cool=0.999)
        R_pol = greedy_reduce(R_pol)
        size_pol = sum(R_pol)
        if size_pol < best_size:
            best_size = size_pol
            best_R = R_pol[:]

    # Two-swap polish
    if time.time() < deadline - 10:
        R_sw = two_swap_polish(best_R, min(8.0, deadline - 10 - time.time()))
        if sum(R_sw) < best_size:
            best_size = sum(R_sw)
            best_R = R_sw[:]

    # Final greedy reduce
    best_R = greedy_reduce(best_R)
    best_size = sum(best_R)

    # Validate
    fc_final = recompute_fc(best_R)
    is_valid = all(c > 0 for c in fc_final)
    if not is_valid:
        # Fallback: greedy hit set
        R_g, fc_g = greedy_hit_set()
        R_g = greedy_reduce(R_g)
        best_R = R_g
        best_size = sum(R_g)

    # Build output: selected edges are those NOT in hitting set
    out = []
    for e in range(m):
        if not best_R[e]:
            x, y = all_edges[e]
            out.append([x, y])

    return out

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")

    d = 7
    n = 1 << d
    all_edges_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_set.add((x, y))

    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))

    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)

                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))

                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1

                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break

    if valid:
        print("Valid!")
    else:
        print("Invalid!")
