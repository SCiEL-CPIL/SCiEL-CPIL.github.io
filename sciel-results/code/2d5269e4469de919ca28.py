import time
import random
import math

def _distinct4(a, b, c, d):
    cnt = 1
    if b != a:
        cnt += 1
    if c != a and c != b:
        cnt += 1
    if d != a and d != b and d != c:
        cnt += 1
    return cnt

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = 128

    # Precompute 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    m = len(aps)
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)

    # Greedy fallback (seed)
    def seed_greedy():
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    s = {color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]}
                    if len(s) < 3:
                        ok = False
                        break
                if ok:
                    for d in range(1, (x - 1) // 2 + 1):
                        if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                            ok = False
                            break
                if ok:
                    colors.append(color)
                    break
        return colors

    seed_colors = seed_greedy()
    best = seed_colors
    best_k = len(set(seed_colors))

    def is_valid(cols):
        for ap in aps:
            if _distinct4(cols[ap[0]], cols[ap[1]], cols[ap[2]], cols[ap[3]]) < 3:
                return False
        return True

    # --- New structural approach: targeted repair with color-specific moves ---
    # Instead of pure random restarts, we:
    # 1. Start from a random k-coloring
    # 2. Compute violations
    # 3. For each violating AP, try changing one of its positions to a color
    #    that would fix it (prioritizing colors that don't create new violations)
    # 4. Use simulated annealing for broader exploration when stuck

    def try_k(k, time_budget, seed):
        rng = random.Random(seed)
        end = time.time() + time_budget
        best_local = None
        best_viol = None

        while time.time() < end:
            # Random initial coloring
            c = [rng.randrange(k) for _ in range(n)]
            # Compute distinct counts for all APs
            ap_dist = [_distinct4(c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]]) for ap in aps]
            viol_indices = [i for i, d in enumerate(ap_dist) if d < 3]
            viol = len(viol_indices)

            if viol == 0:
                return c

            if best_viol is None or viol < best_viol:
                best_viol = viol
                best_local = c[:]

            # Phase 1: Targeted repair - fix violations by changing colors in violating APs
            cur = c[:]
            cur_dist = ap_dist[:]
            cur_viol = viol
            improved = True
            max_repair_rounds = 200

            for round_num in range(max_repair_rounds):
                if time.time() > end:
                    break
                if cur_viol == 0:
                    return cur

                # Get current violations
                viol_set = [i for i, d in enumerate(cur_dist) if d < 3]
                if not viol_set:
                    break

                # Pick a random violating AP
                ap_idx = rng.choice(viol_set)
                ap = aps[ap_idx]
                a, b, cc, dd = ap
                cur_colors = [cur[a], cur[b], cur[cc], cur[dd]]

                # Try changing each position in the AP to a different color
                found_fix = False
                positions = [a, b, cc, dd]
                rng.shuffle(positions)

                for pos in positions:
                    old_col = cur[pos]
                    # Try each color except the old one
                    for new_col in range(k):
                        if new_col == old_col:
                            continue

                        # Check if this change fixes the AP without breaking others
                        delta = 0
                        for other_ap_idx in pos_aps[pos]:
                            oap = aps[other_ap_idx]
                            oa, ob, oc, od = oap
                            na = new_col if oa == pos else cur[oa]
                            nb = new_col if ob == pos else cur[ob]
                            nc = new_col if oc == pos else cur[oc]
                            nd = new_col if od == pos else cur[od]
                            new_d = _distinct4(na, nb, nc, nd)
                            old_d = cur_dist[other_ap_idx]
                            if old_d < 3 and new_d >= 3:
                                delta -= 1
                            elif old_d >= 3 and new_d < 3:
                                delta += 1

                        if delta < 0:
                            # This change improves things
                            cur[pos] = new_col
                            for other_ap_idx in pos_aps[pos]:
                                oap = aps[other_ap_idx]
                                oa, ob, oc, od = oap
                                na = new_col if oa == pos else cur[oa]
                                nb = new_col if ob == pos else cur[ob]
                                nc = new_col if oc == pos else cur[oc]
                                nd = new_col if od == pos else cur[od]
                                cur_dist[other_ap_idx] = _distinct4(na, nb, nc, nd)
                            cur_viol += delta
                            found_fix = True
                            break

                    if found_fix:
                        break

                if not found_fix:
                    # No targeted fix found; do a few random SA steps
                    for _ in range(10):
                        pos = rng.randrange(n)
                        old_col = cur[pos]
                        new_col = rng.randrange(k)
                        if new_col == old_col:
                            continue
                        delta = 0
                        for ap_idx2 in pos_aps[pos]:
                            oap = aps[ap_idx2]
                            oa, ob, oc, od = oap
                            na = new_col if oa == pos else cur[oa]
                            nb = new_col if ob == pos else cur[ob]
                            nc = new_col if oc == pos else cur[oc]
                            nd = new_col if od == pos else cur[od]
                            new_d = _distinct4(na, nb, nc, nd)
                            old_d = cur_dist[ap_idx2]
                            if old_d < 3 and new_d >= 3:
                                delta -= 1
                            elif old_d >= 3 and new_d < 3:
                                delta += 1
                            cur_dist[ap_idx2] = new_d
                        new_viol = cur_viol + delta
                        if new_viol < cur_viol or (new_viol == cur_viol and rng.random() < 0.5):
                            cur[pos] = new_col
                            cur_viol = new_viol
                        else:
                            # Revert
                            for ap_idx2 in pos_aps[pos]:
                                oap = aps[ap_idx2]
                                oa, ob, oc, od = oap
                                na = old_col if oa == pos else cur[oa]
                                nb = old_col if ob == pos else cur[ob]
                                nc = old_col if oc == pos else cur[oc]
                                nd = old_col if od == pos else cur[od]
                                cur_dist[ap_idx2] = _distinct4(na, nb, nc, nd)

                    if cur_viol == 0:
                        return cur
                    if best_viol is None or cur_viol < best_viol:
                        best_viol = cur_viol
                        best_local = cur[:]

            # Phase 2: Simulated annealing for broader exploration
            temp = 5.0
            step = 0
            while time.time() < end:
                step += 1
                pos = rng.randrange(n)
                old_col = cur[pos]
                new_col = rng.randrange(k)
                if new_col == old_col:
                    continue
                delta = 0
                for ap_idx2 in pos_aps[pos]:
                    oap = aps[ap_idx2]
                    oa, ob, oc, od = oap
                    na = new_col if oa == pos else cur[oa]
                    nb = new_col if ob == pos else cur[ob]
                    nc = new_col if oc == pos else cur[oc]
                    nd = new_col if od == pos else cur[od]
                    new_d = _distinct4(na, nb, nc, nd)
                    old_d = cur_dist[ap_idx2]
                    if old_d < 3 and new_d >= 3:
                        delta -= 1
                    elif old_d >= 3 and new_d < 3:
                        delta += 1
                    cur_dist[ap_idx2] = new_d
                new_viol = cur_viol + delta
                if new_viol < cur_viol or (new_viol > cur_viol and rng.random() < math.exp((cur_viol - new_viol) / temp)):
                    cur[pos] = new_col
                    cur_viol = new_viol
                    if cur_viol == 0:
                        return cur
                    if best_viol is None or cur_viol < best_viol:
                        best_viol = cur_viol
                        best_local = cur[:]
                else:
                    for ap_idx2 in pos_aps[pos]:
                        oap = aps[ap_idx2]
                        oa, ob, oc, od = oap
                        na = old_col if oa == pos else cur[oa]
                        nb = old_col if ob == pos else cur[ob]
                        nc = old_col if oc == pos else cur[oc]
                        nd = old_col if od == pos else cur[od]
                        cur_dist[ap_idx2] = _distinct4(na, nb, nc, nd)
                if step % 500 == 0:
                    temp = max(0.1, temp * 0.95)

        if best_viol == 0:
            return best_local
        return None

    # Try k from 3 up
    k = 3
    while time.time() < deadline and k <= best_k:
        budget = min(12.0, deadline - time.time())
        if budget < 2.0:
            break
        res = try_k(k, budget, seed=1000 + k)
        if res is not None:
            best = res
            best_k = k
            break
        k += 1

    # Verify best is valid
    if not is_valid(best):
        best = seed_colors

    return best
