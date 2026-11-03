"""Difference triangle set (4,7) - gap representation + cross-row exchange local search."""
import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 138.0

    def elapsed_ok():
        return time.time() < deadline

    def gaps_to_row(gaps):
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        return row

    def row_diffs(gaps):
        """Return the set of all positive differences (consecutive partial sums)."""
        diffs = set()
        n = len(gaps)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += gaps[j]
                diffs.add(s)
        return diffs

    def all_diffs(gap_list):
        """Union of differences over all rows (represented by gap lists)."""
        used = set()
        for gaps in gap_list:
            used |= row_diffs(gaps)
        return used

    def build_row_gaps(used, k, rng):
        """Build k gaps (row) whose differences avoid `used`. Returns list of k ints or None."""
        gaps = []
        cur = 0
        for _ in range(k):
            # Try small gaps first (tight packing), with light randomization
            found = False
            order = list(range(1, 40))
            rng.shuffle(order)
            # bias toward small: put 1..8 first in random order
            small = list(range(1, 9))
            rng.shuffle(small)
            order = small + [x for x in order if x not in small]
            for g in order:
                new_val = cur + g
                # new differences involving this gap:
                # diff to 0: new_val
                # diff to each previous element: new_val - prefix
                new_diffs = {new_val}
                prefix = cur
                for gg in gaps:
                    prefix -= gg
                    new_diffs.add(new_val - prefix)
                if new_diffs & used:
                    continue
                gaps.append(g)
                cur = new_val
                used |= new_diffs
                found = True
                break
            if not found:
                return None
        return gaps

    def greedy_gap_solution(k, seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            gaps = build_row_gaps(used, k, rng)
            if gaps is None:
                return None
            rows.append(gaps)
        return rows

    # --- Initial solutions: multiple greedy starts ---
    candidates = []
    for seed in range(40):
        if not elapsed_ok():
            break
        sol = greedy_gap_solution(k, seed)
        if sol is not None:
            scope = max(sum(g) for g in sol)
            candidates.append((scope, sol))
    if not candidates:
        # Fallback: pure greedy
        used = set()
        sol = []
        for _ in range(n_rows):
            gaps = []
            cur = 0
            for _ in range(k):
                g = 1
                while True:
                    nv = cur + g
                    nd = {nv}
                    p = cur
                    for gg in gaps:
                        p -= gg
                        nd.add(nv - p)
                    if nd & used:
                        g += 1
                    else:
                        gaps.append(g)
                        cur = nv
                        used |= nd
                        break
            sol.append(gaps)
        candidates.append((max(sum(g) for g in sol), sol))

    candidates.sort(key=lambda x: x[0])
    best_scope, best_sol = candidates[0]

    # --- Local search with cross-row gap-exchange moves ---
    def local_search(sol, rng, time_budget):
        cur_sol = [g[:] for g in sol]
        cur_scope = max(sum(g) for g in cur_sol)
        cur_used = all_diffs(cur_sol)
        end = time.time() + time_budget
        improved_any = False

        while time.time() < end:
            progress = False
            for r in range(n_rows):
                # Try to shrink a gap in row r
                for pos in range(k):
                    old_g = cur_sol[r][pos]
                    if old_g <= 1:
                        continue
                    # Try reducing this gap by 1..(old_g-1)
                    for delta in range(1, old_g):
                        new_g = old_g - delta
                        # recompute row r diffs
                        new_row = cur_sol[r][:]
                        new_row[pos] = new_g
                        nd = row_diffs(new_row)
                        # remove old row diffs from used
                        old_row = cur_sol[r]
                        od = row_diffs(old_row)
                        # candidate used set: cur_used - od + nd
                        # check nd against (cur_used - od)
                        other = cur_used - od
                        if nd & other:
                            continue
                        # valid
                        cur_sol[r] = new_row
                        cur_used = (cur_used - od) | nd
                        ns = sum(new_row)
                        cur_scope = max(cur_scope, max(sum(g) for g in cur_sol))
                        progress = True
                        improved_any = True
                        break
                    if progress:
                        break
                if progress:
                    break
            # Cross-row gap exchange
            if not progress:
                for r1 in range(n_rows):
                    for r2 in range(r1 + 1, n_rows):
                        for p1 in range(k):
                            for p2 in range(k):
                                g1 = cur_sol[r1][p1]
                                g2 = cur_sol[r2][p2]
                                if g1 == g2:
                                    continue
                                # swap
                                row1 = cur_sol[r1][:]
                                row2 = cur_sol[r2][:]
                                row1[p1] = g2
                                row2[p2] = g1
                                d1 = row_diffs(row1)
                                d2 = row_diffs(row2)
                                old1 = row_diffs(cur_sol[r1])
                                old2 = row_diffs(cur_sol[r2])
                                other = cur_used - old1 - old2
                                if (d1 | d2) & other:
                                    continue
                                # valid swap
                                cur_sol[r1] = row1
                                cur_sol[r2] = row2
                                cur_used = other | d1 | d2
                                cur_scope = max(cur_scope, max(sum(g) for g in cur_sol))
                                progress = True
                                improved_any = True
                                break
                            if progress:
                                break
                        if progress:
                            break
                    if progress:
                        break
            if not progress:
                break
        return cur_sol, cur_scope

    # Run local search on top candidates
    rng = random.Random(12345)
    for scope, sol in candidates[:6]:
        if not elapsed_ok():
            break
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # allocate a chunk of time per candidate
        chunk = min(18.0, remaining * 0.6)
        new_sol, new_scope = local_search(sol, rng, chunk)
        if new_scope < best_scope:
            best_scope = new_scope
            best_sol = new_sol
        # also try a fresh perturbation-based restart from best
        if elapsed_ok():
            for restart in range(3):
                if not elapsed_ok():
                    break
                pert = [g[:] for g in best_sol]
                # randomize a couple of gaps
                for _ in range(2):
                    rr = rng.randrange(n_rows)
                    pp = rng.randrange(k)
                    old = pert[rr][pp]
                    pert[rr][pp] = max(1, old + rng.choice([-2, -1, 1, 2]))
                # quick repair: rebuild row if invalid
                used = all_diffs(pert)
                # verify validity
                ok = True
                for g in pert:
                    d = row_diffs(g)
                    if len(d) != len(g) * (len(g) + 1) // 2:
                        ok = False
                # recompute scope
                ps = max(sum(g) for g in pert)
                if ps < best_scope and ok:
                    # try to polish
                    new_sol2, new_scope2 = local_search(pert, rng, min(8.0, max(1.0, deadline - time.time())))
                    if new_scope2 < best_scope:
                        best_scope = new_scope2
                        best_sol = new_sol2

    # Final: convert best gap solution to rows
    rows = [gaps_to_row(g) for g in best_sol]
    # sort each row (already increasing by construction)
    return rows
