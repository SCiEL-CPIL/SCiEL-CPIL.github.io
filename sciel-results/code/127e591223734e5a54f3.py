import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s kill

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            m = bitmasks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            mi = bitmasks[i]
            for j in range(i + 1, n):
                if not (mi & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            m = bitmasks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def is_add_valid(bitmasks, u, v):
        """
        Fast check: adding (u,v) keeps graph K3,3-free.
        Only triples {u,i,j} with i,j in N(v)\{u} and {v,i,j} with i,j in N(u)\{v}
        can become invalid, because those are the only triples whose common-neighbor
        count changes (gains v or u respectively).
        """
        new_common_u = bitmasks[v] & ~(1 << u)   # i,j in N(v)\{u}: triple {u,i,j}
        new_common_v = bitmasks[u] & ~(1 << v)   # i,j in N(u)\{v}: triple {v,i,j}

        # Triples {u, i, j}: common = N(u)∪{v} ∩ N(i) ∩ N(j)
        m = new_common_u
        nc_u = []
        while m:
            lsb = m & -m
            nc_u.append(lsb.bit_length() - 1)
            m -= lsb
        for ii in range(len(nc_u)):
            i = nc_u[ii]
            ai = bitmasks[i]
            for jj in range(ii + 1, len(nc_u)):
                j = nc_u[jj]
                common = (bitmasks[u] | (1 << v)) & ai & bitmasks[j]
                if common.bit_count() >= 3:
                    return False

        # Triples {v, i, j}: common = N(v)∪{u} ∩ N(i) ∩ N(j)
        m = new_common_v
        nc_v = []
        while m:
            lsb = m & -m
            nc_v.append(lsb.bit_length() - 1)
            m -= lsb
        for ii in range(len(nc_v)):
            i = nc_v[ii]
            ai = bitmasks[i]
            for jj in range(ii + 1, len(nc_v)):
                j = nc_v[jj]
                common = (bitmasks[v] | (1 << u)) & ai & bitmasks[j]
                if common.bit_count() >= 3:
                    return False

        return True

    def has_k33(bitmasks):
        """Full validity check (used to verify swap results)."""
        for i in range(n):
            ai = bitmasks[i]
            for j in range(i + 1, n):
                common_ij = ai & bitmasks[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (common_ij & bitmasks[k]).bit_count() >= 3:
                        return False
        return True

    def greedy_add_exhaustive(bitmasks):
        """Repeatedly add any valid non-edge until none remain."""
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search(initial_bitmasks):
        """Greedy add + swap moves (1-for-1, 1-for-2) to escape local optima."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local = bitmasks[:]

        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local = bitmasks[:]

        while time.time() < deadline:
            moved = False

            # Pass 1: add any valid non-edge
            non_edges = get_non_edges(bitmasks)
            if non_edges:
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local = bitmasks[:]

            if not moved:
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    # 1-for-1 swap
                    for _ in range(60):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if has_k33(bm2):
                                bitmasks = bm2
                                current_edges = count_edges(bitmasks)
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local = bitmasks[:]
                                break
                    # 1-for-2 swap
                    if not moved and len(non_edges) >= 2:
                        for _ in range(40):
                            if time.time() > deadline:
                                break
                            a, b = random.choice(all_edges)
                            c, d = random.choice(non_edges)
                            e, f = random.choice(non_edges)
                            if (c, d) == (e, f) or (c, d) == (f, e):
                                continue
                            bm2 = bitmasks[:]
                            bm2[a] &= ~(1 << b)
                            bm2[b] &= ~(1 << a)
                            if is_add_valid(bm2, c, d):
                                bm2[c] |= (1 << d)
                                bm2[d] |= (1 << c)
                            if is_add_valid(bm2, e, f):
                                bm2[e] |= (1 << f)
                                bm2[f] |= (1 << e)
                            if has_k33(bm2):
                                bitmasks = bm2
                                current_edges = count_edges(bitmasks)
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local = bitmasks[:]
                                break

            if not moved:
                # No improvement possible from current state; stop this local search
                break

        return best_local, best_local_edges

    def branch_and_bound_polish(bitmasks):
        """
        Treat the set of non-edges as a candidate pool. Find a large subset of
        non-edges that can ALL be added simultaneously (a K3,3-free completion).
        Use branch-and-bound with a greedy upper bound: a non-edge (u,v) is
        'conflicting' with (x,y) if adding both would be invalid. We want a
        maximal conflict-free subset. Upper bound = number of remaining candidates.
        """
        candidates = get_non_edges(bitmasks)
        if not candidates:
            return bitmasks, 0

        # Precompute for each candidate whether it's individually addable
        addable = []
        for (u, v) in candidates:
            if is_add_valid(bitmasks, u, v):
                addable.append((u, v))
        candidates = addable
        if not candidates:
            return bitmasks, 0

        best_extra = 0
        best_subset = []

        def compatible(base, u, v, x, y):
            """Can (x,y) be added on top of base which already has (u,v)?"""
            bm = base[:]
            bm[u] |= (1 << v)
            bm[v] |= (1 << u)
            return is_add_valid(bm, x, y)

        def branch(idx, chosen, base):
            nonlocal best_extra, best_subset
            if time.time() > deadline:
                return
            # Upper bound: chosen + remaining candidates
            ub = len(chosen) + (len(candidates) - idx)
            if ub <= best_extra:
                return
            if idx == len(candidates):
                if len(chosen) > best_extra:
                    best_extra = len(chosen)
                    best_subset = chosen[:]
                return
            u, v = candidates[idx]
            # Try including (u,v)
            if is_add_valid(base, u, v):
                nb = base[:]
                nb[u] |= (1 << v)
                nb[v] |= (1 << u)
                # Check compatibility with all previously chosen
                ok = True
                for (cu, cv) in chosen:
                    if not compatible(base, u, v, cu, cv):
                        ok = False
                        break
                if ok:
                    chosen.append((u, v))
                    branch(idx + 1, chosen, nb)
                    chosen.pop()
            # Try excluding (u,v)
            branch(idx + 1, chosen, base)

        branch(0, [], bitmasks)

        # Apply best_subset
        result = bitmasks[:]
        for (u, v) in best_subset:
            result[u] |= (1 << v)
            result[v] |= (1 << u)
        return result, best_extra

    # --- Main search loop ---
    best_adj = [0] * n
    best_count = 0

    # Initial greedy fill from empty
    cur = [0] * n
    cur = greedy_add_exhaustive(cur)
    c = count_edges(cur)
    if c > best_count:
        best_count = c
        best_adj = cur[:]

    no_improve = 0
    while time.time() < deadline:
        cur = best_adj[:]
        # Adaptive removal kick
        if no_improve < 10:
            num_removals = random.randint(1, 3)
        elif no_improve < 30:
            num_removals = random.randint(2, 6)
        else:
            num_removals = random.randint(3, 10)

        edges = get_all_edges(cur)
        if not edges:
            break
        random.shuffle(edges)
        for (u, v) in edges[:num_removals]:
            cur[u] &= ~(1 << v)
            cur[v] &= ~(1 << u)

        cur, c = local_search(cur)
        if c > best_count:
            best_count = c
            best_adj = cur[:]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 50:
            # Full restart
            cur = [0] * n
            cur = greedy_add_exhaustive(cur)
            c = count_edges(cur)
            if c > best_count:
                best_count = c
                best_adj = cur[:]
            no_improve = 0

    # Final polish: branch-and-bound over non-edges of best_adj
    if time.time() < deadline:
        polished, extra = branch_and_bound_polish(best_adj)
        if has_k33(polished):
            pc = count_edges(polished)
            if pc > best_count:
                best_count = pc
                best_adj = polished

    # Final exhaustive greedy add pass
    if time.time() < deadline:
        best_adj = greedy_add_exhaustive(best_adj)
        best_count = count_edges(best_adj)

    return bitmasks_to_edges(best_adj)
