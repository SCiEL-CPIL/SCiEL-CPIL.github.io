"""Shortest superpermutation via Hamiltonian-path heuristic on the (n-1)-permutation graph."""
import time
from itertools import permutations as _itperm

_DEADLINE = 120.0
_EPS = 1.0 / 1000.0


def _time_left(limit):
    return limit - time.time()


def _build_perm_lists(n):
    if n <= 1:
        return [[1]]
    return [list(p) for p in _itperm(range(1, n + 1))]


def _overlap_suffix_prefix(a, b):
    """Maximum k such that suffix of a of length k equals prefix of b of length k."""
    m = min(len(a), len(b))
    for k in range(m, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def _try_greedy(cost, m, seed_idx, visited, deadline):
    """Greedy nearest-neighbour walk with light backtracking on ties.
    Returns (path, best_path) or None if timed out / dead-end."""
    visited = list(visited)
    path = [seed_idx]
    visited[seed_idx] = True
    best_path = list(path)

    def _rec():
        nonlocal best_path
        if len(path) == m:
            cur = list(path)
            if cur is not best_path:
                # keep the (single) full path; we only ever reach it once
                best_path = cur
            return True
        cur = path[-1]
        t = time.time()
        if t >= deadline:
            return False
        # candidate neighbours: unvisited with smallest cost
        best_cost = 10 ** 9
        cands = []
        for j in range(m):
            if not visited[j]:
                c = cost[cur][j]
                if c < best_cost:
                    best_cost = c
                    cands = [j]
                elif c == best_cost:
                    cands.append(j)
        # deterministic ordering; tiny backtrack over ties
        # shuffle for variety based on path parity
        if len(cands) > 1:
            # simple deterministic variation: rotate by path length
            r = (len(path) + seed_idx) % len(cands)
            cands = cands[r:] + cands[:r]
        for j in cands:
            path.append(j)
            visited[j] = True
            ok = _rec()
            visited[j] = False
            path.pop()
            if ok:
                return True
            if time.time() >= deadline:
                return False
        return False

    ok = _rec()
    return path, best_path, ok


def _construct_superpermutation(n, perm_list, order, m):
    """Given an ordering of (n-1)-permutations, build the superpermutation word.
    Each (n-1)-perm P becomes P + [n]. Consecutive blocks are overlapped maximally."""
    if m == 1:
        # single (n-1)-perm -> just append n
        return list(perm_list[order[0]]) + [n]
    word = list(perm_list[order[0]]) + [n]
    for idx in range(1, m):
        block = list(perm_list[order[idx]]) + [n]
        # find max overlap between suffix of word and prefix of block
        max_ov = min(len(word), len(block))
        ov = 0
        for k in range(max_ov, 0, -1):
            if word[-k:] == block[:k]:
                ov = k
                break
        word.extend(block[ov:])
    return word


def _is_valid_superperm(word, n):
    seen = set()
    for i in range(len(word) - n + 1):
        t = tuple(word[i:i + n])
        # check it's a permutation of 1..n
        if len(set(t)) == n:
            seen.add(t)
    need = 1
    for i in range(2, n + 1):
        need *= i
    return len(seen) == need


def construct(n):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _EPS

    if n == 1:
        return "1"
    if n == 2:
        # permutations of 1,2: 12 21 -> shortest: 121 (contains 12,21)
        return "121"
    if n == 3:
        # known shortest: 123121321 (length 9 = 1!+2!+3!)
        return "123121321"
    if n == 4:
        # known shortest length = 1!+2!+3!+4! = 33
        return _classic(n)

    # n >= 5: heuristic search
    perm_list = _build_perm_lists(n - 1)
    m = len(perm_list)  # (n-1)!

    # cost matrix: cost[i][j] = (n-1) - overlap(suffix of perm_i, prefix of perm_j)
    # (number of new symbols added when going from block i to block j, excluding the first)
    cost = [[0] * m for _ in range(m)]
    for i in range(m):
        pi = perm_list[i]
        for j in range(m):
            if i == j:
                cost[i][j] = n - 1  # not used directly
                continue
            ov = _overlap_suffix_prefix(pi, perm_list[j])
            cost[i][j] = (n - 1) - ov

    # ----- greedy initial solution -----
    best_order = None
    best_len = None

    # Try several seeds to get diversity
    seeds = [0]
    # add a few heuristic seeds
    seeds.append(1)
    if m > 2:
        seeds.append(m // 2)
    if m > 3:
        seeds.append(m - 1)

    for s in seeds:
        if time.time() >= deadline - 2.0:
            break
        visited = [False] * m
        res = _try_greedy(cost, m, s, visited, deadline - 2.0)
        if res is None:
            continue
        path, full_path, ok = res
        if ok and full_path is not None and len(full_path) == m:
            cand = full_path
        else:
            # dead-end; use the partial path and fill the rest in order (suboptimal)
            cand = list(path)
            remaining = [i for i in range(m) if i not in set(path)]
            cand.extend(remaining)
        word = _construct_superpermutation(n, perm_list, cand, m)
        L = len(word)
        if best_len is None or L < best_len:
            best_len = L
            best_order = cand

    # ----- local search: 2-opt and Or-opt -----
    def _path_length(order):
        # total length = len(first block) + sum of costs along path
        # first block length = n (perm + [n])
        total = n
        for k in range(1, len(order)):
            total += cost[order[k - 1]][order[k]]
        return total

    cur_order = best_order
    cur_len = _path_length(cur_order)

    def _try_2opt(order, limit):
        nonlocal cur_len
        improved = True
        while improved and time.time() < limit:
            improved = False
            m2 = len(order)
            for i in range(m2 - 1):
                a = order[i]
                b = order[i + 1]
                # current cost segment
                cost_before = cost[a][b] if i > 0 else 0
                cost_after = cost[order[m2 - 2]][order[m2 - 1]] if False else 0  # not used
                # For 2-opt on a path: reverse segment [i+1..j]
                # savings = (cost[order[i]][order[i+1]] + cost[order[j]][order[j+1]])
                #          - (cost[order[i]][order[j]] + cost[order[j+1]][order[i+1]])
                # (with appropriate handling at ends)
                for j in range(i + 1, m2):
                    # current: ...order[i], order[i+1]...order[j], order[j+1]...
                    c1 = cost[order[i]][order[i + 1]]
                    if j + 1 < m2:
                        c2 = cost[order[j]][order[j + 1]]
                        n1 = cost[order[i]][order[j]]
                        n2 = cost[order[j + 1]][order[i + 1]]
                        saving = (c1 + c2) - (n1 + n2)
                    else:
                        # j is last; reversing to end
                        c2 = 0
                        n1 = cost[order[i]][order[j]]
                        n2 = 0
                        saving = c1 - n1
                    if saving > 0:
                        # apply reversal
                        order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                        cur_len -= saving
                        improved = True
                        break
                if improved:
                    break
        return order

    def _try_oropt(order, limit):
        nonlocal cur_len
        improved = True
        while improved and time.time() < limit:
            improved = False
            m2 = len(order)
            for seglen in (1, 2, 3):
                if seglen >= m2:
                    continue
                for i in range(m2):
                    if i + seglen > m2:
                        continue
                    seg = order[i:i + seglen]
                    # current
                    c_left = cost[order[i - 1]][seg[0]] if i > 0 else 0
                    c_right = cost[seg[-1]][order[i + seglen]] if i + seglen < m2 else 0
                    # try inserting at position k (k != i, k != i+1)
                    best_saving = 0
                    best_k = -1
                    # k in [0, m2-seglen], k != i
                    for k in range(m2 - seglen + 1):
                        if k == i:
                            continue
                        # remove seg at i, then insert at k
                        # adjust k if k > i
                        k_adj = k
                        if k > i:
                            k_adj = k - seglen
                        # new neighbours
                        nl = cost[order[k_adj - 1]][seg[0]] if k_adj > 0 else 0
                        nr = cost[seg[-1]][order[k_adj + seglen]] if k_adj + seglen < m2 - seglen else 0
                        # careful: after removal, length is m2 - seglen
                        # position k_adj is where seg will be inserted
                        # left neighbour of seg: order[k_adj-1] (in the list with seg removed)
                        # right neighbour: order[k_adj + seglen] (in the list with seg removed)
                        # we don't actually remove; just compute
                        saving = (c_left + c_right) - (nl + nr)
                        if saving > best_saving:
                            best_saving = saving
                            best_k = k
                    if best_saving > 0 and best_k >= 0:
                        # perform move: remove seg at i, insert at best_k
                        new_order = order[:i] + order[i + seglen:]
                        k_adj = best_k
                        if best_k > i:
                            k_adj = best_k - seglen
                        new_order = new_order[:k_adj] + seg + new_order[k_adj:]
                        order[:] = new_order
                        cur_len -= best_saving
                        improved = True
                        break
                if improved:
                    break
        return order

    # iterate local search until no improvement or deadline
    ls_limit = deadline - 1.0
    while time.time() < ls_limit:
        prev_len = cur_len
        _try_2opt(cur_order, ls_limit)
        if cur_len < prev_len:
            prev_len = cur_len
        _try_oropt(cur_order, ls_limit)
        if cur_len >= prev_len:
            break

    # build final word
    word = _construct_superpermutation(n, perm_list, cur_order, m)

    # safety: if somehow invalid, fall back to classic
    if not _is_valid_superperm(word, n):
        word = _classic(n)

    return "".join(map(str, word))


def _classic(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                           if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
