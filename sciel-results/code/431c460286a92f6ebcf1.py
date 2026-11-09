import itertools, time, random

N = 16
_eight_sets = list(itertools.combinations(range(N), 8))
_pos_splits = list(itertools.combinations(range(8), 4))
_set_masks = []
for s in _eight_sets:
    lst = []
    for a_pos in _pos_splits:
        a_pos_set = set(a_pos)
        Am = 0; Bm = 0
        for i in range(8):
            if i in a_pos_set:
                Am |= (1 << s[i])
            else:
                Bm |= (1 << s[i])
        lst.append((Am, Bm))
    _set_masks.append(lst)

def _has_q3(mask):
    for lst in _set_masks:
        for Am, Bm in lst:
            a = Am
            while a:
                lsb = a & (-a)
                v = lsb.bit_length() - 1
                if (mask[v] & Bm).bit_count() < 3:
                    break
                a -= lsb
            else:
                b = Bm
                while b:
                    lsb = b & (-b)
                    v = lsb.bit_length() - 1
                    if (mask[v] & Am).bit_count() < 3:
                        break
                    b -= lsb
                else:
                    return True
    return False

def _to_edges(mask):
    out = []
    for u in range(N):
        m = mask[u]
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if v > u:
                out.append([u, v])
            m -= lsb
    return out

def _edge_count(mask):
    return sum(mask[u].bit_count() for u in range(N)) // 2

def construct():
    t0 = time.time()
    DEADLINE = 150.0 - 5.0
    rng = random.Random(12345)

    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]

    def greedy_build():
        mask = [0] * N
        edges = all_edges[:]
        rng.shuffle(edges)
        for (u, v) in edges:
            if time.time() > DEADLINE - 30:
                break
            mask[u] |= (1 << v); mask[v] |= (1 << u)
            if _has_q3(mask):
                mask[u] ^= (1 << v); mask[v] ^= (1 << u)
        return mask

    def local_search(mask):
        improved = True
        while improved and time.time() < DEADLINE - 8:
            improved = False
            # add moves
            for (u, v) in all_edges:
                if time.time() > DEADLINE - 8:
                    break
                if (mask[u] >> v) & 1:
                    continue
                mask[u] |= (1 << v); mask[v] |= (1 << u)
                if not _has_q3(mask):
                    improved = True
                else:
                    mask[u] ^= (1 << v); mask[v] ^= (1 << u)
            # swap moves: remove (a,b), add (u,v)
            for (a, b) in all_edges:
                if time.time() > DEADLINE - 8:
                    break
                if not ((mask[a] >> b) & 1):
                    continue
                for (u, v) in all_edges:
                    if time.time() > DEADLINE - 8:
                        break
                    if (mask[u] >> v) & 1:
                        continue
                    # remove a-b, add u-v
                    mask[a] ^= (1 << b); mask[b] ^= (1 << a)
                    mask[u] |= (1 << v); mask[v] |= (1 << u)
                    if not _has_q3(mask):
                        improved = True
                    else:
                        mask[u] ^= (1 << v); mask[v] ^= (1 << u)
                        mask[a] |= (1 << b); mask[b] |= (1 << a)
                    if time.time() > DEADLINE - 8:
                        break
        return mask

    best_mask = None
    best_cnt = -1
    # multiple restarts
    while time.time() < DEADLINE - 20:
        mask = greedy_build()
        if time.time() < DEADLINE - 15:
            mask = local_search(mask)
        cnt = _edge_count(mask)
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = mask[:]
        if time.time() > DEADLINE - 10:
            break

    if best_mask is None:
        # fallback K3,13
        best_mask = [0] * N
        for a in range(3):
            for b in range(3, 16):
                best_mask[a] |= (1 << b); best_mask[b] |= (1 << a)
    return _to_edges(best_mask)
