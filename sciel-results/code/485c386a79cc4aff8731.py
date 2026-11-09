import time
import itertools
import random

_DEADLINE = time.time() + 135  # safety margin before 150s wall-clock kill


def _elapsed_ok():
    return time.time() < _DEADLINE


def _verify(n, colors):
    """Check that no colour class contains a C4 (two vertices with two common neighbours).
    Efficient bitset check. Returns True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # adjacency bitmasks per colour: adj[c][v] = bitmask of neighbours of v in colour c
    adj = [ [0]*n for _ in range(5) ]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]; idx += 1
            if c < 0 or c > 4:
                return False
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):  # at least two common neighbours -> C4
                    return False
    return True


def _cyclic(n, k):
    """Colour edge (i,j) by circular distance 1..k (k = n//2, requires n odd)."""
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            dist = min(d, n - d)
            out.append(dist - 1)
    return out


def _build_cyclic_ext(n):
    """Build a C4-free 5-colouring for n=26 using product structure."""
    if n == 26:
        colors = []
        for i in range(26):
            for j in range(i + 1, 26):
                a1, b1 = divmod(i, 13)
                a2, b2 = divmod(j, 13)
                d = (b2 - b1) % 13
                if d == 0:
                    d = 13
                if a1 == a2:
                    c = d % 5
                else:
                    c = (d + 2) % 5
                colors.append(c)
        return colors
    return None


def _local_search(n, colors, max_iter=100000):
    """Local search to eliminate C4 violations."""
    if _verify(n, colors):
        return colors
    
    random.seed(12345)
    
    cur = colors[:]
    
    def build_adj(cols, n):
        adj = [[0]*n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = cols[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj
    
    def count_violations(adj, n):
        cnt = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[c][u] & adj[c][v]
                    if common & (common - 1):
                        cnt += 1
        return cnt
    
    adj = build_adj(cur, n)
    cur_viol = count_violations(adj, n)
    
    for it in range(max_iter):
        if not _elapsed_ok():
            break
        if cur_viol == 0:
            return cur
        
        i = random.randint(0, n-2)
        j = random.randint(i+1, n-1)
        idx = 0
        for ii in range(n):
            for jj in range(ii+1, n):
                if ii == i and jj == j:
                    break
                idx += 1
            else:
                continue
            break
        
        old_c = cur[idx]
        best_c = -1
        best_v = cur_viol
        
        for new_c in range(5):
            if new_c == old_c:
                continue
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            v = count_violations(adj, n)
            
            adj[new_c][i] &= ~(1 << j)
            adj[new_c][j] &= ~(1 << i)
            adj[old_c][i] |= (1 << j)
            adj[old_c][j] |= (1 << i)
            
            if v < best_v:
                best_v = v
                best_c = new_c
        
        if best_c >= 0:
            cur[idx] = best_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[best_c][i] |= (1 << j)
            adj[best_c][j] |= (1 << i)
            cur_viol = best_v
    
    return cur


def _build():
    best = None
    best_n = 0

    # 1) cyclic construction for odd n
    for n in range(3, 29, 2):
        if not _elapsed_ok():
            break
        cols = _cyclic(n, n // 2)
        if _verify(n, cols):
            if n > best_n:
                best_n = n
                best = (n, cols)

    # 2) Try n=26 with product construction
    if _elapsed_ok():
        cols26 = _build_cyclic_ext(26)
        if cols26 is not None:
            if _verify(26, cols26):
                if 26 > best_n:
                    best_n = 26
                    best = (26, cols26)
            else:
                cols26_fix = _local_search(26, cols26, max_iter=50000)
                if _verify(26, cols26_fix):
                    if 26 > best_n:
                        best_n = 26
                        best = (26, cols26_fix)

    # 3) Try to extend to 27, 28
    if best is not None and _elapsed_ok():
        for target in range(best_n + 1, 29):
            if not _elapsed_ok():
                break
            if target % 2 == 1:
                cand = _cyclic(target, target // 2)
                if _verify(target, cand):
                    if target > best_n:
                        best_n = target
                        best = (target, cand)
                    continue
            random.seed(target)
            base_n, base_cols = best
            cand = base_cols[:]
            for i in range(base_n):
                cand.append(random.randint(0, 4))
            cand_fix = _local_search(target, cand, max_iter=30000)
            if _verify(target, cand_fix):
                if target > best_n:
                    best_n = target
                    best = (target, cand_fix)

    if best is None:
        n = 11
        cols = _cyclic(n, n // 2)
        best = (n, cols)

    return {'n': best[0], 'colors': best[1]}


def construct():
    try:
        return _build()
    except Exception:
        n = 11
        return {'n': n, 'colors': _cyclic(n, n // 2)}
