"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _alpha_bnb(adj, n, timeout=10):
    best = [0]
    start_t = time.time()
    def bnb(cand, sz):
        if time.time() - start_t > timeout:
            return
        if sz + cand.bit_count() <= best[0]:
            return
        if cand == 0:
            if sz > best[0]:
                best[0] = sz
            return
        best_v = -1
        best_deg = -1
        v = 0
        tmp = cand
        while tmp:
            if tmp & 1:
                deg = (adj[v] & cand).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            tmp >>= 1
            v += 1
        v_bit = 1 << best_v
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1)
        bnb(cand & ~v_bit, sz)
    bnb((1 << n) - 1, 0)
    return best[0]

def _check_triangle_free(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                if adj[i] & adj[j]:
                    return False
    return True

def _try_cayley(S, n, timeout=8):
    S_set = set(S)
    adj = [0] * n
    for s in S_set:
        neg_s = (-s) % n
        for i in range(n):
            j1 = (i + s) % n
            j2 = (i + neg_s) % n
            if j1 != i:
                adj[i] |= (1 << j1)
            if j2 != i:
                adj[i] |= (1 << j2)
    for i in range(n):
        for j in range(n):
            if (adj[i] >> j) & 1:
                adj[j] |= (1 << i)
    if not _check_triangle_free(adj, n):
        return None
    alpha = _alpha_bnb(adj, n, timeout=timeout)
    if alpha < 10:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 108
    
    # Try n=40 first, then 39, etc.
    for n in [40, 39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break
        
        # Try various modular connection sets
        for mod in range(3, 12):
            if mod >= n:
                continue
            # Single residue
            for r in range(1, mod):
                S = set(s for s in range(1, n) if s % mod == r)
                if len(S) >= n // 5:
                    result = _try_cayley(S, n, timeout=5)
                    if result:
                        return result
            # Two residues
            for r1 in range(1, mod):
                for r2 in range(r1 + 1, mod):
                    S = set(s for s in range(1, n) if s % mod in (r1, r2))
                    if len(S) >= n // 4:
                        result = _try_cayley(S, n, timeout=5)
                        if result:
                            return result
    
    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
# EVOLVE-BLOCK-END
