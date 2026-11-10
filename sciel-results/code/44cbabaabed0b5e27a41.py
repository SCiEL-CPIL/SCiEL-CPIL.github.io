import time
from math import ceil

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for e in edges:
        adj[e[0]] |= 1 << e[1]
        adj[e[1]] |= 1 << e[0]
    
    total_masks = 1 << n
    full = total_masks - 1
    
    pc = [0] * total_masks
    for i in range(1, total_masks):
        pc[i] = pc[i >> 1] + (i & 1)
    
    cliq = [False] * total_masks
    indep = [False] * total_masks
    cliq[0] = True
    indep[0] = True
    
    for m in range(1, total_masks):
        lo = m & (-m)
        v = lo.bit_length() - 1
        rest = m ^ lo
        if cliq[rest] and (rest & ~adj[v]) == 0:
            cliq[m] = True
        if indep[rest] and (rest & adj[v]) == 0:
            indep[m] = True
    
    good = [False] * total_masks
    for i in range(1, total_masks):
        good[i] = cliq[i] or indep[i]
    
    maxsz = [0] * total_masks
    good_list = [m for m in range(1, total_masks) if good[m]]
    for s in good_list:
        sz = pc[s]
        comp = full ^ s
        sub = comp
        while True:
            m = s | sub
            if sz > maxsz[m]:
                maxsz[m] = sz
            if sub == 0:
                break
            sub = (sub - 1) & comp
    
    parts_for_v = [[] for _ in range(n)]
    for m in good_list:
        sz = pc[m]
        t = m
        while t:
            lsb = t & (-t)
            v = lsb.bit_length() - 1
            parts_for_v[v].append((sz, m))
            t ^= lsb
    for v in range(n):
        parts_for_v[v].sort(key=lambda x: -x[0])
    
    gm = maxsz[full]
    lb0 = ceil(n / gm) if gm > 0 else n
    
    t0 = time.time()
    cutoff = 82.0
    
    memo_fail = {}
    
    def feasible(rem, used, limit):
        if rem == 0:
            return True
        if used == limit:
            return False
        rs = pc[rem]
        mp = maxsz[rem]
        if mp == 0:
            return False
        if used + ceil(rs / mp) > limit:
            return False
        k = (rem, used)
        if k in memo_fail:
            return False
        bv = -1
        mc = 1 << 30
        bc = []
        t = rem
        while t:
            lsb = t & (-t)
            v = lsb.bit_length() - 1
            t ^= lsb
            cnt = 0
            cl = []
            for sz, S in parts_for_v[v]:
                if (S & rem) == S:
                    cnt += 1
                    cl.append(S)
            if cnt < mc:
                mc = cnt
                bv = v
                bc = cl
                if cnt == 0:
                    break
        if bv < 0 or mc == 0:
            memo_fail[k] = True
            return False
        for S in bc:
            if feasible(rem ^ S, used + 1, limit):
                return True
        memo_fail[k] = True
        return False
    
    def build(rem, used, limit):
        if rem == 0:
            return []
        if used == limit:
            return None
        rs = pc[rem]
        mp = maxsz[rem]
        if mp == 0:
            return None
        if used + ceil(rs / mp) > limit:
            return None
        bv = -1
        mc = 1 << 30
        bc = []
        t = rem
        while t:
            lsb = t & (-t)
            v = lsb.bit_length() - 1
            t ^= lsb
            cnt = 0
            cl = []
            for sz, S in parts_for_v[v]:
                if (S & rem) == S:
                    cnt += 1
                    cl.append(S)
            if cnt < mc:
                mc = cnt
                bv = v
                bc = cl
                if cnt == 0:
                    break
        if bv < 0 or mc == 0:
            return None
        for S in bc:
            sub = build(rem ^ S, used + 1, limit)
            if sub is not None:
                vs = []
                t = S
                while t:
                    lsb = t & (-t)
                    vs.append(lsb.bit_length() - 1)
                    t ^= lsb
                return [vs] + sub
        return None
    
    for k in range(lb0, n + 1):
        if time.time() - t0 > cutoff:
            break
        memo_fail.clear()
        if feasible(full, 0, k):
            res = build(full, 0, k)
            if res is not None:
                out = []
                for part in res:
                    out.append([int(x) for x in part])
                return out
    
    return _fallback()

def _fallback():
    n = P['n']
    es = set()
    for u, v in P['edges']:
        es.add((min(u, v), max(u, v)))
    parts = []
    for v in range(n):
        ok = False
        for p in parts:
            ci = True
            ii = True
            for u in p:
                e = (min(u, v), max(u, v))
                if e in es:
                    ii = False
                else:
                    ci = False
            if ci or ii:
                p.append(v)
                ok = True
                break
        if not ok:
            parts.append([v])
    return parts
