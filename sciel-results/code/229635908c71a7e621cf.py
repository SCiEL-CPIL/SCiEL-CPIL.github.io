import time
from collections import defaultdict

n, m, p = 4, 4, 6
AM = n * m   # 16
BM = m * p   # 24
CM = p * n   # 24

# Strassen 2x2x2 core. Each term: (x, y, z) sparse dicts over local 2x2 coords (r,c)
STR = [
    ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}),
    ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):1}),
    ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
    ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
    ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):1,(1,0):1}),
    ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,0):1,(1,1):1}),
    ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,1):1,(1,1):1}),
]

def _prune(d):
    return {k: c for k, c in d.items() if c != 0}

def _add(dst, d, s):
    for k, c in d.items():
        dst[k] = dst.get(k, 0) + s * c

def _scale(d, s):
    return {k: c * s for k, c in d.items() if c != 0}

def _key(d):
    return tuple(sorted(d.items()))

def construct():
    start = time.time()
    deadline = start + 138.0
    terms = []

    # ---- Part A: 4x4 x 4x4 -> 4x4 (columns 0..3) ----
    # 2x2 big-block Kronecker of Strassen.
    # big A block (br,bc) local (r,c) -> A[(2br+r)*4 + (2bc+c)]
    # big B block (br,bc) local (r,c) -> B[(2br+r)*6 + (2bc+c)]
    # big C block (br,bc) local (r,c) -> C[(2bc+c)*4 + (2br+r)]
    for bi, (xg, yg) in enumerate([(STR[s][0], STR[s][1]) for s in range(7)]):
        for s in range(7):
            if time.time() > deadline:
                break
            xs, ys, zs = STR[s]
            u = {}; v = {}; w = {}
            for (br, bc), sg in xg.items():
                for (r, c), xc in xs.items():
                    u[(2*br + r)*4 + (2*bc + c)] = u.get((2*br + r)*4 + (2*bc + c), 0) + sg*xc
            for (br, bc), sg in yg.items():
                for (r, c), yc in ys.items():
                    v[(2*br + r)*6 + (2*bc + c)] = v.get((2*br + r)*6 + (2*bc + c), 0) + sg*yc
            for (br, bc), sc in zs.items():
                for (r, c), zc in STR[bi][2].items():
                    w[(2*bc + c)*4 + (2*br + r)] = w.get((2*bc + c)*4 + (2*br + r), 0) + sc*zc
            terms.append((_prune(u), _prune(v), _prune(w)))

    # ---- Part B: 4x4 x 4x2 -> 4x2 (columns 4..5) ----
    # 4 blocks: A(2br,2bc) x B(2br, 4+2bc) -> C(4+2bc, 2br)
    for br in range(2):
        for bc in range(2):
            for s in range(7):
                if time.time() > deadline:
                    break
                xs, ys, zs = STR[s]
                u = {}; v = {}; w = {}
                for (r, c), xc in xs.items():
                    u[(2*br + r)*4 + (2*bc + c)] = u.get((2*br + r)*4 + (2*bc + c), 0) + xc
                for (r, c), yc in ys.items():
                    v[(2*br + r)*6 + (4 + 2*bc + c)] = v.get((2*br + r)*6 + (4 + 2*bc + c), 0) + yc
                for (r, c), zc in zs.items():
                    w[(4 + 2*bc + c)*4 + (2*br + r)] = w.get((4 + 2*bc + c)*4 + (2*br + r), 0) + zc
                terms.append((_prune(u), _prune(v), _prune(w)))

    # ---- Merge: combine terms sharing identical (u,v) key by summing w ----
    def merge(terms_in):
        groups = defaultdict(lambda: defaultdict(float))
        for u, v, w in terms_in:
            key = (_key(u), _key(v))
            for k, c in w.items():
                groups[key][k] += c
        out = []
        for (uk, vk), wsum in groups.items():
            wsum = _prune(wsum)
            if wsum:
                out.append((dict(uk), dict(vk), wsum))
        return out

    cur = terms
    prev = len(cur)
    for _ in range(50):
        if time.time() > deadline:
            break
        cur = merge(cur)
        if len(cur) >= prev:
            break
        prev = len(cur)

    rank = len(cur)
    U = [[0.0]*rank for _ in range(AM)]
    V = [[0.0]*rank for _ in range(BM)]
    W = [[0.0]*rank for _ in range(CM)]
    for col, (u, v, w) in enumerate(cur):
        for i, c in u.items():
            U[i][col] = c
        for i, c in v.items():
            V[i][col] = c
        for i, c in w.items():
            W[i][col] = c

    return {"U": U, "V": V, "W": W}
