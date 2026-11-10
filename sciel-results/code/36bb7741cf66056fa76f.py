"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM 4-5 band gap."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 210.0

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def _build_inv_eps_G(cell, N=17):
    size = len(cell)
    eps_low, eps_high = 1.0, 12.0
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i, j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
    half = (N - 1) // 2
    G_vals = np.arange(-half, half + 1)
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_G = np.zeros((N, N), dtype=complex)
    for im, m in enumerate(G_vals):
        for in_, n in enumerate(G_vals):
            inv_eps_G[im, in_] = inv_eps_fft[m % size, n % size] / (size * size)
    return inv_eps_G, G_vals

def _tm_bands(kx, ky, inv_eps_G, G_vals, N=17, n_bands=5):
    half = (N - 1) // 2
    num = N * N
    kg = np.zeros((num, 2))
    idx = 0
    for m in G_vals:
        for n in G_vals:
            kg[idx, 0] = kx + m
            kg[idx, 1] = ky + n
            idx += 1
    dot = kg @ kg.T
    M = np.zeros((num, num), dtype=complex)
    for i in range(num):
        mi = int(round(kg[i, 0] - kx))
        ni = int(round(kg[i, 1] - ky))
        for j in range(num):
            mj = int(round(kg[j, 0] - kx))
            nj = int(round(kg[j, 1] - ky))
            dm, dn = mi - mj, ni - nj
            if -half <= dm <= half and -half <= dn <= half:
                M[i, j] = dot[i, j] * inv_eps_G[dm + half, dn + half]
    M = (M + M.conj().T) / 2
    try:
        ev = np.linalg.eigvalsh(M.real)
    except:
        return None
    ev = np.sort(ev)
    if len(ev) < n_bands:
        return None
    return np.sqrt(np.maximum(ev[:n_bands], 0))

def _evaluate(cell, N=17):
    if _check_deadline():
        return -1.0
    inv_eps_G, G_vals = _build_inv_eps_G(cell, N)
    kpts = [(0,0), (0.5,0), (0.5,0.5)]
    for t in np.linspace(0.1, 0.4, 5):
        kpts += [(t,0), (t,t), (0.5,t)]
    for t1 in [0.1, 0.2, 0.3]:
        for t2 in [0.05, 0.1, 0.15, 0.2]:
            if t1 <= t2 and t1+t2 <= 0.5:
                kpts.append((t1,t2))
    kpts = kpts[:30]
    max4, min5 = 0.0, float('inf')
    for kx, ky in kpts:
        b = _tm_bands(kx, ky, inv_eps_G, G_vals, N, 5)
        if b is None or len(b) < 5:
            continue
        max4 = max(max4, b[3])
        min5 = min(min5, b[4])
    if min5 == float('inf'):
        return -1.0
    return 2*(min5-max4)/(min5+max4)

def _symmetrize(cell):
    size = len(cell)
    c = [r[:] for r in cell]
    for i in range(size):
        for j in range(size):
            c[i][j] = (c[i][j] + c[size-1-i][size-1-j] + c[i][size-1-j] + c[size-1-i][j]) / 4
    for i in range(size):
        for j in range(size):
            c[i][j] = 1 if c[i][j] >= 0.5 else 0
    return c

def design(problem):
    global _START_TIME
    _START_TIME = time.time()
    size = problem["pixels"]
    
    def circle(r):
        c = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x, y = (i+0.5)/size-0.5, (j+0.5)/size-0.5
                if x*x+y*y < r*r:
                    c[i][j] = 1
        return c
    
    def square(s):
        c = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x, y = (i+0.5)/size-0.5, (j+0.5)/size-0.5
                if abs(x) < s/2 and abs(y) < s/2:
                    c[i][j] = 1
        return c
    
    def cross(w, l):
        c = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x, y = (i+0.5)/size-0.5, (j+0.5)/size-0.5
                if (abs(x) < w/2 and abs(y) < l/2) or (abs(y) < w/2 and abs(x) < l/2):
                    c[i][j] = 1
        return c
    
    def dielectric_bg(shape_fn, **kw):
        c = [[1]*size for _ in range(size)]
        inner = shape_fn(**kw)
        for i in range(size):
            for j in range(size):
                if inner[i][j]:
                    c[i][j] = 0
        return c
    
    candidates = [
        ("c03", circle(0.3)),
        ("c035", circle(0.35)),
        ("c04", circle(0.4)),
        ("sq04", square(0.4)),
        ("sq05", square(0.5)),
        ("sq06", square(0.6)),
        ("cr", cross(0.2, 0.5)),
        ("cr2", cross(0.15, 0.45)),
        ("dbg_sq", dielectric_bg(square, s=0.4)),
        ("dbg_c", dielectric_bg(circle, r=0.25)),
        ("dbg_cr", dielectric_bg(cross, w=0.15, l=0.45)),
    ]
    
    best_score, best_cell = -1.0, None
    for name, cell in candidates:
        if _check_deadline():
            break
        cell = _symmetrize(cell)
        sc = _evaluate(cell)
        if sc > best_score:
            best_score, best_cell = sc, cell
    
    if best_cell is None:
        return _symmetrize(circle(0.2))
    
    # Local search
    cur = [r[:] for r in best_cell]
    cur_sc = best_score
    for it in range(30):
        if _check_deadline():
            break
        improved = False
        for i in range(size):
            if _check_deadline():
                break
            for j in range(size):
                if _check_deadline():
                    break
                old = cur[i][j]
                syms = [(i,j),(size-1-i,size-1-j),(i,size-1-j),(size-1-i,j)]
                if len(set(cur[a][b] for a,b in syms)) > 1:
                    continue
                for a,b in syms:
                    cur[a][b] = 1 - old
                sc = _evaluate(cur)
                if sc > cur_sc:
                    cur_sc = sc
                    improved = True
                    if sc > best_score:
                        best_score, best_cell = sc, [r[:] for r in cur]
                else:
                    for a,b in syms:
                        cur[a][b] = old
        if not improved:
            break
    
    return best_cell
