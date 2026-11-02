N = 100
DW = 2.0 / N

import numpy as np
import time

def _overlaps(h):
    """overlap[k] = DW * sum_{j=0}^{N-k-1} h[j]*(1-h[j+k]) for k=0..N-1"""
    out = np.empty(N)
    for k in range(N):
        v = N - k
        out[k] = DW * np.dot(h[:v], 1.0 - h[k:])
    return out

def _max_overlap(h):
    return float(np.max(_overlaps(h)))

def _soft_max_and_grad(h, beta):
    ov = _overlaps(h)
    m = ov.max()
    e = np.exp(beta * (ov - m))
    s = e.sum()
    p = e / s
    sm = (m + np.log(s)) / beta
    grad = np.zeros(N)
    for k in range(N):
        pk = p[k]
        if pk < 1e-12:
            continue
        v = N - k
        grad[:v] += pk * DW * (1.0 - h[k:k+v])
        grad[k:] += pk * DW * (-h[:v])
    return sm, grad

def _project(h):
    """Project to sum(h)=N/2, 0<=h<=1 via bisection on lambda."""
    target = N / 2.0
    # bisection on shift
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        s = np.sum(np.clip(h + mid, 0.0, 1.0))
        if s > target:
            hi = mid
        else:
            lo = mid
    lam = 0.5 * (lo + hi)
    hp = np.clip(h + lam, 0.0, 1.0)
    # exact fix
    res = target - hp.sum()
    if abs(res) > 1e-12:
        if res > 0:
            room = 1.0 - hp
            idx = np.where(room > 1e-9)[0]
        else:
            room = hp
            idx = np.where(room > 1e-9)[0]
        if len(idx):
            w = room[idx]; w /= w.sum()
            hp[idx] += res * w
    hp = np.clip(hp, 0.0, 1.0)
    # final exact
    res = target - hp.sum()
    if abs(res) > 1e-14:
        j = int(np.argmin(np.abs(hp - 0.5)))
        hp[j] = np.clip(hp[j] + res, 0.0, 1.0)
    return hp

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    rng = np.random.default_rng(7)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)  # 0.5
    
    def inits():
        # constant
        yield np.full(N, 0.5)
        # sinusoidal patterns
        for f in [1, 2, 3, 5]:
            for A in [0.2, 0.4, 0.49]:
                i = np.arange(N)
                yield 0.5 + A * np.sin(2*np.pi*f*i/N)
        # random
        for _ in range(6):
            r = rng.random(N)
            yield r / r.sum() * (N/2.0)
    
    for h0 in inits():
        if time.time() > deadline - 20.0:
            break
        h = _project(h0.copy())
        cur = _max_overlap(h)
        if cur < best_score:
            best_score, best_h = cur, h.copy()
        
        # Adam
        m = np.zeros(N); v = np.zeros(N); t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        base_lr = 0.02
        for it in range(400):
            if time.time() > deadline - 5.0:
                break
            beta = 5.0 + 145.0 * (it / 400.0)
            sm, grad = _soft_max_and_grad(h, beta)
            t += 1
            m = b1*m + (1-b1)*grad
            v = b2*v + (1-b2)*grad*grad
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            lr = base_lr / (1.0 + 0.02*it)
            h_new = h - lr * mh / (np.sqrt(vh) + eps)
            h_new = _project(h_new)
            ns = _max_overlap(h_new)
            if ns < cur:
                cur = ns; h = h_new
                if cur < best_score:
                    best_score, best_h = cur, h.copy()
            else:
                # revert; occasionally perturb
                if rng.random() < 0.1:
                    i, j = rng.choice(N, 2, replace=False)
                    d = rng.uniform(-0.1, 0.1)
                    ht = h.copy(); ht[i]+=d; ht[j]-=d
                    ht = _project(ht)
                    ts = _max_overlap(ht)
                    if ts < cur:
                        cur = ts; h = ht
                        if cur < best_score:
                            best_score, best_h = cur, h.copy()
    
    # subgradient polish on best
    h = best_h.copy()
    cur = _max_overlap(h)
    for it in range(2000):
        if time.time() > deadline - 2.0:
            break
        ov = _overlaps(h)
        k = int(np.argmax(ov))
        v = N - k
        g = np.zeros(N)
        g[:v] += DW * (1.0 - h[k:k+v])
        g[k:] += DW * (-h[:v])
        lr = 0.005 / (1.0 + 0.001*it)
        h_new = _project(h - lr * g)
        ns = _max_overlap(h_new)
        if ns < cur:
            cur = ns; h = h_new
            if cur < best_score:
                best_score, best_h = cur, h.copy()
    
    return _project(best_h)
