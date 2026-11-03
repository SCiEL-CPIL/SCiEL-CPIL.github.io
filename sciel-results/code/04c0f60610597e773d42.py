import numpy as np
import time

N = 50
DEADLINE = 118.0

def _radius(c):
    d = np.sqrt(((c[:,None,:]-c[None,:,:])**2).sum(-1))
    np.fill_diagonal(d, np.inf)
    rp = d.min()/2
    rw = np.minimum(np.minimum(0.5-c[:,0], c[:,0]+0.5),
                    np.minimum(0.5-c[:,1], c[:,1]+0.5)).min()
    return min(rp, rw)

def _relax(c, R, iters, eta0, eta1):
    c = c.copy()
    n = len(c)
    for it in range(iters):
        t = it/iters
        eta = eta0 + (eta1-eta0)*t
        diff = c[:,None,:]-c[None,:,:]
        d = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2*R - d, 0.0)/d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:,:,None]*diff).sum(1)
        # walls
        f[:,0] += -np.maximum(c[:,0]-(0.5-R),0.0) + np.maximum((-0.5+R)-c[:,0],0.0)
        f[:,1] += -np.maximum(c[:,1]-(0.5-R),0.0) + np.maximum((-0.5+R)-c[:,1],0.0)
        c += eta*f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0):
    c = start.copy()
    best = (c.copy(), _radius(c))
    R = R0
    while R <= R1 and (time.time()-t0) < DEADLINE:
        c = _relax(c, R, iters, eta0, eta1)
        r = _radius(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time()-t0) > DEADLINE-5:
            break
    return best

def _init_7x7_plus():
    ticks = (np.arange(7)+0.5)/7 - 0.5
    g = np.array([(x,y) for y in ticks for x in ticks])
    # add 50th near a gap
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    ticks = (np.arange(8)+0.5)/8 - 0.5
    g = np.array([(x,y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03,0.03,c.shape)
    return c

def _init_rand(rng):
    return rng.uniform(-0.45,0.45,(N,2))

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    inits = [_init_7x7_plus(), _init_jitter8(rng), _init_rand(rng)]
    best = None
    for start in inits:
        if (time.time()-t0) > DEADLINE-10:
            break
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1500, 0.02, 0.002, t0)
        if best is None or r > best[1]:
            best = (c, r)
    if best is None:
        best = (_init_7x7_plus(), 0.0)
    c = best[0]
    # final fine tune: a few more relax at current radius
    r = _radius(c)
    c = _relax(c, r, 2000, 0.01, 0.0005)
    if _radius(c) >= r:
        c = c
    return c
