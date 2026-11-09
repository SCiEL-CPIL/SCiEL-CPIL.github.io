import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

_DEADLINE = 600.0  # safety margin under 630

def _elapsed():
    return time.time() - _START

def _verify(domain, values, K):
    # values clipped, f(1) adjusted
    vals = np.clip(values, -10, 10)
    # find index of 1
    S = np.sum(vals / domain.astype(float))
    # adjust f(1)
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        vals = vals.copy()
        vals[idx1] = vals[idx1] - S
    else:
        # f(1) = -S added
        vals = np.concatenate([vals, [-S]])
        domain = np.concatenate([domain, [1]])
    # check M(x) <= 1.0001 for x in [1,10K]
    d = domain.astype(float)
    v = vals
    # M(x) = sum v * floor(x/d)
    maxK = 10 * K
    # chunk
    bad = False
    chunk = 2000
    for x0 in range(1, maxK+1, chunk):
        x1 = min(x0+chunk, maxK+1)
        xs = np.arange(x0, x1, dtype=float)
        # floor matrix (chunk x m)
        # m could be up to K
        fl = np.floor(xs[:, None] / d[None, :])
        M = fl @ v
        if np.any(M > 1.0001):
            return False
    return True

def _compute_a(domain, values):
    vals = np.clip(values, -10, 10)
    S = np.sum(vals / domain.astype(float))
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        vals = vals.copy()
        vals[idx1] = vals[idx1] - S
    else:
        vals = np.concatenate([vals, [-S]])
        domain = np.concatenate([domain, [1]])
    a = -np.sum(vals * np.log(domain.astype(float)) / domain.astype(float))
    return a

def prime_number_theorem_partial_function():
    global _START
    _START = time.time()
    # fallback seed
    seed_domain = np.array([1,2,3,5,30], dtype=np.int64)
    seed_values = np.array([3.47155351603266,-1.0,-1.0,-1.0,1.0], dtype=np.float64)
    best = (seed_domain, seed_values)
    best_a = _compute_a(seed_domain, seed_values)
    
    for K in [800, 1200, 1600, 2000]:
        if _elapsed() > _DEADLINE - 120:
            break
        try:
            # variables f(2)..f(K)
            ks = np.arange(2, K+1, dtype=float)
            c = np.log(ks) / ks  # minimize c.f
            n = len(ks)
            # bounds
            bounds = [(-10.0, 10.0)] * n
            # constraints A f <= 1.0001 for x in 1..10K
            maxK = 10 * K
            # build sparse A: rows = x, cols = k-2
            # A[x-1, k-2] = floor(x/k) - x/k
            # To save time, build in chunks and store
            rows = []
            cols = []
            data = []
            # vectorize per x chunk
            xchunk = 1000
            for x0 in range(1, maxK+1, xchunk):
                x1 = min(x0+xchunk, maxK+1)
                xs = np.arange(x0, x1, dtype=float)
                # compute floor(xs[:,None]/ks[None,:]) - xs[:,None]/ks[None,:]
                # this is (chunk x n)
                mat = np.floor(xs[:, None] / ks[None, :]) - xs[:, None] / ks[None, :]
                # append
                r = np.repeat(np.arange(x0-1, x1-1), n)
                cc = np.tile(np.arange(n), len(xs))
                rows.append(r); cols.append(cc); data.append(mat.ravel())
            A = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(maxK, n))
            b = np.full(maxK, 1.0001)
            # f(1) bounds: -10 <= -sum f/k <= 10  ->  sum f/k <= 10 and sum f/k >= -10
            # sum f/k = dot(f, 1/k)
            invk = 1.0 / ks
            A1 = csr_matrix((invk, (np.zeros(n), np.arange(n))), shape=(1, n))
            A2 = csr_matrix((-invk, (np.zeros(n), np.arange(n))), shape=(1, n))
            A_all = csr_matrix.vstack([A, A1, A2], format='csr')
            b_all = np.concatenate([b, [10.0], [10.0]])
            res = linprog(c, A_ub=A_all, b_ub=b_all, bounds=bounds, method='highs')
            if res.success:
                f = res.x
                # build domain/values
                dom = np.arange(2, K+1, dtype=np.int64)
                vals = f
                # add 1 with placeholder (will be adjusted)
                domain = np.concatenate([[1], dom])
                values = np.concatenate([[0.0], vals])
                # verify
                if _verify(domain, values, K):
                    a = _compute_a(domain, values)
                    if a > best_a:
                        best_a = a
                        best = (domain, values)
            if _elapsed() > _DEADLINE - 120:
                break
        except Exception:
            pass
    return best
