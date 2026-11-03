import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

def construct():
    t0 = time.time()
    deadline = t0 + 140
    n, m, p = 3, 3, 4
    nm, mp, pn = n*m, m*p, p*n

    def verify(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    def opt_solve(R, U0=None, V0=None, W0=None):
        if time.time() > deadline - 5:
            return None
        if U0 is None:
            np.random.seed(int(time.time()*1000) % 999983)
            U0 = np.random.randn(nm, R) * 0.3
            V0 = np.random.randn(mp, R) * 0.3
            W0 = np.random.randn(pn, R) * 0.3
        
        def obj(x):
            if time.time() > deadline - 2:
                return 1e12
            Uf = x[:nm*R].reshape(nm, R)
            Vf = x[nm*R:nm*R+mp*R].reshape(mp, R)
            Wf = x[nm*R+mp*R:].reshape(pn, R)
            e = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(Uf[i*m+j], Vf[j*p+k] * Wf[k*n+i])
                        e += (s - 1.0)**2
            return e
        
        x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            res = minimize(obj, x0, method='L-BFGS-B', maxiter=400,
                          options={'ftol':1e-15, 'gtol':1e-12})
        except:
            return None
        
        if res.fun < 1e-9:
            Uf = res.x[:nm*R].reshape(nm, R)
            Vf = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
            Wf = res.x[nm*R+mp*R:].reshape(pn, R)
            Ur = np.round(Uf*2)/2
            Vr = np.round(Vf*2)/2
            Wr = np.round(Wf*2)/2
            if verify(Ur, Vr, Wr, R):
                return (Ur, Vr, Wr)
        return None

    # Try ranks from 24 upward
    for R in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 12:
            break
        for att in range(6):
            if time.time() > deadline - 5:
                break
            sol = opt_solve(R)
            if sol is not None:
                Ur, Vr, Wr = sol
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    # Fallback: schoolbook rank 36
    R = n*m*p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
