"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T_target, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return np.allclose(T, T_target, atol=tol)

def _optimize_rank(r, n, m, p, T_target, deadline, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0; lr = 0.1
    
    for it in range(5000):
        if time.time() > deadline - 12:
            break
        t += 1
        
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(r):
            T += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        diff = T - T_target
        if np.max(np.abs(diff)) < 1e-7:
            return U, V, W
        
        dU = np.zeros_like(U); dV = np.zeros_like(V); dW = np.zeros_like(W)
        for c in range(r):
            VW = np.einsum('j,k->jk', V[:,c], W[:,c])
            UW = np.einsum('i,k->ik', U[:,c], W[:,c])
            UV = np.einsum('i,j->ij', U[:,c], V[:,c])
            dU[:,c] = np.einsum('ijk,jk->i', diff, VW)
            dV[:,c] = np.einsum('ijk,ik->j', diff, UW)
            dW[:,c] = np.einsum('ijk,ij->k', diff, UV)
        
        mU = b1*mU+(1-b1)*dU; vU = b2*vU+(1-b2)*dU**2
        U -= lr*(mU/(1-b1**t))/(np.sqrt(vU/(1-b2**t))+eps)
        mV = b1*mV+(1-b1)*dV; vV = b2*vV+(1-b2)*dV**2
        V -= lr*(mV/(1-b1**t))/(np.sqrt(vV/(1-b2**t))+eps)
        mW = b1*mW+(1-b1)*dW; vW = b2*vW+(1-b2)*dW**2
        W -= lr*(mW/(1-b1**t))/(np.sqrt(vW/(1-b2**t))+eps)
        
        if it % 500 == 0:
            lr *= 0.9
    
    return None

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col]=1; V[j*p+k, col]=1; W[k*n+i, col]=1; col+=1
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 138
    T_target = _build_target(n, m, p)
    U_sch, V_sch, W_sch, rank_sch = _schoolbook(n, m, p)
    best = (U_sch, V_sch, W_sch, rank_sch)
    lb = max(n*m, m*p, p*n)
    
    for r in range(lb, rank_sch):
        if time.time() > deadline - 20:
            break
        found = None
        tleft = deadline - 20 - time.time()
        nr = max(1, min(int(tleft/4.0), 25))
        for s in range(nr):
            if time.time() > deadline - 20:
                break
            res = _optimize_rank(r, n, m, p, T_target, deadline, seed=s*104729+7)
            if res is not None:
                U,V,W = res
                if _verify_fast(U.tolist(), V.tolist(), W.tolist(), T_target):
                    found = (U.tolist(), V.tolist(), W.tolist(), r)
                    break
        if found is not None:
            best = found
            break
    
    U, V, W, rank = best
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
