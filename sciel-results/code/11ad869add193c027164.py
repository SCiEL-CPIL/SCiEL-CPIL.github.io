"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.
Uses Strassen (rank 7) for cols 0,1 and optimized search for cols 2,3,4.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 145.0

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify decomposition correctness."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """Rank-18: Strassen(7) + Strassen(7) + Schoolbook(4)."""
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen(col0, col1, base):
        r = base
        U[0][r]=1; U[3][r]=1; V[col0][r]=1; V[5+col1-2][r]=1; W[col0][r]=1; W[col1+3][r]=1
        # Fix indices: V[j*p+k], W[k*n+i]
        V[0*p+col0][r]=1; V[1*p+col1][r]=1; W[col0*n+0][r]=1; W[col1*n+1][r]=1
        
        r = base+1
        U[2][r]=1; U[3][r]=1; V[0*p+col0][r]=1; W[col0*n+1][r]=1; W[col1*n+1][r]=-1
        
        r = base+2
        U[0][r]=1; V[0*p+col1][r]=1; V[1*p+col1][r]=-1; W[col1*n+0][r]=1; W[col1*n+1][r]=1
        
        r = base+3
        U[3][r]=1; V[1*p+col0][r]=1; V[0*p+col0][r]=-1; W[col0*n+0][r]=1; W[col0*n+1][r]=1
        
        r = base+4
        U[0][r]=1; U[1][r]=1; V[1*p+col1][r]=1; W[col0*n+0][r]=-1; W[col1*n+0][r]=1
        
        r = base+5
        U[2][r]=1; U[0][r]=-1; V[0*p+col0][r]=1; V[0*p+col1][r]=1; W[col1*n+1][r]=1
        
        r = base+6
        U[1][r]=1; U[3][r]=-1; V[1*p+col0][r]=1; V[1*p+col1][r]=1; W[col0*n+0][r]=1
    
    add_strassen(0, 1, 0)
    add_strassen(2, 3, 7)
    
    r = 14
    U[0][r]=1; V[0*p+4][r]=1; W[4*n+0][r]=1
    r = 15
    U[1][r]=1; V[1*p+4][r]=1; W[4*n+0][r]=1
    r = 16
    U[2][r]=1; V[0*p+4][r]=1; W[4*n+1][r]=1
    r = 17
    U[3][r]=1; V[1*p+4][r]=1; W[4*n+1][r]=1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _build_target(n=2, m=2, p=5):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-13):
    """ALS refinement."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve U
        VW = np.einsum('vr,wr->vwr', V, W)
        VWf = VW.transpose(0,2,1).reshape(-1, R)
        for u in range(n*m):
            sol, _, _, _ = np.linalg.lstsq(VWf, T[u].reshape(-1), rcond=None)
            U[u] = sol
        # Solve V
        UW = np.einsum('ur,wr->uwr', U, W)
        UWf = UW.transpose(0,2,1).reshape(-1, R)
        for v in range(m*p):
            sol, _, _, _ = np.linalg.lstsq(UWf, T[:,v,:].reshape(-1), rcond=None)
            V[v] = sol
        # Solve W
        UV = np.einsum('ur,vr->uvr', U, V)
        UVf = UV.reshape(-1, R)
        for w in range(p*n):
            sol, _, _, _ = np.linalg.lstsq(UVf, T[:,:,w].reshape(-1), rcond=None)
            W[w] = sol
        
        recon = np.zeros_like(T)
        for r in range(R):
            recon += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.sqrt(np.sum((recon-T)**2) / T.size)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _rationalize(U, V, W, T, R, n, m, p):
    """Try to find half-integer/rational exact solution."""
    recon = np.zeros_like(T)
    for r in range(R):
        recon += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    if not np.allclose(recon, T, atol=1e-9):
        return None
    
    for denom in [1, 2, 4, 8]:
        Us = np.round(U * denom) / denom
        Vs = np.round(V * denom) / denom
        Ws = np.round(W * denom) / denom
        recon_s = np.zeros_like(T)
        for r in range(R):
            recon_s += np.einsum('u,v,w->uvw', Us[:,r], Vs[:,r], Ws[:,r])
        if np.allclose(recon_s, T, atol=1e-10):
            Ul = [[float(x) for x in row] for row in Us]
            Vl = [[float(x) for x in row] for row in Vs]
            Wl = [[float(x) for x in row] for row in Ws]
            for mat in [Ul, Vl, Wl]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                            mat[i][j] = int(round(mat[i][j]))
            if _verify(Ul, Vl, Wl, n, m, p):
                return {"U": Ul, "V": Vl, "W": Wl}
    return None

def _search_rank(R, n, m, p, T, timeout=30, n_trials=40):
    """Search for rank-R decomposition."""
    start = time.time()
    for trial in range(n_trials):
        if time.time() - start > timeout - 3:
            break
        
        if trial % 3 == 0:
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3
        elif trial % 3 == 1:
            # Structured: Strassen-like for first part
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            # Initialize first 7 as Strassen for cols 0,1
            if R >= 7:
                U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[6,0]=1
                U[2,1]=1; U[3,1]=1; V[0,1]=1; W[5,1]=1; W[6,1]=-1
                U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[6,2]=1
                U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[5,3]=1
                U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
                U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[6,5]=1
                U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
            # Add noise to remaining
            for r in range(min(7,R), R):
                U[:,r] = np.random.randn(n*m) * 0.3
                V[:,r] = np.random.randn(m*p) * 0.3
                W[:,r] = np.random.randn(p*n) * 0.3
        else:
            scale = 0.1 + 0.4 * (trial % 10) / 10
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
        if err < 1e-8:
            result = _rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    return None

def _construct_rank16_attempt(n=2, m=2, p=5):
    """
    Attempt rank-16: Strassen(7) for cols 0,1 + rank-9 for cols 2,3,4.
    The 2x2x3 part needs rank 9 if we share A-factors with Strassen.
    """
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen for cols 0,1 (indices 0-6)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[6][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[5][1]=1; W[6][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[6][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[5][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[6][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    
    # For cols 2,3,4: B indices 4,5,6,7,8,9; C indices 2,3,4,7,8,9
    # Try a rank-9 decomposition
    # N1: A00*(B02+B13) -> C02 += N1, C13 += N1
    U[0][7]=1; V[4][7]=1; V[8][7]=1; W[2][7]=1; W[8][7]=1
    # N2: (A10+A11)*B02 -> C12 += N2, C13 -= N2  
    U[2][8]=1; U[3][8]=1; V[4][8]=1; W[7][8]=1; W[8][8]=-1
    # N3: (A00+A01)*B13 -> C03 += N3, C13 += N3
    U[0][9]=1; U[1][9]=1; V[8][9]=1; W[3][9]=1; W[8][9]=1
    # N4: A11*(B03-B02) -> C03 += N4
    U[3][10]=1; V[5][10]=1; V[4][10]=-1; W[3][10]=1
    # N5: A00*(B12-B13) -> C02 -= N5, C12 += N5
    U[0][11]=1; V[7][11]=1; V[8][11]=-1; W[2][11]=-1; W[7][11]=1
    # N6: (A10-A00)*B02 -> C12 += N6
    U[2][12]=1; U[0][12]=-1; V[4][12]=1; W[7][12]=1
    # N7: (A01-A11)*B13 -> C03 += N7, C13 -= N7
    U[1][13]=1; U[3][13]=-1; V[8][13]=1; W[3][13]=1; W[8][13]=-1
    # N8: A00*B04 -> C04 += N8
    U[0][14]=1; V[6][14]=1; W[4][14]=1
    # N9: A10*B04 -> C14 += N9
    U[2][15]=1; V[6][15]=1; W[9][15]=1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _schoolbook(n=2, m=2, p=5):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Guaranteed rank-18
    fallback = _construct_rank18(n, m, p)
    
    # Try rank-16 constructive
    r16 = _construct_rank16_attempt(n, m, p)
    if r16 is not None:
        return r16
    
    T = _build_target(n, m, p)
    
    # Search for rank 15
    remaining = deadline - time.time() - 5
    if remaining > 30:
        result = _search_rank(15, n, m, p, T, timeout=min(35, remaining), n_trials=30)
        if result is not None:
            return result
    
    # Search for rank 14
    remaining = deadline - time.time() - 5
    if remaining > 25:
        result = _search_rank(14, n, m, p, T, timeout=min(28, remaining), n_trials=25)
        if result is not None:
            return result
    
    # Search for rank 16 (in case constructive failed but search succeeds)
    remaining = deadline - time.time() - 5
    if remaining > 20 and r16 is None:
        result = _search_rank(16, n, m, p, T, timeout=min(22, remaining), n_trials=20)
        if result is not None:
            return result
    
    if fallback is not None:
        return fallback
    return _schoolbook(n, m, p)
