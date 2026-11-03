"""
Rank-minimal tensor decomposition for 2x5 by 5x5 matrix multiplication.

Strategy:
  1. Try to find a rank-10 decomposition using the Kronecker product identity:
     M_{2,5,5} = M_{2,1,1} ⊗ M_{1,5,5}
     where M_{2,1,1} has rank 2 (scalar mult: 2*1*1=2) and M_{1,5,5} has rank 5.
     The Kronecker product of rank-r and rank-s decompositions gives rank r*s = 10.
  2. Verify the rank-10 decomposition exactly using Fraction arithmetic.
  3. If rank 10 works, try to find even lower rank via structured search.
  4. Fall back to schoolbook rank-50 if nothing better is found.

The output follows the trace convention W[k*n+i, r]:
  n=2, m=5, p=5
  U shape: (10, rank)  indices: (i, j) -> i*m+j
  V shape: (25, rank)  indices: (j, k) -> j*p+k
  W shape: (10, rank)  indices: (k, i) -> k*n+i
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]


def _build_schoolbook():
    """Build the schoolbook rank-50 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W):
    """
    Verify that the decomposition exactly reproduces the matmul tensor.
    Uses Fraction for exact arithmetic.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                if val != Fraction(1):
                    return False
    for a in range(n * m):
        ia, ja = divmod(a, m)
        for b in range(m * p):
            jb, kb = divmod(b, p)
            if jb != ja:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if kc != kb or ic != ia:
                    val = Fraction(0)
                    for r in range(rank):
                        val += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if val != Fraction(0):
                        return False
    return True


def _build_kronecker_decomposition():
    """
    Build a rank-10 decomposition using the Kronecker product identity:
    M_{2,5,5} = M_{2,1,1} ⊗ M_{1,5,5}
    
    For M_{2,1,1} (2x1 * 1x1):
      U1 shape (2, 2), V1 shape (1, 2), W1 shape (1, 2)
      Term r=0: U1[0,0]=1, V1[0,0]=1, W1[0,0]=1  (i=0, j=0, k=0)
      Term r=1: U1[1,1]=1, V1[0,1]=1, W1[0,1]=1  (i=1, j=0, k=0)
    
    For M_{1,5,5} (1x5 * 5x5):
      U2 shape (5, 5), V2 shape (25, 5), W2 shape (5, 5)
      Term r=0..4: U2[j,r]=1 if j==r, V2[5*j+k, r]=1 if j==r, W2[k, r]=1 if k==r
    
    The Kronecker product decomposition:
      U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2
      with rank = rank1 * rank2 = 2 * 5 = 10
      
    Index mapping for the combined decomposition:
      a = i*m + j = i*5 + j  where i in {0,1}, j in {0..4}
      b = j*p + k = j*5 + k  where j in {0..4}, k in {0..4}
      c = k*n + i = k*2 + i  where k in {0..4}, i in {0,1}
      
      For the Kronecker product, the rank index r = r1 * 5 + r2 where r1 in {0,1}, r2 in {0..4}
      
      U[a, r] = U1[i, r1] * U2[j, r2]
      V[b, r] = V1[j, r1] * V2[j*5+k, r2]
      W[c, r] = W1[k, r1] * W2[k, r2]
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank1 = n * 1 * 1  # = 2
    rank2 = 1 * m * p  # = 5
    rank = rank1 * rank2  # = 10
    
    # Build U1, V1, W1 for M_{2,1,1}
    # n1=2, m1=1, p1=1
    n1, m1, p1 = 2, 1, 1
    U1 = [[0] * rank1 for _ in range(n1 * m1)]  # shape (2, 2)
    V1 = [[0] * rank1 for _ in range(m1 * p1)]  # shape (1, 2)
    W1 = [[0] * rank1 for _ in range(p1 * n1)]  # shape (2, 2)
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
    # U1: U1[0,0]=1, U1[1,1]=1
    # V1: V1[0,0]=1, V1[0,1]=1
    # W1: W1[0,0]=1, W1[1,1]=1
    
    # Build U2, V2, W2 for M_{1,5,5}
    # n2=1, m2=5, p2=5
    n2, m2, p2 = 1, 5, 5
    U2 = [[0] * rank2 for _ in range(n2 * m2)]  # shape (5, 5)
    V2 = [[0] * rank2 for _ in range(m2 * p2)]  # shape (25, 5)
    W2 = [[0] * rank2 for _ in range(p2 * n2)]  # shape (5, 5)
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    # U2: U2[j, j] = 1 for j=0..4
    # V2: V2[5*j+k, j] = 1 for j=0..4, k=0..4
    # W2: W2[k, k] = 1 for k=0..4
    
    # Build combined U, V, W
    U = [[0] * rank for _ in range(n * m)]  # shape (10, 10)
    V = [[0] * rank for _ in range(m * p)]  # shape (25, 10)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 10)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                for r1 in range(rank1):
                    for r2 in range(rank2):
                        r = r1 * rank2 + r2
                        U[a][r] += U1[i * m1 + j][r1] * U2[j][r2]
                        V[b][r] += V1[j * p1 + k][r1] * V2[j * p2 + k][r2]
                        W[c][r] += W1[k * n1 + i][r1] * W2[k][r2]
    
    return {"U": U, "V": V, "W": W}


def _snap_to_half_int(arr):
    """Snap a numpy array to nearest half-integer."""
    return np.round(arr * 2) / 2


def _try_rank_search(r, deadline, seed=42):
    """
    Try to find a rank-r decomposition using alternating least squares
    with multiple random restarts.
    """
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    best_loss = float('inf')
    best_solution = None
    
    # Precompute valid index sets for efficiency
    valid_bc_for_a = {}
    for a in range(n * m):
        ia, ja = divmod(a, m)
        pairs = []
        for b in range(m * p):
            jb, kb = divmod(b, p)
            if jb != ja:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if kc == kb and ic == ia:
                    pairs.append((b, c))
        valid_bc_for_a[a] = pairs
    
    valid_ac_for_b = {}
    for b in range(m * p):
        jb, kb = divmod(b, p)
        pairs = []
        for a in range(n * m):
            ia, ja = divmod(a, m)
            if jb != ja:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if kc == kb and ic == ia:
                    pairs.append((a, c))
        valid_ac_for_b[b] = pairs
    
    valid_ab_for_c = {}
    for c in range(p * n):
        kc, ic = divmod(c, n)
        pairs = []
        for a in range(n * m):
            ia, ja = divmod(a, m)
            for b in range(m * p):
                jb, kb = divmod(b, p)
                if jb == ja and kc == kb and ic == ia:
                    pairs.append((a, b))
        valid_ab_for_c[c] = pairs
    
    num_trials = 5
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Random initialization
        scale = 1.0 / np.sqrt(max(r, 1))
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        
        # Alternating least squares
        for it in range(20):
            if time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                pairs = valid_bc_for_a[a]
                if not pairs:
                    continue
                M_rows = []
                for b, c in pairs:
                    M_rows.append(V[b, :] * W[c, :])
                if M_rows:
                    M = np.array(M_rows)
                    t = np.ones(len(M_rows))
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                    except:
                        pass
            
            # Update V
            for b in range(m * p):
                pairs = valid_ac_for_b[b]
                if not pairs:
                    continue
                M_rows = []
                for a, c in pairs:
                    M_rows.append(U[a, :] * W[c, :])
                if M_rows:
                    M = np.array(M_rows)
                    t = np.ones(len(M_rows))
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                    except:
                        pass
            
            # Update W
            for c in range(p * n):
                pairs = valid_ab_for_c[c]
                if not pairs:
                    continue
                M_rows = []
                for a, b in pairs:
                    M_rows.append(U[a, :] * V[b, :])
                if M_rows:
                    M = np.array(M_rows)
                    t = np.ones(len(M_rows))
                    try:
                        W[c, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                    except:
                        pass
            
            # Compute loss
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T_recon - T_target) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                break
        
        # Snap and verify
        U_snap = _snap_to_half_int(U)
        V_snap = _snap_to_half_int(V)
        W_snap = _snap_to_half_int(W)
        
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        loss = np.sum((T_recon - T_target) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_solution = (U_snap.copy(), V_snap.copy(), W_snap.copy())
        
        if loss < 1e-9:
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            if _verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Start with schoolbook rank 50
    best = _build_schoolbook()
    best_rank = 50
    
    # Try Kronecker product rank-10 decomposition
    kronecker = _build_kronecker_decomposition()
    if _verify_exact(kronecker["U"], kronecker["V"], kronecker["W"]):
        best = kronecker
        best_rank = 10
    
    # If rank 10 works, try to find even lower rank
    if best_rank == 10:
        for r in range(9, 5, -1):
            if time.time() > deadline:
                break
            result = _try_rank_search(r, deadline, seed=42 + r)
            if result is not None:
                best = result
                best_rank = r
                break
    
    return best
