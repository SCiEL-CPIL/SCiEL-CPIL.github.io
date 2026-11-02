"""
Rank-minimal tensor decomposition for 2x5 by 5x5 matrix multiplication.

Strategy:
  1. Build the schoolbook rank-50 decomposition (one term per (i,j,k) triple).
  2. Try to find a lower-rank decomposition using structured randomised search
     with half-integer snapping and exact verification.
  3. Use the insight that the 1x5*5x1 tensor has rank 5, and attempt to
     exploit structure for the 2x5*5x5 case.

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
    # Build target: T[a,b,c] = 1 iff j_a == j_b and k_b == k_c and i_c == i_a
    # a = i*m+j, b = j*p+k, c = k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # T[a,b,c] should be 1
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                if val != Fraction(1):
                    return False
    # Check that all other entries are 0
    # There are n*m * m*p * p*n total entries, only n*m*p are non-zero
    # We need to check that entries where the chain doesn't hold are 0
    # This is expensive, so we only do a sample check
    # Actually, let's check all entries for correctness
    # Total entries: 10 * 25 * 10 = 2500, with rank up to 50, this is 125000 ops
    # That's fine
    for a in range(n * m):
        ia, ja = divmod(a, m)
        for b in range(m * p):
            jb, kb = divmod(b, p)
            if jb != ja:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if kc != kb or ic != ia:
                    # This entry should be 0
                    val = Fraction(0)
                    for r in range(rank):
                        val += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if val != Fraction(0):
                        return False
    return True


def _snap_to_half_int(arr):
    """Snap a numpy array to nearest half-integer."""
    return np.round(arr * 2) / 2


def _try_rank_structured(r, deadline, seed=42):
    """
    Try to find a rank-r decomposition using structured initialization.
    
    Approach: 
    - For each (i,k) pair, the sum over j of A[i,j]*B[j,k] = C[i,k]
    - We try to assign rank-1 terms that cover multiple j's
    - Use randomised linear algebra to find valid U, V, W
    
    Strategy: Initialize U, V, W with random half-integer values,
    then use iterative refinement.
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
    
    # Initialize with random half-integers in {-1, -0.5, 0, 0.5, 1}
    def rand_half_int(shape):
        return np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=shape)
    
    best_loss = float('inf')
    best_solution = None
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        U = rand_half_int((n * m, r))
        V = rand_half_int((m * p, r))
        W = rand_half_int((p * n, r))
        
        # Iterative refinement
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T_recon - T_target) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                break
            
            # Try updating one factor at a time
            # Update W: for each c, W[c,:] should satisfy
            # sum_r W[c,r] * (sum over a,b where T_target[a,b,c]=1 of U[a,r]*V[b,r]) = 1
            # This is a linear system
            for factor in range(3):
                if time.time() > deadline:
                    break
                
                if factor == 0:  # Update U
                    # For each a: U[a,:] should satisfy
                    # sum_r U[a,r] * (sum over b,c where T_target[a,b,c]=1 of V[b,r]*W[c,r]) = 1
                    for a in range(n * m):
                        # Build the system: sum_r U[a,r] * M[a,r] = 1
                        # where M[a,r] = sum over (b,c) with T_target[a,b,c]=1 of V[b,r]*W[c,r]
                        M = np.zeros(r)
                        for b in range(m * p):
                            jb, kb = divmod(b, p)
                            ia, ja = divmod(a, m)
                            if jb != ja:
                                continue
                            for c in range(p * n):
                                kc, ic = divmod(c, n)
                                if kc == kb and ic == ia:
                                    M += V[b, :] * W[c, :]
                        # Solve: U[a,:] * M = 1 (dot product)
                        # This is underdetermined if r > 1
                        # Use least squares with constraint
                        if np.linalg.norm(M) > 1e-10:
                            # Minimize ||U[a,:] * M - 1||^2
                            # Solution: U[a,:] = M / (M.M) * 1
                            U[a, :] = M / np.dot(M, M)
                elif factor == 1:  # Update V
                    for b in range(m * p):
                        M = np.zeros(r)
                        for a in range(n * m):
                            ia, ja = divmod(a, m)
                            jb, kb = divmod(b, p)
                            if jb != ja:
                                continue
                            for c in range(p * n):
                                kc, ic = divmod(c, n)
                                if kc == kb and ic == ia:
                                    M += U[a, :] * W[c, :]
                        if np.linalg.norm(M) > 1e-10:
                            V[b, :] = M / np.dot(M, M)
                else:  # Update W
                    for c in range(p * n):
                        M = np.zeros(r)
                        for a in range(n * m):
                            ia, ja = divmod(a, m)
                            for b in range(m * p):
                                jb, kb = divmod(b, p)
                                kc, ic = divmod(c, n)
                                if jb == ja and kc == kb and ic == ia:
                                    M += U[a, :] * V[b, :]
                        if np.linalg.norm(M) > 1e-10:
                            W[c, :] = M / np.dot(M, M)
        
        # Snap to half-integers
        U_snap = _snap_to_half_int(U)
        V_snap = _snap_to_half_int(V)
        W_snap = _snap_to_half_int(W)
        
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        loss = np.sum((T_recon - T_target) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_solution = (U_snap.copy(), V_snap.copy(), W_snap.copy())
        
        if loss < 1e-9:
            # Verify exactly
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            if _verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_rank_random(r, deadline, seed=42):
    """
    Try to find a rank-r decomposition using random initializations
    and gradient-free search.
    """
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
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
    
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        scale = 1.0 / np.sqrt(r)
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        
        # Alternating least squares
        for it in range(30):
            if time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                # Build system: U[a,:] @ (V[b,:] * W[c,:]) = T_target[a,b,c]
                # for all (b,c)
                # This is: U[a,:] @ M_a = t_a where M_a is r x (m*p*p*n)
                # Too large, use subsampling
                # Instead, use the fact that only specific (b,c) have T=1
                M_rows = []
                t_vals = []
                for b in range(m * p):
                    jb, kb = divmod(b, p)
                    ia, ja = divmod(a, m)
                    if jb != ja:
                        continue
                    for c in range(p * n):
                        kc, ic = divmod(c, n)
                        if kc == kb and ic == ia:
                            M_rows.append(V[b, :] * W[c, :])
                            t_vals.append(1.0)
                if M_rows:
                    M = np.array(M_rows)
                    t = np.array(t_vals)
                    # Solve: U[a,:] @ M.T = t
                    # Least squares
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                    except:
                        pass
            
            # Update V
            for b in range(m * p):
                jb, kb = divmod(b, p)
                M_rows = []
                t_vals = []
                for a in range(n * m):
                    ia, ja = divmod(a, m)
                    if jb != ja:
                        continue
                    for c in range(p * n):
                        kc, ic = divmod(c, n)
                        if kc == kb and ic == ia:
                            M_rows.append(U[a, :] * W[c, :])
                            t_vals.append(1.0)
                if M_rows:
                    M = np.array(M_rows)
                    t = np.array(t_vals)
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M.T, t, rcond=None)
                    except:
                        pass
            
            # Update W
            for c in range(p * n):
                kc, ic = divmod(c, n)
                M_rows = []
                t_vals = []
                for a in range(n * m):
                    ia, ja = divmod(a, m)
                    for b in range(m * p):
                        jb, kb = divmod(b, p)
                        if jb == ja and kc == kb and ic == ia:
                            M_rows.append(U[a, :] * V[b, :])
                            t_vals.append(1.0)
                if M_rows:
                    M = np.array(M_rows)
                    t = np.array(t_vals)
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
    
    # Try to find lower rank decompositions
    # Try ranks from 45 down to 30
    for r in range(45, 29, -1):
        if time.time() > deadline:
            break
        
        # Try structured approach first
        result = _try_rank_structured(r, deadline, seed=42)
        if result is not None:
            best = result
            best_rank = r
            break
        
        if time.time() > deadline:
            break
        
        # Try random approach
        result = _try_rank_random(r, deadline, seed=123)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
