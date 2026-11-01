"""Matrix multiplication 3x3x3 rank-minimal decomposition search."""
import time
import random
import itertools
import math
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to be safe before 150s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = n * m, m * p, p * n  # all 9
    
    # Schoolbook rank 27 decomposition as fallback
    def schoolbook():
        rank = 27
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Check if a decomposition is valid (tensor product equals multiplication tensor)
    def is_valid(U, V, W, tol=1e-6):
        """Check that U * V * W reconstructs the matrix multiplication tensor."""
        rank = len(U[0])
        # Tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Where a = i*m+j (input A index), b = j*p+k (input B index), c = k*n+i (output index)
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # T[a,b,c] should be 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0
        # Actually T[a,b,c] = 1 iff a and b share the same j (the contraction index)
        # and c = (b_j * n + a_i) ... let me re-derive.
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 iff the indices are consistent
        # i.e., the j in U matches the j in V, and the k in V matches the k in W, and the i in W matches the i in U
        # This gives 1 when all three indices are "compatible" for a single term in the product.
        
        max_err = 0.0
        for a in range(nm):
            i, ja = divmod(a, m)
            for b in range(mp):
                jb, k = divmod(b, p)
                if ja != jb:
                    expected = 0.0
                else:
                    expected = 0.0
                    # Actually for the multiplication tensor, T[a,b,c] = delta(ja, jb) * delta(k, kc) * delta(i, ic)
                    # Let me just check against the known structure
                    pass
                for c in range(pn):
                    kc, ic = divmod(c, n)
                    # T[a,b,c] = 1 if ja==jb AND k==kc AND i==ic
                    expected = 1.0 if (ja == jb and k == kc and i == ic) else 0.0
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err
                    if max_err > tol:
                        return False
        return True
    
    # Better validity check using numpy
    import numpy as np
    
    def is_valid_np(U, V, W, tol=1e-4):
        """Efficient validity check."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        
        # Compute tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Using einsum
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Expected tensor
        T_exp = np.zeros((nm, mp, pn), dtype=np.float64)
        for a in range(nm):
            i, ja = divmod(a, m)
            for b in range(mp):
                jb, k = divmod(b, p)
                if ja == jb:
                    for c in range(pn):
                        kc, ic = divmod(c, n)
                        if k == kc and i == ic:
                            T_exp[a, b, c] = 1.0
        
        max_err = np.max(np.abs(T - T_exp))
        return max_err < tol, max_err
    
    # Try rank 23 - this is the known minimal rank for 3x3 matrix multiplication
    # Use a structured approach: start from rank 27 and try to eliminate terms
    
    # The key insight: we need to find U, V, W of shape (9, rank) such that
    # einsum('ar,br,cr->abc', U, V, W) equals the multiplication tensor
    
    # Strategy: Use random initialization + gradient-like refinement
    # Since we can't do true gradient descent easily, use a coordinate descent approach
    
    # Build the target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for a in range(nm):
        i, ja = divmod(a, m)
        for b in range(mp):
            jb, k = divmod(b, p)
            if ja == jb:
                for c in range(pn):
                    kc, ic = divmod(c, n)
                    if k == kc and i == ic:
                        T_target[a, b, c] = 1.0
    
    # Count non-zero entries
    nz = np.count_nonzero(T_target)
    # There should be 27 non-zero entries (one for each (i,j,k) triple)
    
    def time_remaining():
        return deadline - time.time()
    
    # Try to find rank 23 decomposition using iterative refinement
    # Start with random matrices and try to refine
    
    best_rank = 27
    best_result = schoolbook()
    
    # For ranks from 23 up to 26, try to find a valid decomposition
    for target_rank in [23, 24, 25, 26]:
        if time_remaining() < 30:
            break
        
        found = False
        attempts = 0
        max_attempts = 5000
        
        while not found and attempts < max_attempts and time_remaining() > 5:
            attempts += 1
            
            # Initialize randomly
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Coordinate descent: iteratively update one row at a time
            # For a given row U[a,:], the optimal U[a,:] given V and W is:
            # We want sum_r U[a,r] * V[b,r] * W[c,r] = T_target[a,b,c] for all b,c
            # This is a linear system in U[a,:]
            # Let M[a,b,c] = V[b,:] ⊗ W[c,:] (outer product), then U[a,:] · M[a,b,c] = T[a,b,c]
            # Stack all (b,c) pairs: it's a linear system
            
            for iteration in range(200):
                if time_remaining() < 5:
                    break
                
                # Update each row
                for a in range(nm):
                    # Build linear system for U[a,:]
                    # For each (b,c): sum_r U[a,r] * (V[b,r] * W[c,r]) = T_target[a,b,c]
                    # This is: U[a,:] · G = t where G is (mp*pn, rank) matrix
                    G = np.zeros((mp * pn, target_rank))
                    t_vec = np.zeros(mp * pn)
                    idx = 0
                    for b in range(mp):
                        for c in range(pn):
                            G[idx, :] = V[b, :] * W[c, :]
                            t_vec[idx] = T_target[a, b, c]
                            idx += 1
                    
                    # Solve least squares: U[a,:] = (G^T G)^{-1} G^T t
                    try:
                        U[a, :] = np.linalg.lstsq(G, t_vec, rcond=None)[0]
                    except:
                        pass
                
                for b in range(mp):
                    G = np.zeros((nm * pn, target_rank))
                    t_vec = np.zeros(nm * pn)
                    idx = 0
                    for a in range(nm):
                        for c in range(pn):
                            G[idx, :] = U[a, :] * W[c, :]
                            t_vec[idx] = T_target[a, b, c]
                            idx += 1
                    try:
                        V[b, :] = np.linalg.lstsq(G, t_vec, rcond=None)[0]
                    except:
                        pass
                
                for c in range(pn):
                    G = np.zeros((nm * mp, target_rank))
                    t_vec = np.zeros(nm * mp)
                    idx = 0
                    for a in range(nm):
                        for b in range(mp):
                            G[idx, :] = U[a, :] * V[b, :]
                            t_vec[idx] = T_target[a, b, c]
                            idx += 1
                    try:
                        W[c, :] = np.linalg.lstsq(G, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                if iteration % 20 == 0:
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T - T_target))
                    if err < 1e-4:
                        # Found a valid decomposition!
                        found = True
                        break
            
            if found:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found:
            break
    
    # Final validity check
    if best_rank < 27:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(T - T_target))
        if err > 1e-3:
            # Invalid, fall back to schoolbook
            best_result = schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
