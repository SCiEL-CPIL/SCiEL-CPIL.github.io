import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from 40 down to 20
    best_result = None
    
    for rank in range(40, 19, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, minimize ||T[a,:,:] - U[a,:] * (V*W)^T||
                # V*W is (m*p, p*n) with entry (b,c) = sum_r V[b,r]*W[c,r]
                VW = V @ W.T  # (m*p, p*n)
                # For each a: T[a,:,:] = U[a,:] @ VW  => U[a,:] = T[a,:,:] @ VW^T @ (VW @ VW^T)^{-1}
                # But this is expensive. Let's do it per row.
                for a in range(n*m):
                    # Minimize ||T[a,:,:] - U[a,:] @ VW||_F^2
                    # Gradient: U[a,:] = T[a,:,:] @ VW^T @ (VW @ VW^T)^{-1}
                    # This is a least squares problem
                    try:
                        U[a,:] = np.linalg.lstsq(VW, T[a,:,:].reshape(-1), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                UW = U @ W.T  # (n*m, p*n)
                for b in range(m*p):
                    # T[:,b,:] = VW[b,:]... wait, let me reconsider
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed b: T[:,b,:] = (U * V[b,:]) @ W^T where (U*V[b,:])[a,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # Actually: T[:,b,:] = U @ diag(V[b,:]) @ W^T... no
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed b: let's define M[a,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # This equals (U * V[b,:]) @ W^T where (U*V[b,:])[a,r] = U[a,r]*V[b,r]
                    # So M = (U * V[b,:]) @ W^T
                    # We want M = T[:,b,:]
                    # So: T[:,b,:] = (U * V[b,:]) @ W^T
                    # Let X = U * V[b,:] (element-wise), then T[:,b,:] = X @ W^T
                    # X = T[:,b,:] @ W @ (W^T @ W)^{-1}... but X is constrained: X[a,r] = U[a,r]*V[b,r]
                    # This is not a simple linear problem in V[b,:].
                    # Let me use a different approach.
                    # For fixed b and r: the contribution to T is U[:,r] * V[b,r] * W[:,r]
                    # T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
                    # Let P_r = U[:,r] * W[:,r] (element-wise), shape (n*m, p*n)
                    # Then T[:,b,:] = sum_r V[b,r] * P_r
                    # So: T[:,b,:] (flattened) = [P_0, P_1, ..., P_{rank-1}] @ V[b,:]^T
                    # This is a linear least squares in V[b,:]
                    P_flat = np.array([np.outer(U[:,r], W[:,r]).flatten() for r in range(rank)])  # (rank, n*m*p*n)
                    target = T[:,b,:].flatten()
                    try:
                        V[b,:] = np.linalg.lstsq(P_flat.T, target, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                UV = U @ V.T  # (n*m, m*p)
                for c in range(p*n):
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                    # Let Q_r = U[:,r] * V[:,r], shape (n*m, m*p)
                    # T[:,:,c] (flattened) = [Q_0, Q_1, ..., Q_{rank-1}] @ W[c,:]^T
                    Q_flat = np.array([np.outer(U[:,r], V[:,r]).flatten() for r in range(rank)])  # (rank, n*m*m*p)
                    target = T[:,:,c].flatten()
                    try:
                        W[c,:] = np.linalg.lstsq(Q_flat.T, target, rcond=None)[0]
                    except:
                        pass
            
            # Check if we got an exact decomposition
            # Compute the reconstructed tensor and check max error
            # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is expensive for full tensor, but we can check a subset or use einsum
            # Let's check the Frobenius norm of the error
            # T_recon = np.einsum('ar,br,cr->abc', U, V, W)  # This might be too large
            # Instead, check specific entries
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        max_err = max(max_err, abs(val - 1.0))
            # Also check that non-matching entries are ~0
            # Sample some non-matching entries
            import random
            for _ in range(100):
                a = random.randint(0, n*m-1)
                b = random.randint(0, m*p-1)
                c = random.randint(0, p*n-1)
                # Check if this is a matching entry
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k2 = b % p
                k1 = c // n
                i2 = c % n
                if not (i1 == i2 and j1 == j2 and k1 == k2):
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    max_err = max(max_err, abs(val))
            
            if max_err < 1e-6:
                # Found a low-rank decomposition! Try to round to rationals
                # Convert to fractions with small denominators
                from fractions import Fraction
                U_frac = [[Fraction(U[a,r]).limit_denominator(1000) for r in range(rank)] for a in range(n*m)]
                V_frac = [[Fraction(V[b,r]).limit_denominator(1000) for r in range(rank)] for b in range(m*p)]
                W_frac = [[Fraction(W[c,r]).limit_denominator(1000) for r in range(rank)] for c in range(p*n)]
                
                # Verify the rounded version
                max_err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = sum(float(U_frac[a][r]) * float(V_frac[b][r]) * float(W_frac[c][r]) for r in range(rank))
                            max_err_rounded = max(max_err_rounded, abs(val - 1.0))
                
                if max_err_rounded < 1e-9:
                    # Convert to list of floats (integers or half-integers)
                    U_out = [[float(x) for x in row] for row in U_frac]
                    V_out = [[float(x) for x in row] for row in V_frac]
                    W_out = [[float(x) for x in row] for row in W_frac]
                    return {"U": U_out, "V": V_out, "W": W_out}
            
            # If not exact but close, still keep as candidate
            if max_err < 1e-4 and (best_result is None or max_err < best_result[0]):
                best_result = (max_err, U, V, W, rank)
        
        # If we found a valid decomposition at this rank, return it
        # (the loop above would have returned already if exact)
    
    # If no better found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
