import numpy as np
from itertools import product
import time

def construct():
    t0 = time.time()
    deadline = t0 + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 4  # (n x m) * (m x p)
    # a = i*m + l, b = l*p + j, c = j*n + i
    # T[a,b,c] = 1 if c = (b % p)*n + (a // m) and (a % m) == (b // p)
    
    # Build sparse list of non-zero tensor entries
    tensor_entries = []
    for i in range(n):
        for l in range(m):
            for j in range(p):
                a = i * m + l
                b = l * p + j
                c = j * n + i
                tensor_entries.append((a, b, c))
    
    tensor_set = set(tensor_entries)
    
    # Fast vectorized verification
    def verify_fast(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # For each rank component k, compute the outer product U[:,k] ⊗ V[:,k] ⊗ W[:,k]
        # and sum them to get the reconstructed tensor
        # T_recon[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
        
        # Efficient: for each k, add outer product
        T_recon = np.zeros((n*m, m*p, p*n))
        for k in range(rank):
            T_recon += np.outer(np.outer(U[:,k], V[:,k]), W[:,k])
        
        # Check non-zero entries
        for (a, b, c) in tensor_entries:
            if abs(T_recon[a,b,c] - 1.0) > 1e-10:
                return False
        
        # Check zero entries
        for i in range(n*m):
            for l in range(m):
                for j in range(p):
                    a = i * m + l
                    b = l * p + j
                    c = j * n + i
                    # This is already covered above; check a few random non-entries
        # Spot check some non-entries
        import random
        random.seed(42)
        checked = 0
        for _ in range(200):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            if (a,b,c) not in tensor_set:
                if abs(T_recon[a,b,c]) > 1e-10:
                    return False
                checked += 1
        return True
    
    # Try rank 3 first (would give best score)
    # For 3x4x4 matrix multiplication, rank 3 is unlikely but let's verify
    # Actually, the lower bound for matmul rank is max(n*m, m*p, p*n) = max(12,16,12) = 16
    # Wait no, that's not right. The rank lower bound is at least max(n, m, p) for some cases.
    # For n x m * m x p, the rank is at least max(n, m, p) in general? No.
    # Actually for 2x2x2, rank is 7 (Strassen). For 3x3x3, it's at most 23, at least 19.
    # For 3x4x4, the schoolbook rank is 4 (since we can do it with m=4 multiplications).
    # Can we do better than 4? The lower bound is at least max(n,m,p) = 4? 
    # Actually the trivial lower bound is max(n, m, p) only for square matrices in some cases.
    # For rectangular, the rank is at least max(n, m, p)? No, that's not a valid lower bound.
    # The valid lower bound is: rank >= max(n*m/p, m*p/n, p*n/m) ... no that's not right either.
    # 
    # Actually, the rank of the matrix multiplication tensor for n x m * m x p is at least
    # max(n, m, p) is NOT a valid lower bound in general.
    # 
    # For 3x4x4: schoolbook gives rank 4. Can we do rank 3?
    # The tensor has 3*4*4 = 48 non-zero entries. With rank 3, we have 3*(12+16+12) = 120 parameters.
    # This is a system of 9216 equations (48 = 1, 9168 = 0).
    # 
    # Let's try a random search for rank 3 within time budget.
    
    best_result = None
    
    # First, establish the rank-4 solution as baseline
    rank = m  # = 4
    U4 = np.zeros((n * m, rank))
    V4 = np.zeros((m * p, rank))
    W4 = np.zeros((p * n, rank))
    
    for i in range(n):
        for l in range(m):
            for j in range(p):
                U4[i * m + l, l] = 1
                V4[l * p + j, l] = 1
                W4[j * n + i, l] = 1
    
    if verify_fast(U4, V4, W4):
        best_result = (4, U4, V4, W4)
    
    # Now try to find rank 3
    if time.time() < deadline - 30:
        # Try rank 3 with random initialization and local optimization
        # The constraints are: sum_k U[a,k]V[b,k]W[c,k] = T[a,b,c]
        # This is a 3-tensor completion problem.
        
        # Strategy: parameterize and use alternating minimization
        # Fix U, solve for V (linear), fix V, solve for W (linear), etc.
        
        rng = np.random.RandomState(42)
        rank3 = 3
        
        for attempt in range(50):
            if time.time() > deadline - 25:
                break
            
            # Random initialization
            U3 = rng.randn(n*m, rank3) * 0.5
            V3 = rng.randn(m*p, rank3) * 0.5
            W3 = rng.randn(p*n, rank3) * 0.5
            
            # Alternating minimization
            converged = False
            for it in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Fix U and V, solve for W
                # For each c: sum_k W[c,k] * (sum over a,b where T[a,b,c]=1: U[a,k]*V[b,k]) = 1
                # Actually: for each (a,b,c) with T=1: sum_k U[a,k]*V[b,k]*W[c,k] = 1
                # For each (a,b,c) with T=0: sum_k U[a,k]*V[b,k]*W[c,k] = 0
                
                # This is a linear system in W if U,V are fixed.
                # Variables: W[c,k] for c in [0, p*n), k in [0, rank3)
                # Number of variables: p*n*rank3 = 12*3 = 36
                # Number of equations: 48 (from T=1 entries) + some from T=0
                
                # Let's use the 48 positive equations
                # A * W_flat = b, where A is 48 x 36
                A = np.zeros((len(tensor_entries), p*n*rank3))
                b = np.ones(len(tensor_entries))
                
                for idx, (a, bb, c) in enumerate(tensor_entries):
                    for k in range(rank3):
                        A[idx, c*rank3 + k] = U3[a, k] * V3[bb, k]
                
                # Solve least squares
                W3_flat, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                W3 = W3_flat.reshape(p*n, rank3)
                
                # Fix V and W, solve for U
                A = np.zeros((len(tensor_entries), n*m*rank3))
                b = np.ones(len(tensor_entries))
                for idx, (a, bb, c) in enumerate(tensor_entries):
                    for k in range(rank3):
                        A[idx, a*rank3 + k] = V3[bb, k] * W3[c, k]
                
                U3_flat, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                U3 = U3_flat.reshape(n*m, rank3)
                
                # Fix U and W, solve for V
                A = np.zeros((len(tensor_entries), m*p*rank3))
                b = np.ones(len(tensor_entries))
                for idx, (a, bb, c) in enumerate(tensor_entries):
                    for k in range(rank3):
                        A[idx, bb*rank3 + k] = U3[a, k] * W3[c, k]
                
                V3_flat, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                V3 = V3_flat.reshape(m*p, rank3)
                
                # Check if positive constraints are satisfied
                max_err = 0
                for (a, bb, c) in tensor_entries:
                    val = np.sum(U3[a, :] * V3[bb, :] * W3[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    # Check negative constraints (sample)
                    neg_ok = True
                    for _ in range(50):
                        a = rng.randint(0, n*m-1)
                        bb = rng.randint(0, m*p-1)
                        c = rng.randint(0, p*n-1)
                        if (a,bb,c) not in tensor_set:
                            val = np.sum(U3[a, :] * V3[bb, :] * W3[c, :])
                            if abs(val) > 1e-4:
                                neg_ok = False
                                break
                    if neg_ok:
                        converged = True
                        break
            
            if converged:
                # Full verification
                if verify_fast(U3, V3, W3):
                    best_result = (3, U3, V3, W3)
                    break
    
    # If rank 3 wasn't found, use rank 4
    if best_result is not None:
        r, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to rank 4
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
