import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a rank-16 decomposition using the structural bound
    # For 2x4x5, we can potentially use rank 16 = 2*4*(5-1) or similar
    # Let's try a structured approach based on Strassen-like blocks
    
    # First, let's try to construct a rank-16 decomposition directly
    # Using the fact that we can split the 4x5 multiplication into blocks
    
    # For 2x4 * 4x5:
    # We can view this as computing two 2x5 products from a 4x5 matrix
    # Using a rank-16 scheme:
    
    # Let's try a different approach: use the known rank bound
    # For 2x4x5, the rank is at most 2*4*5 = 40 (schoolbook)
    # But we can do better. Let's try rank 16.
    
    # Actually, let's try to find a rank-16 decomposition by solving a system
    # We'll fix U and V to have a specific structure and solve for W
    
    def try_rank16():
        """Try to find a rank-16 decomposition"""
        rank = 16
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Use a structured approach:
        # For each of the 2 rows of A, we need to compute 4*5 = 20 products
        # But we can share computations
        
        # Let's try: for each i in {0,1}, k in {0..4}, we need sum_j A[i,j]*B[j,k]
        # We can use 8 ranks per i (total 16)
        
        # Structure: r = 8*i + t where t in {0..7}
        # For fixed i, we want: sum_j A[i,j]*B[j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
        
        # Let's set up the system:
        # For each (i,k): sum_j A[i,j]*B[j,k] = sum_t W[k,i,t] * (sum_j U[i,j,t]*V[j,k,t])
        
        # This is getting complex. Let's try a simpler structured approach.
        
        # Actually, let's try to use the fact that we can split B into two 4x2.5 matrices
        # Or use a different blocking strategy
        
        # Let's try: split the 4 columns of A into two groups of 2
        # A = [A1, A2] where A1, A2 are 2x2
        # B = [B1; B2] where B1, B2 are 2x5
        
        # Then A*B = A1*B1 + A2*B2
        # Each 2x2 * 2x5 can be done with rank 2*2*5 = 20, but we can do better
        
        # Actually, 2x2 * 2x5: we can use rank 8 (2*2*2) by splitting the 5 into 2+3
        # Hmm, this is getting complicated
        
        # Let's just try random search for rank 16 with a better initialization
        for trial in range(200):
            if time.time() - start_time > 120:
                return None
            
            # Initialize U and V with a sparse pattern
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # For each rank r, assign it to a specific (i, j_group, k_group)
            # i in {0,1}, j_group in {0,1} (split 4 into 2+2), k_group in {0,1,2} (split 5 into 2+2+1)
            # Total: 2*2*3 = 12, but we have 16 ranks
            
            # Let's use a different structure:
            # r = 4*i + 2*jg + kg where i in {0,1}, jg in {0,1}, kg in {0,1,2}
            # That's 12 ranks. Let's add 4 more for cross terms.
            
            # Actually, let's just do random search with a better verify
            
            for r in range(rank):
                # Each rank handles a subset of the computation
                # Assign random but structured values
                for idx in range(n*m):
                    # Sparse: only a few non-zero entries per column
                    if random.random() < 0.3:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(m*p):
                    if random.random() < 0.3:
                        V[idx, r] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p*n, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    b = np.ones(m)
                    
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        valid = False
                        break
                    
                    if not np.allclose(A @ w, b, atol=1e-9):
                        valid = False
                        break
                    
                    for r in range(rank):
                        val = w[r]
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) > 1e-6 and abs(val - nearest_half) > 1e-6:
                            valid = False
                            break
                    
                    if not valid:
                        break
                    
                    W[k*n+i, :] = w
            
            if valid and verify(U, V, W):
                for idx in range(p*n):
                    for r in range(rank):
                        val = W[idx, r]
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) < abs(val - nearest_half):
                            W[idx, r] = nearest_int
                        else:
                            W[idx, r] = nearest_half
                return (U, V, W)
        
        return None
    
    # Try rank 16 first
    result = try_rank16()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 20
    for rank in range(17, 25):
        if time.time() - start_time > 120:
            break
        
        found = False
        for trial in range(50):
            if time.time() - start_time > 120:
                break
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            for r in range(rank):
                for idx in range(n*m):
                    if random.random() < 0.3:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(m*p):
                    if random.random() < 0.3:
                        V[idx, r] = random.choice([-1, 1])
            
            W = np.zeros((p*n, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    b = np.ones(m)
                    
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        valid = False
                        break
                    
                    if not np.allclose(A @ w, b, atol=1e-9):
                        valid = False
                        break
                    
                    for r in range(rank):
                        val = w[r]
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) > 1e-6 and abs(val - nearest_half) > 1e-6:
                            valid = False
                            break
                    
                    if not valid:
                        break
                    
                    W[k*n+i, :] = w
            
            if valid and verify(U, V, W):
                found = True
                for idx in range(p*n):
                    for r in range(rank):
                        val = W[idx, r]
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) < abs(val - nearest_half):
                            W[idx, r] = nearest_int
                        else:
                            W[idx, r] = nearest_half
                break
        
        if found:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: trivial rank 40
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
