import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def verify(U, V, W):
        """Check if decomposition is exact."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        C = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
        return np.allclose(C, T, atol=1e-9)
    
    def snap_to_half_int(arr):
        return np.round(np.array(arr) * 2) / 2.0
    
    # Fallback: schoolbook rank 24
    def schoolbook():
        rank = nm * mp * pn // (n * m * p) * (n * m * p)  # just nm*mp*pn/(n*m*p)... no
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = 110.0
    
    # Try ranks from 12 up to 23
    # For each rank, try to find a valid decomposition
    
    # Strategy: Use the fact that we can express the multiplication as
    # a sum of rank-1 tensors. Try structured initializations.
    
    # Known: rank(2x3x4) is at most 19 (by known results)
    # Let's try to find solutions with small half-integer coefficients
    
    # Approach: Try random sparse matrices with entries from {-1, -0.5, 0, 0.5, 1}
    # and verify.
    
    candidates = []
    
    for rank in range(12, 24):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random attempts for this rank
        attempts = max(100, int((deadline - time.time()) / max(rank - 11, 1) / 0.001))
        attempts = min(attempts, 5000)
        
        found = False
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse U, V, W with half-integer entries
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            for c in range(rank):
                # Each rank-1 term: choose which (i,j), (j,k), (k,i) are active
                # For a valid term, we need consistency: the j index in U must match j in V,
                # and k in V must match k in W, and i in W must match i in U.
                
                # Choose random i, j, k
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Assign random half-integer weights
                w_u = np.random.choice([-1, -0.5, 0.5, 1])
                w_v = np.random.choice([-1, -0.5, 0.5, 1])
                w_w = np.random.choice([-1, -0.5, 0.5, 1])
                
                U[u_idx, c] = w_u
                V[v_idx, c] = w_v
                W[w_idx, c] = w_w
            
            # Check if this decomposes T
            if verify(U, V, W):
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no random solution found, try a more structured approach
    # Use the block structure: A is 2x3, B is 3x4
    # C[0,:] = A[0,:]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
    # C[1,:] = A[1,:]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
    
    # This is just schoolbook. Let's try Strassen-like tricks.
    
    # Alternative: try to use the known rank-19 decomposition if we can find it
    # For now, return schoolbook as fallback
    return schoolbook()
