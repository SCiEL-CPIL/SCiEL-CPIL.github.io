import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    
    # We split the problem into two parts based on the columns of B (which is m x p)
    # Part 1: B cols 0..2 (3 cols) -> M_{3,3,3}
    # Part 2: B cols 3..6 (4 cols) -> M_{3,3,4}
    
    # Step 1: Find a low-rank decomposition for M_{3,3,3}
    # T_333 has shape (n*m, m*p1, p1*n) = (9, 9, 9)
    # Indices: A(i,k), B(k,j), C(i,j)
    # T_333[i*m+k, k*p1+j, j*n+i] = 1
    
    p1 = 3
    T_333 = np.zeros((n*m, m*p1, p1*n), dtype=np.float64)
    for i in range(n):
        for k in range(m): # k is the summation index, which is the col of A / row of B
            for j in range(p1):
                T_333[i*m+k, k*p1+j, j*n+i] = 1.0
                
    # Note: The standard tensor definition for M_{n,m,p} is T[i,k, k,j, i,j] = 1.
    # Here m=3, p1=3.
    
    U_333 = None
    V_333 = None
    W_333 = None
    rank_333 = n * m * p1 # 27
    
    # Try to find a decomposition with rank < 27
    for R in range(23, 28):
        if time.time() > deadline - 20:
            break
            
        # Flatten the tensor
        T_flat = T_333.flatten()
        
        # Define the function to reconstruct the tensor
        def func(params):
            # params contains U, V, W flattened
            # U: n*m x R
            # V: m*p1 x R
            # W: p1*n x R
            size_U = n * m * R
            size_V = m * p1 * R
            size_W = p1 * n * R
            
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:]
            
            U = U_flat.reshape(n*m, R)
            V = V_flat.reshape(m*p1, R)
            W = W_flat.reshape(p1*n, R)
            
            # Compute T_comp = sum_r u_r v_r^T w_r^T ?
            # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            return T_comp.flatten() - T_flat
        
        # Initial guess
        # Random initialization
        np.random.seed(42)
        x0 = np.random.randn(n*m*R + m*p1*R + p1*n*R) * 0.1
        
        try:
            result = least_squares(func, x0, method='lm', max_nfev=10000)
            if result.cost < 1e-10:
                # Extract U, V, W
                size_U = n * m * R
                size_V = m * p1 * R
                size_W = p1 * n * R
                
                U_flat = result.x[:size_U]
                V_flat = result.x[size_U:size_U+size_V]
                W_flat = result.x[size_U+size_V:]
                
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p1, R)
                W = W_flat.reshape(p1*n, R)
                
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                T_comp = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                if np.allclose(T_comp, T_333, atol=1e-6):
                    U_333 = U_snap
                    V_333 = V_snap
                    W_333 = W_snap
                    rank_333 = R
                    break
        except Exception:
            continue
            
    # If no better rank found, use trivial
    if U_333 is None:
        rank_333 = n * m * p1
        U_333 = np.zeros((n*m, rank_333))
        V_333 = np.zeros((m*p1, rank_333))
        W_333 = np.zeros((p1*n, rank_333))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p1):
                    U_333[i*m+k, col] = 1
                    V_333[k*p1+j, col] = 1
                    W_333[j*n+i, col] = 1
                    col += 1

    # Step 2: Construct the decomposition for M_{3,3,4} (Part 2)
    # This is M_{3,3,3} (for the first 3 cols of B2? No, B2 has 4 cols)
    # B2 is 3x4.
    # We can split B2 into B2a (3x3) and B2b (3x1).
    # But we already have a decomposition for M_{3,3,3}.
    # The tensor for Part 2 is T_2.
    # T_2 corresponds to A (3x3) * B2 (3x4) -> C2 (3x4).
    # This is equivalent to M_{3,3,4}.
    # We can decompose M_{3,3,4} as M_{3,3,3} (cols 0-2 of B2) + M_{3,3,1} (col 3 of B2).
    # Wait, B2 has 4 columns. Let's call them j=0,1,2,3 relative to B2.
    # The decomposition for the 3x3 part (j=0,1,2) is the same as T_333 but with shifted indices.
    # The decomposition for the 3x1 part (j=3) is trivial (rank 9).
    
    p2 = 4
    rank_334 = rank_333 + n * m * 1 # 9
    
    U_334 = np.zeros((n*m, rank_334))
    V_334 = np.zeros((m*p2, rank_334))
    W_334 = np.zeros((p2*n, rank_334))
    
    # Copy the 3x3 part decomposition into the first rank_333 columns
    # The indices for B2 in the 3x3 part are k (row of B2) and j (col 0..2 of B2).
    # In the full B matrix, these are rows k and cols 3+j.
    # So V_334[k*p2 + (3+j), r] = V_333[k*3 + j, r]
    
    for r in range(rank_333):
        for a in range(n*m):
            U_334[a, r] = U_333[a, r]
        for k in range(m):
            for j in range(p1):
                V_334[k*p2 + (3+j), r] = V_333[k*p1 + j, r]
        for j in range(p1):
            for i in range(n):
                W_334[(3+j)*n + i, r] = W_333[j*n + i, r]
                
    # Add the 3x1 part (trivial)
    col = rank_333
    for i in range(n):
        for k in range(m):
            for j in range(3, p2): # j=3 only
                U_334[i*m+k, col] = 1
                V_334[k*p2 + j, col] = 1
                W_334[j*n + i, col] = 1
                col += 1

    # Step 3: Combine Part 1 and Part 2
    # Part 1: B cols 0..2.
    # Part 2: B cols 3..6.
    # Total Rank = rank_333 + rank_334
    
    total_rank = rank_333 + rank_334
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill Part 1
    # U is same for both parts (A is shared)
    # V_1 uses B cols 0..2
    # W_1 uses C cols 0..2
    
    for r in range(rank_333):
        for a in range(n*m):
            U[a, r] = U_333[a, r]
        for k in range(m):
            for j in range(p1):
                V[k*p + j, r] = V_333[k*p1 + j, r]
        for j in range(p1):
            for i in range(n):
                W[j*n + i, r] = W_333[j*n + i, r]
                
    # Fill Part 2
    # U is same
    # V_2 uses B cols 3..6
    # W_2 uses C cols 3..6
    
    for r in range(rank_334):
        for a in range(n*m):
            U[a, rank_333 + r] = U_334[a, r]
        for k in range(m):
            for j in range(p2):
                V[k*p + (3+j), rank_333 + r] = V_334[k*p2 + j, r]
        for j in range(p2):
            for i in range(n):
                W[(3+j)*n + i, rank_333 + r] = W_334[j*n + i, r]

    # Verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i*m+k, k*p+j, j*n+i] = 1.0
                
    if not np.allclose(T_comp, T_target, atol=1e-6):
        # Fallback to trivial if verification fails
        # Reconstruct trivial
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m+k, col] = 1
                    V[k*p+j, col] = 1
                    W[j*n+i, col] = 1
                    col += 1
                    U_arr = U
                    V_arr = V
                    W_arr = W
        # Re-verify
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if not np.allclose(T_comp, T_target, atol=1e-6):
             raise ValueError("Verification failed")

    return {"U": U, "V": V, "W": W}
