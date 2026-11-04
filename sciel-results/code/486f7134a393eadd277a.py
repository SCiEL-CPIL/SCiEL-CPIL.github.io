import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n + i, r]
    """
    rank = U.shape[1]
    # Reconstruct tensor
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # The term is u * v * w?
        # Tensor T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # Note: W index is k*n + i, so W_{ki,r} corresponds to element (k,i) in matrix W_r.
        # Let's check the trace convention.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # T_{i,j,k} = A_{ij} B_{jk} C_{ki}^*? No.
        # The tensor is defined by T_{i,j,k} = A_{ij} B_{jk} (C_{ki})?
        # Usually, the multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if i,j,k are valid.
        # Decomposition: M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # Here W is indexed by (k,i) in the flattened vector W[k*n+i].
        # So W_r is a p x n matrix.
        # Let's verify with a simple case.
        # For r, we add U_r \otimes V_r \otimes W_r to T.
        # T_{i,j,k} += U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        
        # Efficient calculation:
        # T_r = np.einsum('ij,jk,ki->ijk', u, v, w)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    
    # Target tensor: T_target[i,j,k] = 1 for all i,j,k
    T_target = np.ones((n, m, p))
    
    diff = np.abs(T - T_target)
    max_diff = np.max(diff)
    return max_diff < 1e-6

def _find_decomposition_3x4x3(max_rank=30, time_limit=10):
    """
    Try to find a rank decomposition for 3x4x3 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 3, 4, 3
    start_time = time.time()
    
    # We try to find a decomposition of rank R
    # Start with a lower bound or a guess.
    # Trivial is 36. Let's try 30, 28, 26...
    for R in range(30, 15, -2):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random values
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Simple gradient descent or just check if random init is close? No.
            # We need an optimizer.
            # Using scipy.optimize.minimize might be slow or fail.
            # Let's use a simple iterative refinement or just check if we can solve the linear system?
            # The problem is non-linear (rank factorization).
            
            # Alternative: Use the fact that we can fix U and V and solve for W?
            # No, all are coupled.
            
            # Given the time constraint and complexity, a robust solver is hard to implement quickly.
            # However, we can try a specific known decomposition if available.
            # For now, let's assume we can't find it quickly and fall back to trivial.
            # But we MUST try to improve.
            
            # Let's try a rank 1 decomposition? No.
            # Let's try to construct a rank 30 decomposition using a block method?
            # 3x4x3 can be split into 2x4x3 and 1x4x3?
            # 2x4x3: Rank?
            # 1x4x3: Rank 1? No, 1x4x3 is just a vector product.
            # C = a * B. a is 1x4, B is 4x3.
            # C_{k} = sum_j a_j B_{jk}.
            # This is a rank 1 tensor? No.
            # T_{j,k} = a_j B_{jk}.
            # Decomposition: U_{j,r} V_{jk,r} W_{k,r}?
            # For 1x4x3, n=1, m=4, p=3.
            # T_{j,k} = sum_r U_{j,r} V_{jk,r} W_{k,r}.
            # This is not a standard matrix multiplication tensor definition.
            # The tensor is for C = A B.
            # If A is 1x4, B is 4x3, C is 1x3.
            # T_{i,j,k} where i=0.
            # T_{0,j,k} = A_{0,j} B_{j,k} C_{k,0}^*?
            # The decomposition is for the operation C = A B.
            # The tensor M_{1,4,3} has entries 1.
            # Rank of M_{1,4,3}?
            # T_{j,k} = sum_r U_{j,r} V_{jk,r} W_{k,r}.
            # We want T_{j,k} = 1.
            # This is not possible with rank 1?
            # If U_{j,1}=1, V_{jk,1}=1, W_{k,1}=1, then T_{j,k}=1.
            # So Rank 1 is sufficient for 1x4x3?
            # Wait, the tensor is M_{n,m,p}.
            # For n=1, m=4, p=3.
            # T_{0,j,k} = 1.
            # Decomposition: sum_r U_{0,j,r} V_{j,k,r} W_{k,0,r}.
            # If we set U_{0,j,1}=1, V_{j,k,1}=1, W_{k,0,1}=1, then sum is 1.
            # So Rank 1 works for 1xm x p?
            # Let's check 2x2x2. Rank 7.
            # Why is 1x2x2 rank 1?
            # T_{0,j,k} = 1.
            # Yes, it's a "slice" of the tensor.
            
            # So, 3x4x3 can be split into 2x4x3 and 1x4x3.
            # Rank(3x4x3) <= Rank(2x4x3) + Rank(1x4x3).
            # Rank(1x4x3) = 1.
            # So we need Rank(2x4x3).
            
            # 2x4x3:
            # Split 2x4x3 into 2x2x3 and 2x2x3?
            # Or 2x4x2 and 2x4x1?
            # Rank(2x4x1) = 1?
            # T_{i,j,0} = 1.
            # Decomposition: sum_r U_{i,j,r} V_{j,0,r} W_{0,i,r}.
            # Set U_{i,j,1}=1, V_{j,0,1}=1, W_{0,i,1}=1.
            # Then T_{i,j,0} = 1.
            # So Rank(2x4x1) = 1.
            
            # So Rank(2x4x3) <= Rank(2x4x2) + Rank(2x4x1).
            # Rank(2x4x1) = 1.
            # Need Rank(2x4x2).
            
            # 2x4x2:
            # Split 2x4x2 into 2x2x2 and 2x2x2.
            # Rank(2x2x2) = 7 (Strassen).
            # So Rank(2x4x2) <= 7 + 7 = 14.
            
            # So Rank(2x4x3) <= 14 + 1 = 15.
            # So Rank(3x4x3) <= 15 + 1 = 16.
            
            # This is a huge improvement over 36!
            # Total Rank for 3x4x6 = 2 * Rank(3x4x3) <= 2 * 16 = 32.
            # Score: 72 / 32 = 2.25.

    return None # Placeholder, will implement the construction below

def _construct_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Rank 7.
    """
    # Strassen's 7 multiplications for 2x2 matrices.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Indices:
    # U: (i*m+j, r)
    # V: (j*p+k, r)
    # W: (k*n+i, r)
    
    # r=0: M1
    # U: a11, a22 -> 1, 1. Indices: 0, 3.
    # V: b11, b22 -> 1, 1. Indices: 0, 3.
    # W: c11, c22 -> 1, 1. Indices: 0, 3. (c11 is k=0,i=0; c22 is k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2
    # U: a21, a22 -> 1, 1. Indices: 2, 3.
    # V: b11 -> 1. Index: 0.
    # W: c21, c22 -> 1, -1. Indices: 2 (k=1,i=0), 3 (k=1,i=1).
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # r=2: M3
    # U: a11 -> 1. Index: 0.
    # V: b12, b22 -> 1, -1. Indices: 1, 3.
    # W: c12, c22 -> 1, 1. Indices: 1 (k=0,i=1), 3 (k=1,i=1).
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: M4
    # U: a22 -> 1. Index: 3.
    # V: b21, b11 -> 1, -1. Indices: 2, 0.
    # W: c11, c21 -> 1, 1. Indices: 0, 2.
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # r=4: M5
    # U: a11, a12 -> 1, 1. Indices: 0, 1.
    # V: b22 -> 1. Index: 3.
    # W: c11, c12 -> -1, 1. Indices: 0, 1.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6
    # U: a21, a11 -> 1, -1. Indices: 2, 0.
    # V: b11, b12 -> 1, 1. Indices: 0, 1.
    # W: c22 -> 1. Index: 3.
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7
    # U: a12, a22 -> 1, -1. Indices: 1, 3.
    # V: b21, b22 -> 1, 1. Indices: 2, 3.
    # W: c11 -> 1. Index: 0.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _make_trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target: 3x4x6
    # Strategy: Split into two 3x4x3 problems.
    # Construct 3x4x3 decomposition using block method.
    # 3x4x3 = (2x4x3) + (1x4x3)
    # 2x4x3 = (2x4x2) + (2x4x1)
    # 2x4x2 = (2x2x2) + (2x2x2)
    # 2x4x1 is rank 1.
    # 1x4x3 is rank 1.
    # 2x2x2 is rank 7.
    
    # Total Rank for 3x4x3 = 7 + 7 + 1 + 1 = 16.
    # Total Rank for 3x4x6 = 16 + 16 = 32.
    
    # We need to assemble these.
    
    def get_decomp_2x2x2():
        return _construct_strassen_2x2x2()
    
    def get_decomp_1xm_p(n, m, p):
        # Rank 1
        U = np.zeros((n*m, 1), dtype=int)
        V = np.zeros((m*p, 1), dtype=int)
        W = np.zeros((p*n, 1), dtype=int)
        U[:, 0] = 1
        V[:, 0] = 1
        W[:, 0] = 1
        return U, V, W

    def combine_decomps(n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
        # Combine two decompositions for (n1+m1, m1+m2, p1+p2)? No.
        # We are splitting the dimensions.
        # For 2x4x2, we split m=4 into m=2 and m=2.
        # A is 2x4, B is 4x2.
        # B = [B1, B2] where B1 is 2x2, B2 is 2x2.
        # C = A [B1, B2] = [A B1, A B2].
        # This is not a direct sum of tensors in the same way.
        # The tensor for 2x4x2 is not the sum of two 2x2x2 tensors directly.
        # We need to map the indices.
        
        # Let's construct 2x4x2 directly from two 2x2x2.
        # U (2x4, rank), V (4x2, rank), W (2x2, rank).
        # Rank = 14.
        # First 2x2x2 (cols 0-1 to 0-1 of B):
        # U1 (2x2, 7), V1 (2x2, 7), W1 (2x2, 7).
        # Second 2x2x2 (cols 2-3 to 0-1 of B):
        # U2 (2x2, 7), V2 (2x2, 7), W2 (2x2, 7).
        
        # U_total (2x4, 14).
        # U_total[:, 0:7] = U1.
        # U_total[:, 7:14] = U2.
        
        # V_total (4x2, 14).
        # V1 is for B1 (rows 0-1 of B).
        # V2 is for B2 (rows 2-3 of B).
        # V_total[0:2, 0:7] = V1.
        # V_total[2:4, 7:14] = V2.
        # Other parts 0.
        
        # W_total (2x2, 14).
        # W_total[:, 0:7] = W1.
        # W_total[:, 7:14] = W2.
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        rank = rank1 + rank2
        n, m, p = 2, 4, 2
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # Fill U
        U[0:2, 0:rank1] = U1
        U[0:2, rank1:rank] = U2
        
        # Fill V
        V[0:2, 0:rank1] = V1
        V[2:4, rank1:rank] = V2
        
        # Fill W
        W[:, 0:rank1] = W1
        W[:, rank1:rank] = W2
        
        return U, V, W

    def get_decomp_2x4x2():
        U1, V1, W1 = get_decomp_2x2x2()
        U2, V2, W2 = get_decomp_2x2x2()
        return combine_decomps(2, 2, 2, 2, 2, 2, U1, V1, W1, U2, V2, W2)

    def get_decomp_2x4x3():
        # 2x4x3 = 2x4x2 + 2x4x1
        U2x4x2, V2x4x2, W2x4x2 = get_decomp_2x4x2()
        U2x4x1, V2x4x1, W2x4x1 = get_decomp_1xm_p(2, 4, 1)
        
        rank1 = U2x4x2.shape[1]
        rank2 = U2x4x1.shape[1]
        rank = rank1 + rank2
        n, m, p = 2, 4, 3
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        U[:, 0:rank1] = U2x4x2
        U[:, rank1:rank] = U2x4x1
        
        V[:, 0:rank1] = V2x4x2
        # V2x4x1 is for p=1. We need to map it to p=3.
        # The 2x4x1 tensor is for the first column of B?
        # No, we split p=3 into p=2 and p=1.
        # B is 4x3. B = [B_2x2, B_2x1]? No, B is 4x3.
        # Split columns of B: B = [B1 (4x2), B2 (4x1)].
        # C = A [B1, B2] = [A B1, A B2].
        # C is 2x3. C = [C1 (2x2), C2 (2x1)].
        # So W (2x3) is split into W1 (2x2) and W2 (2x1).
        
        V[0:4, 0:rank1] = V2x4x2 # V2x4x2 is 4x2.
        # V2x4x1 is 4x1.
        V[0:4, rank1:rank] = V2x4x1
        
        W[:, 0:rank1] = W2x4x2 # W2x4x2 is 2x2.
        W[:, rank1:rank] = W2x4x1 # W2x4x1 is 2x1.
        
        return U, V, W

    def get_decomp_3x4x3():
        # 3x4x3 = 2x4x3 + 1x4x3
        U2x4x3, V2x4x3, W2x4x3 = get_decomp_2x4x3()
        U1x4x3, V1x4x3, W1x4x3 = get_decomp_1xm_p(1, 4, 3)
        
        rank1 = U2x4x3.shape[1]
        rank2 = U1x4x3.shape[1]
        rank = rank1 + rank2
        n, m, p = 3, 4, 3
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # U (3x4).
        # U2x4x3 is 2x4. Map to first 2 rows.
        U[0:2, 0:rank1] = U2x4x3
        # U1x4x3 is 1x4. Map to last 1 row.
        U[2:3, rank1:rank] = U1x4x3
        
        # V (4x3).
        V[:, 0:rank1] = V2x4x3
        V[:, rank1:rank] = V1x4x3
        
        # W (3x3).
        # W2x4x3 is 2x3 (p=3, n=2).
        # W1x4x3 is 1x3 (p=3, n=1).
        # W is 3x3 (p=3, n=3).
        # W2x4x3 occupies rows 0-1 of W.
        # W1x4x3 occupies row 2 of W.
        W[0:2, 0:rank1] = W2x4x3
        W[2:3, rank1:rank] = W1x4x3
        
        return U, V, W

    # Construct 3x4x6
    # 3x4x6 = 3x4x3 + 3x4x3
    U3x4x3_1, V3x4x3_1, W3x4x3_1 = get_decomp_3x4x3()
    U3x4x3_2, V3x4x3_2, W3x4x3_2 = get_decomp_3x4x3()
    
    rank1 = U3x4x3_1.shape[1]
    rank2 = U3x4x3_2.shape[1]
    rank = rank1 + rank2
    n, m, p = 3, 4, 6
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    U[:, 0:rank1] = U3x4x3_1
    U[:, rank1:rank] = U3x4x3_2
    
    # V (4x6).
    # V3x4x3_1 is 4x3. Map to first 3 cols.
    # V3x4x3_2 is 4x3. Map to last 3 cols.
    V[0:4, 0:rank1] = V3x4x3_1
    V[0:4, rank1:rank] = V3x4x3_2
    
    # W (6x3).
    # W3x4x3_1 is 3x3 (p=3, n=3). Map to first 3 rows.
    # W3x4x3_2 is 3x3 (p=3, n=3). Map to last 3 rows.
    W[0:3, 0:rank1] = W3x4x3_1
    W[3:6, rank1:rank] = W3x4x3_2
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to trivial
        U, V, W = _make_trivial(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

