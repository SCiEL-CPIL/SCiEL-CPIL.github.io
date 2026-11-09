"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # 1. Schoolbook Fallback (Rank = n*m*p = 48)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # 2. Verification Logic
    # The tensor T[i,j,k] must be 1 for all i,j,k.
    # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # Note: The problem states "exact tensor decomposition". 
    # Usually this implies integer or half-integer entries.
    # We will search for integer solutions first, then half-integers.
    
    def verify(U, V, W, tol=1e-9):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = U.shape[1]
        # Reshape for easier access
        # U[i*m+j, r]
        # V[j*p+k, r]
        # W[k*n+i, r]
        
        # We can vectorize the check using einsum or loops.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Flatten indices for numpy
        # U_flat: (n*m, rank)
        # V_flat: (m*p, rank)
        # W_flat: (p*n, rank)
        
        # Construct T
        T = np.zeros((n, m, p), dtype=np.float64)
        
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            # Reshape
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is a tensor contraction.
            # We can use einsum: 'ijr,jkr,kir->ijk' but we are summing over r manually here.
            # Let's do it efficiently with broadcasting if possible, or just loops since n,m,p are small.
            # n=2, m=4, p=6. Total elements 48. Rank <= 48.
            # Loops are fine.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    # 3. Search Strategy
    # The rank of the matrix multiplication tensor for dimensions (n, m, p) is at least max(n, m, p) * min(n, m, p)? No.
    # Lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p)? 
    # Actually, the rank is at least the "slice rank" or similar.
    # For 2x4x6, schoolbook is 48.
    # Strassen-like algorithms can reduce this.
    # However, finding an exact low-rank decomposition is hard.
    
    # Strategy: 
    # 1. Try to find a decomposition with rank < 48 using random linear combinations of schoolbook terms?
    #    This is essentially finding a basis for the space of rank-1 tensors that sums to T.
    #    This is equivalent to finding a sparse representation.
    
    # Alternative Strategy: 
    # Use the fact that the tensor is "separable" in a specific way? No, it's not.
    
    # Let's try a heuristic search for small ranks.
    # We know rank >= 4? 
    # For 1xN x NxP, rank is N.
    # For 2x2x2, rank is 7.
    # For 2x4x6, let's estimate.
    # If we split 4 into 2+2, and 6 into 3+3?
    # Block diagonal?
    # If A = [[A1, A2], [A3, A4]] where A1 is 2x2, A2 is 2x2, etc.
    # This doesn't help directly unless the structure is block diagonal.
    
    # Let's try a randomized search for integer solutions with small rank.
    # If we assume the entries are in {-1, 0, 1}, we can do a local search.
    # But the space is huge.
    
    # Better approach: 
    # The tensor T is 1 everywhere.
    # We want sum_r u_r v_r w_r = T.
    # This is a tensor completion problem.
    
    # Let's try to improve the schoolbook rank by combining terms.
    # Schoolbook terms are e_i e_j e_k.
    # Can we replace a set of terms with a smaller set?
    # Example: In 2x2x2, we replace 8 terms with 7.
    # The relation is: e_0 e_0 e_0 + e_1 e_1 e_1 + ... 
    # Actually, Strassen's algorithm uses specific linear combinations.
    
    # Since we don't have a specific algorithm for 2x4x6, we will attempt a numerical optimization
    # to find a low rank decomposition, then snap to integers.
    
    # Optimization:
    # Minimize || sum_r u_r v_r w_r - T ||_F^2
    # Subject to rank r.
    
    # We will try ranks from a lower bound up to 48.
    # Lower bound for rank of n x m x p matmul tensor:
    # rank >= max(n*m, m*p, p*n) is not correct.
    # rank >= n*m*p / min(n,m,p)? No.
    # rank >= max(n, m, p) is a weak bound.
    # For 2x4x6, let's start trying from rank 20 upwards?
    # Actually, let's just try to beat 48.
    
    # Given the time constraint and the difficulty of finding exact integer solutions for arbitrary ranks,
    # and the fact that the "seed" is 48, any improvement is good.
    # However, random gradient descent on tensor rank is non-convex and often fails to find exact integer solutions.
    
    # Let's try a specific structural improvement.
    # If we can factor the middle dimension?
    # m=4.
    # Can we use rank 4 * something?
    # The "slice rank" of T is 4? No.
    
    # Let's try to use the "subtensor" method.
    # If we can decompose 2x4x6 into smaller blocks.
    # 2x4x6 = 2x2x6 + 2x2x6?
    # Rank(2x4x6) <= Rank(2x2x6) + Rank(2x2x6) = 2 * Rank(2x2x6).
    # Rank(2x2x6) <= Rank(2x2x2) * 3? No.
    # Rank(2x2x6) <= 7 * 3 = 21? (If we treat it as 3 blocks of 2x2x2).
    # Then Rank(2x4x6) <= 42.
    
    # Let's try to construct a decomposition of rank 42 using 3 blocks of Strassen's 2x2x2 algorithm.
    # This is a valid structural improvement!
    
    # Strassen's algorithm for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A00)(B00 + B10)
    # M7 = (A10 - A11)(B01 + B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M1 - M2 + M6
    # C11 = M2 - M3 + M1 + M5  (Wait, standard Strassen formulas vary, let's use a standard set)
    
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 - P4 + P5 + P7
    # C12 = P3 + P4
    # C21 = P1 - P2 + P6
    # C22 = P2 - P3 + P1 + P5
    
    # We can implement this for 2x2x2 blocks.
    # We need to map the 2x4x6 problem to three 2x2x2 problems?
    # A is 2x4. B is 4x6.
    # Let A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # Let B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6.
    # C = A_left B_top + A_right B_bottom.
    # This splits into two 2x2x6 multiplications.
    # Rank(2x2x6) <= Rank(2x2x2) * 3?
    # B_top is 2x6. We can split B_top into [B1, B2, B3] where each is 2x2.
    # A_left is 2x2.
    # A_left B_top = A_left B1 + A_left B2 + A_left B3.
    # This is 3 separate 2x2x2 multiplications.
    # So C = A_left B1 + A_left B2 + A_left B3 + A_right B4 + A_right B5 + A_right B6.
    # This is 6 separate 2x2x2 multiplications.
    # Rank = 6 * 7 = 42.
    
    # This is a valid decomposition of rank 42.
    # Let's construct it.
    
    def get_strassen_block(U, V, W, offset_u, offset_v, offset_w, rank_offset):
        """
        Adds Strassen terms for a 2x2x2 block.
        U, V, W are the global matrices.
        offset_u, offset_v, offset_w are the starting indices in the flattened U, V, W.
        rank_offset is the starting column index in U, V, W.
        """
        # Strassen coefficients
        # P1 = (A11+A22)(B11+B22)
        # Indices for A (2x2):
        # A11 is at (0,0) -> flat 0
        # A22 is at (1,1) -> flat 3
        # A12 is at (0,1) -> flat 1
        # A21 is at (1,0) -> flat 2
        
        # Indices for B (2x2):
        # B11 is at (0,0) -> flat 0
        # B22 is at (1,1) -> flat 3
        # B12 is at (0,1) -> flat 1
        # B21 is at (1,0) -> flat 2
        
        # Indices for C (2x2) -> W (p*n)
        # C11 is at (0,0) -> W flat index 0*n + 0 = 0
        # C12 is at (0,1) -> W flat index 1*n + 0 = n
        # C21 is at (1,0) -> W flat index 0*n + 1 = 1
        # C22 is at (1,1) -> W flat index 1*n + 1 = n+1
        
        # Wait, W is (p*n, rank). Index is k*n + i.
        # C is 2x2. i in 0..1, k in 0..1.
        # C[0,0] -> k=0, i=0 -> idx 0
        # C[0,1] -> k=1, i=0 -> idx 1
        # C[1,0] -> k=0, i=1 -> idx 2
        # C[1,1] -> k=1, i=1 -> idx 3
        
        # Let's define the 7 multiplications for Strassen.
        # We need to map these to the U, V, W vectors.
        # For each multiplication Pj, we add a rank-1 term to the global decomposition.
        # The term is u_j v_j w_j.
        # u_j is a vector of length 4 (2x2 A).
        # v_j is a vector of length 4 (2x2 B).
        # w_j is a vector of length 4 (2x2 C).
        
        # P1 = (A11+A22)(B11+B22)
        # u1: A11=1, A22=1. Vector [1, 0, 0, 1]
        # v1: B11=1, B22=1. Vector [1, 0, 0, 1]
        # w1: Contributes to C.
        # C11 += P1, C22 += P1.
        # w1: C11=1, C22=1. Vector [1, 0, 0, 1] (indices 0 and 3)
        
        # P2 = (A21+A22)B11
        # u2: A21=1, A22=1. Vector [0, 0, 1, 1]
        # v2: B11=1. Vector [1, 0, 0, 0]
        # w2: C21 += P2? No, C21 = P1 - P2 + P6. So -P2.
        #      C22 += P2? No, C22 = P2 - P3 + P1 + P5. So +P2.
        # w2: C21=-1, C22=1. Vector [0, -1, 1, 0] (indices 1 and 2)
        #      Wait, C21 is idx 2? No.
        #      C[i,k]. i=row, k=col.
        #      C[1,0] is row 1, col 0. k=0, i=1. W idx = 0*2+1 = 1.
        #      C[1,1] is row 1, col 1. k=1, i=1. W idx = 1*2+1 = 3.
        #      So w2: idx 1 is -1, idx 3 is 1. Vector [0, -1, 0, 1].
        
        # P3 = A11(B21-B22)
        # u3: A11=1. Vector [1, 0, 0, 0]
        # v3: B21=1, B22=-1. Vector [0, 0, 1, -1]
        # w3: C12 += P3? C12 = P3 + P4. So +P3.
        #      C22 += P3? C22 = P2 - P3 + ... So -P3.
        # w3: C12=1, C22=-1.
        #      C12 is row 0, col 1. k=1, i=0. W idx = 1*2+0 = 2.
        #      C22 is row 1, col 1. k=1, i=1. W idx = 3.
        #      Vector [0, 0, 1, -1].
        
        # P4 = A22(B12-B11)
        # u4: A22=1. Vector [0, 0, 0, 1]
        # v4: B12=1, B11=-1. Vector [-1, 1, 0, 0]
        # w4: C12 += P4? C12 = P3 + P4. So +P4.
        #      C11 += P4? C11 = P1 - P4 + ... So -P4.
        # w4: C12=1, C11=-1.
        #      C12 idx 2. C11 idx 0.
        #      Vector [-1, 0, 1, 0].
        
        # P5 = (A11+A12)B22
        # u5: A11=1, A12=1. Vector [1, 1, 0, 0]
        # v5: B22=1. Vector [0, 0, 0, 1]
        # w5: C11 += P5? C11 = ... + P5. So +P5.
        #      C22 += P5? C22 = ... + P5. So +P5.
        # w5: C11=1, C22=1.
        #      Vector [1, 0, 0, 1].
        
        # P6 = (A21-A11)(B11+B12)
        # u6: A21=1, A11=-1. Vector [-1, 0, 1, 0]
        # v6: B11=1, B12=1. Vector [1, 1, 0, 0]
        # w6: C21 += P6? C21 = ... + P6. So +P6.
        # w6: C21=1.
        #      C21 idx 1.
        #      Vector [0, 1, 0, 0].
        
        # P7 = (A12-A22)(B21+B22)
        # u7: A12=1, A22=-1. Vector [0, 1, 0, -1]
        # v7: B21=1, B22=1. Vector [0, 0, 1, 1]
        # w7: C11 += P7? C11 = ... + P7. So +P7.
        #      C22 += P7? C22 = ... + P7. So +P7.
        # w7: C11=1, C22=1.
        #      Vector [1, 0, 0, 1].
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, -1, 0, 1]), # P2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, -1]), # P3
            ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 1, 0]), # P4
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]),   # P5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),  # P6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])   # P7
        ]
        
        for u_vec, v_vec, w_vec in terms:
            for idx, val in enumerate(u_vec):
                if val != 0:
                    U[offset_u + idx, rank_offset] += val
            for idx, val in enumerate(v_vec):
                if val != 0:
                    V[offset_v + idx, rank_offset] += val
            for idx, val in enumerate(w_vec):
                if val != 0:
                    W[offset_w + idx, rank_offset] += val
            rank_offset += 1
        
        return rank_offset

    # Construct the 6-block Strassen decomposition
    # A (2x4) split into A1 (2x2), A2 (2x2)
    # B (4x6) split into B1 (2x2), B2 (2x2), B3 (2x2) for A1
    #             and B4 (2x2), B5 (2x2), B6 (2x2) for A2
    
    # Total rank = 6 * 7 = 42.
    
    if time.time() < deadline - 10:
        rank_total = 42
        U = np.zeros((n*m, rank_total), dtype=np.int32)
        V = np.zeros((m*p, rank_total), dtype=np.int32)
        W = np.zeros((p*n, rank_total), dtype=np.int32)
        
        current_rank = 0
        
        # Block 1: A1 * B1
        # A1 is A[0:2, 0:2]. U offset 0.
        # B1 is B[0:2, 0:2]. V offset 0.
        # C1 is C[0:2, 0:2]. W offset 0.
        current_rank = get_strassen_block(U, V, W, 0, 0, 0, current_rank)
        
        # Block 2: A1 * B2
        # A1 is A[0:2, 0:2]. U offset 0.
        # B2 is B[0:2, 2:4]. V offset 2*6 + 2 = 14.
        # C2 is C[0:2, 2:4]. W offset 2*2 + 0 = 4. (k=2, i=0 -> 2*2+0=4)
        current_rank = get_strassen_block(U, V, W, 0, 14, 4, current_rank)
        
        # Block 3: A1 * B3
        # A1 is A[0:2, 0:2]. U offset 0.
        # B3 is B[0:2, 4:6]. V offset 2*6 + 4 = 16.
        # C3 is C[0:2, 4:6]. W offset 4*2 + 0 = 8.
        current_rank = get_strassen_block(U, V, W, 0, 16, 8, current_rank)
        
        # Block 4: A2 * B4
        # A2 is A[0:2, 2:4]. U offset 2*4 + 2 = 10.
        # B4 is B[2:4, 0:2]. V offset 2*6 + 0 = 12.
        # C4 is C[0:2, 0:2]. W offset 0.
        current_rank = get_strassen_block(U, V, W, 10, 12, 0, current_rank)
        
        # Block 5: A2 * B5
        # A2 is A[0:2, 2:4]. U offset 10.
        # B5 is B[2:4, 2:4]. V offset 12 + 2 = 14.
        # C5 is C[0:2, 2:4]. W offset 4.
        current_rank = get_strassen_block(U, V, W, 10, 14, 4, current_rank)
        
        # Block 6: A2 * B6
        # A2 is A[0:2, 2:4]. U offset 10.
        # B6 is B[2:4, 4:6]. V offset 12 + 4 = 16.
        # C6 is C[0:2, 4:6]. W offset 8.
        current_rank = get_strassen_block(U, V, W, 10, 16, 8, current_rank)
        
        # Verify
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # If verification fails, fall back to schoolbook
            pass

    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
