"""Program for matrix multiplication tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # We will build the decomposition recursively or via direct block construction.
    # The tensor for 2x4x7 multiplication can be decomposed by splitting the dimensions.
    # A(2x4) * B(4x7) -> C(2x7)
    
    # We can split B into B1(4x2) and B2(4x5).
    # C = A*B1 + A*B2
    # Rank(C) <= Rank(A*B1) + Rank(A*B2)
    
    # Let's decompose 2x4x2 and 2x4x5.
    
    # 2x4x2: Split A into A1(2x2), A2(2x2). B is 4x2.
    # C = A1*B1_part + A2*B2_part? No, B is shared.
    # Actually, it's easier to split the middle dimension (columns of A / rows of B).
    # A(2x4) = [A_left(2x2) | A_right(2x2)]
    # B(4x7) = [B_top(2x7); B_bottom(2x7)]
    # C = A_left * B_top + A_right * B_bottom
    
    # This splits the problem into two independent 2x2x7 multiplications.
    # Rank(2x4x7) <= 2 * Rank(2x2x7)
    
    # Now we need Rank(2x2x7).
    # Split B(2x7) into B1(2x2) and B2(2x5).
    # Rank(2x2x7) <= Rank(2x2x2) + Rank(2x2x5)
    
    # Rank(2x2x2) is 7 (Strassen).
    # Rank(2x2x5): Split B2(2x5) into B21(2x2) and B22(2x3).
    # Rank(2x2x5) <= Rank(2x2x2) + Rank(2x2x3)
    
    # Rank(2x2x3): Split B(2x3) into B1(2x2) and B2(2x1).
    # Rank(2x2x3) <= Rank(2x2x2) + Rank(2x2x1)
    
    # Rank(2x2x1) is 4 (Schoolbook: 2*2*1 = 4, and it's known to be 4).
    
    # So:
    # R(2,2,1) = 4
    # R(2,2,2) = 7
    # R(2,2,3) <= 7 + 4 = 11
    # R(2,2,5) <= 7 + 11 = 18
    # R(2,2,7) <= 7 + 18 = 25
    # R(2,4,7) <= 2 * 25 = 50
    
    # This gives a rank of 50.
    # We need to construct the actual U, V, W matrices for this decomposition.
    
    # Strategy:
    # 1. Define a function to get the factors for a base case (2x2x1, 2x2x2).
    # 2. Define a function to combine factors for a split (A*B1 + A*B2).
    
    # Let's implement the combination logic.
    # If we have factors for T1 (size n1 x m1 x p1) and T2 (size n2 x m2 x p2),
    # and we are forming T = T1 + T2 where the dimensions align appropriately.
    
    # Actually, it's easier to build up the specific factors for 2x4x7 directly
    # by embedding the smaller decompositions.
    
    # Let's define the base decompositions.
    
    # Base: 2x2x1
    # A(2x2), B(2x1) -> C(2x1)
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # Rank 4.
    # U (4x4), V (2x4), W (2x4)
    # Indices: U is (i, j) where i in 0..1, j in 0..1 -> 4 rows.
    # V is (j, k) where j in 0..1, k in 0..0 -> 2 rows.
    # W is (k, i) where k in 0..0, i in 0..1 -> 2 rows.
    # Wait, the interface is U(n*m, r), V(m*p, r), W(p*n, r).
    # For 2x2x1: n=2, m=2, p=1.
    # U is (4, r), V is (2, r), W is (2, r).
    
    def get_2x2x1():
        r = 4
        U = np.zeros((4, r))
        V = np.zeros((2, r))
        W = np.zeros((2, r))
        # Terms:
        # 1: A[0,0]*B[0,0] -> C[0,0]
        # U[0,0]=1, V[0,0]=1, W[0,0]=1
        # 2: A[0,1]*B[1,0] -> C[0,0]
        # U[1,1]=1, V[1,1]=1, W[0,1]=1
        # 3: A[1,0]*B[0,0] -> C[1,0]
        # U[2,2]=1, V[0,2]=1, W[1,2]=1
        # 4: A[1,1]*B[1,0] -> C[1,0]
        # U[3,3]=1, V[1,3]=1, W[1,3]=1
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    def get_2x2x2():
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        # Strassen decomposition
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices: (i, j) -> i*m+j. m=2.
        # V indices: (j, k) -> j*p+k. p=2.
        # W indices: (k, i) -> k*n+i. n=2.
        
        # Term 1: M1
        # U: A00, A11 -> U[0,0]=1, U[3,0]=1
        # V: B00, B11 -> V[0,0]=1, V[3,0]=1
        # W: C00, C11 -> W[0,0]=1, W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: M2
        # U: A10, A11 -> U[2,1]=1, U[3,1]=1
        # V: B00 -> V[0,1]=1
        # W: C10, C11 -> W[2,1]=1, W[3,1]=-1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 3: M3
        # U: A00 -> U[0,2]=1
        # V: B01, B11 -> V[1,2]=1, V[3,2]=-1
        # W: C00, C01 -> W[0,2]=1, W[1,2]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # Term 4: M4
        # U: A11 -> U[3,3]=1
        # V: B10, B00 -> V[2,3]=1, V[0,3]=-1
        # W: C00, C10 -> W[0,3]=1, W[2,3]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 5: M5
        # U: A00, A01 -> U[0,4]=1, U[1,4]=1
        # V: B11 -> V[3,4]=1
        # W: C00, C01 -> W[0,4]=-1, W[1,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 6: M6
        # U: A10, A00 -> U[2,5]=1, U[0,5]=-1
        # V: B00, B01 -> V[0,5]=1, V[1,5]=1
        # W: C11 -> W[3,5]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 7: M7
        # U: A01, A11 -> U[1,6]=1, U[3,6]=-1
        # V: B10, B11 -> V[2,6]=1, V[3,6]=1
        # W: C00, C11 -> W[0,6]=1, W[3,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Now we need to combine these to form 2x4x7.
    # The structure is:
    # C = A_left * B_top + A_right * B_bottom
    # where A_left is 2x2, B_top is 2x7.
    # And A_right is 2x2, B_bottom is 2x7.
    
    # Let's define a helper to embed a smaller decomposition into a larger one.
    # If we have a decomposition for (n, m, p) with factors U, V, W,
    # and we want to place it at offset (i_offset, j_offset, k_offset) in a larger (N, M, P) problem.
    
    def embed(U_small, V_small, W_small, i_off, j_off, k_off, N, M, P, r_total):
        U_big = np.zeros((N*M, r_total))
        V_big = np.zeros((M*P, r_total))
        W_big = np.zeros((P*N, r_total))
        
        n_s, m_s, p_s = U_small.shape[0]//m_s, U_small.shape[0]//m_s, W_small.shape[0]//n_s
        # Actually, shapes are given.
        # U_small is (n_s*m_s, r_s)
        # V_small is (m_s*p_s, r_s)
        # W_small is (p_s*n_s, r_s)
        
        r_s = U_small.shape[1]
        
        for c in range(r_s):
            for i in range(n_s):
                for j in range(m_s):
                    val = U_small[i*m_s + j, c]
                    if val != 0:
                        U_big[(i + i_off)*M + (j + j_off), c] = val
            
            for j in range(m_s):
                for k in range(p_s):
                    val = V_small[j*p_s + k, c]
                    if val != 0:
                        V_big[(j + j_off)*P + (k + k_off), c] = val
            
            for k in range(p_s):
                for i in range(n_s):
                    val = W_small[k*n_s + i, c]
                    if val != 0:
                        W_big[(k + k_off)*N + (i + i_off), c] = val
        
        return U_big, V_big, W_big

    # We will build 2x4x7 by combining:
    # 1. Left part: 2x2x7 using A[0:2, 0:2] and B[0:2, :]
    # 2. Right part: 2x2x7 using A[0:2, 2:4] and B[2:4, :]
    
    # First, build 2x2x7.
    # 2x2x7 = 2x2x2 (B[0:2]) + 2x2x5 (B[2:7])
    # 2x2x5 = 2x2x2 (B[0:2]) + 2x2x3 (B[2:5])
    # 2x2x3 = 2x2x2 (B[0:2]) + 2x2x1 (B[2:3])
    
    # Let's build the factors for 2x2x7 directly by accumulating.
    
    # Total rank for 2x2x7 is 7 + 18 = 25.
    # Total rank for 2x4x7 is 2 * 25 = 50.
    
    U_total = np.zeros((2*4, 50))
    V_total = np.zeros((4*7, 50))
    W_total = np.zeros((7*2, 50))
    
    # We need to generate the factors for 2x2x7 and then embed them twice.
    
    # Let's write a recursive function to get factors for 2x2xp.
    def get_2x2xp(p):
        # Returns U, V, W for 2x2xp
        # Base cases:
        if p == 1:
            return get_2x2x1()
        if p == 2:
            return get_2x2x2()
        
        # Split p into p1 and p2. Let's pick p1 = min(2, p), p2 = p - p1.
        # Actually, to match the logic above:
        # If p is even, split 2 and p-2?
        # Let's just split into 2 and p-2 if p>=3.
        if p >= 3:
            p1 = 2
            p2 = p - 2
            U1, V1, W1 = get_2x2xp(p1)
            U2, V2, W2 = get_2x2xp(p2)
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            r_total = r1 + r2
            
            U = np.zeros((4, r_total))
            V = np.zeros((2*p, r_total))
            W = np.zeros((p*2, r_total))
            
            # Embed U1, V1, W1
            U[:, :r1] = U1
            V[:, :r1] = V1
            W[:, :r1] = W1
            
            # Embed U2, V2, W2
            # U2 is same as U1 (A is same 2x2)
            U[:, r1:] = U2
            
            # V2 needs to be shifted in k.
            V[:, r1:] = 0
            for c in range(r2):
                for j in range(2):
                    for k in range(p2):
                        V[j*p + (k + p1), r1 + c] = V2[j*p2 + k, c]
                        
            # W2 needs to be shifted in k.
            W[:, r1:] = 0
            for c in range(r2):
                for k in range(p2):
                    for i in range(2):
                        W[(k + p1)*2 + i, r1 + c] = W2[k*2 + i, c]
                        
            return U, V, W
        else:
            # Should not reach here if p is integer >= 1
            raise ValueError("p must be >= 1")

    # Get factors for 2x2x7
    U_2x2x7, V_2x2x7, W_2x2x7 = get_2x2xp(7)
    r_2x2x7 = U_2x2x7.shape[1]
    
    # Now embed this into 2x4x7.
    # Part 1: A[0:2, 0:2] * B[0:2, 0:7]
    # i_off=0, j_off=0, k_off=0
    # Part 2: A[0:2, 2:4] * B[2:4, 0:7]
    # i_off=0, j_off=2, k_off=0
    
    # Note: The dimensions of the big problem are N=2, M=4, P=7.
    # U shape: (2*4, 50) = (8, 50)
    # V shape: (4*7, 50) = (28, 50)
    # W shape: (7*2, 50) = (14, 50)
    
    # Embed Part 1
    for c in range(r_2x2x7):
        for i in range(2):
            for j in range(2):
                U_total[i*4 + j, c] = U_2x2x7[i*2 + j, c]
        
        for j in range(2):
            for k in range(7):
                V_total[j*7 + k, c] = V_2x2x7[j*7 + k, c]
        
        for k in range(7):
            for i in range(2):
                W_total[k*2 + i, c] = W_2x2x7[k*2 + i, c]
                
    # Embed Part 2
    # A indices: i in 0..1, j in 2..3
    # B indices: j in 2..3, k in 0..6
    # C indices: k in 0..6, i in 0..1
    
    for c in range(r_2x2x7):
        for i in range(2):
            for j in range(2):
                U_total[i*4 + (j + 2), r_2x2x7 + c] = U_2x2x7[i*2 + j, c]
        
        for j in range(2):
            for k in range(7):
                V_total[(j + 2)*7 + k, r_2x2x7 + c] = V_2x2x7[j*7 + k, c]
        
        for k in range(7):
            for i in range(2):
                W_total[k*2 + i, r_2x2x7 + c] = W_2x2x7[k*2 + i, c]
                
    # Verify the decomposition
    # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c] should be 1 if (i,j,k) is valid, else 0.
    # Actually, the tensor is defined such that T[i*m+j, j*p+k, k*n+i] = 1.
    # So for each i, j, k, the sum should be 1.
    
    valid = True
    for i in range(2):
        for j in range(4):
            for k in range(7):
                val = np.sum(U_total[i*4+j, :] * V_total[j*7+k, :] * W_total[k*2+i, :])
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(2, 4, 7, 56)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Trim to actual rank used (50)
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def _schoolbook(n, m, p, rank):
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-END
