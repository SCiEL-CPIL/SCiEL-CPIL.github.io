import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # The tensor T_{i,j,k} is 1 if we are computing C[i,k] from A[i,j]B[j,k]
    # Standard matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor indices: i in [0,n), j in [0,m), k in [0,p)
    # The task specifies output factor W follows trace convention W[k*n+i, r].
    # U shape: (n*m, rank), index i*m+j
    # V shape: (m*p, rank), index j*p+k
    # W shape: (p*n, rank), index k*n+i
    
    # We want to find rank r and matrices U, V, W such that:
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Note: The standard tensor decomposition for matrix multiplication M_{n,m,p} 
    # usually has the form:
    # T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
    # Here the indices for W are k*n+i.
    
    # Strategy:
    # 1. Try to find a low rank decomposition using SDP relaxation or direct search.
    # 2. Since we need exact integer/half-integer entries, we can search for a 
    #    decomposition with small rational numbers.
    # 3. A known lower bound for M_{2,4,6} is not trivially 48. 
    #    M_{2,2,2} has rank 7 (Strassen).
    #    M_{2,4,6} can be viewed as block multiplication.
    #    Let's try to construct a decomposition based on Strassen-like identities 
    #    or by solving a linear system for a guessed rank.
    
    # Given the complexity of exact SDP with rational constraints, 
    # and the 120s limit, a robust approach for small dimensions is to 
    # try to find a decomposition of rank R by solving a linear system 
    # for the entries of U, V, W, fixing one matrix to simplify the non-linearity.
    
    # However, the non-linearity (product of 3 variables) makes it hard.
    # Alternative: Use the fact that for small ranks, we can try to find 
    # a decomposition with entries in {0, 1, -1} or similar.
    
    # Let's try a specific structured approach:
    # M_{2,4,6} is equivalent to multiplying a 2x4 matrix by a 4x6 matrix.
    # This can be split into two M_{2,2,3} problems? Or M_{2,4,3} + M_{2,4,3}?
    # Actually, M_{n, m, p} rank is at most M_{n, m, p1} + M_{n, m, p2} if p=p1+p2.
    # Rank(M_{2,4,6}) <= Rank(M_{2,4,3}) + Rank(M_{2,4,3}).
    # Rank(M_{2,4,3}) <= Rank(M_{2,2,3}) + Rank(M_{2,2,3}).
    # What is Rank(M_{2,2,3})?
    # It is known that Rank(M_{2,2,2}) = 7.
    # Rank(M_{2,2,3}) is at most 10 (trivial 2*2*3=12, Strassen-like?).
    # Actually, there are known results. 
    # Let's try to find a decomposition of rank 42 (from inspiration) or lower.
    
    # The inspiration mentions rank 42. 
    # 48 is the trivial rank.
    # Let's try to verify if a rank 42 decomposition exists with simple entries.
    
    # A more direct approach: 
    # We can formulate the problem as finding a matrix X of size (n*m*p) x (n*m*p) 
    # but that's not right.
    
    # Let's use the "linearization" trick for small ranks.
    # If we assume the entries are from a small set, we can search.
    # But the space is large.
    
    # Better: Use the SDP relaxation idea to get a lower bound, but we need an exact solution.
    # For M_{2,4,6}, let's try to construct a rank 42 decomposition using block Strassen.
    
    # Block Strassen for 2x2 blocks:
    # If we view A as 1x2 blocks of 2x2, and B as 2x3 blocks of 2x2?
    # A is 2x4 -> 1x2 blocks of 2x2.
    # B is 4x6 -> 2x3 blocks of 2x2.
    # C is 2x6 -> 1x3 blocks of 2x2.
    
    # Let A = [A1, A2] where A1, A2 are 2x2.
    # Let B = [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2.
    # C = [C1, C2, C3] where Ci are 2x2.
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # C3 = A1*B13 + A A2*B23
    
    # We need to compute 3 products of 2x2 matrices? No, 6 products.
    # Using Strassen for each 2x2 multiplication?
    # Each 2x2 multiplication has rank 7.
    # We have 6 such multiplications. Total rank 6*7 = 42.
    # This matches the inspiration score of 1.1429 (48/42 = 1.1428...).
    
    # Can we do better?
    # Maybe we can share some of the Strassen intermediate sums?
    # The 6 multiplications are:
    # 1. A1*B11
    # 2. A2*B21
    # 3. A1*B12
    # 4. A2*B22
    # 5. A1*B13
    # 6. A2*B23
    
    # Notice that A1 is used in 1, 3, 5. A2 is used in 2, 4, 6.
    # The right operands are all different.
    # It's not immediately obvious how to share Strassen steps across different right operands 
    # unless we combine them.
    
    # Let's try to combine B11, B12, B13 into a single matrix B1 = [B11, B12, B13] (2x6).
    # Then C1, C2, C3 come from A1 * B1.
    # This is a 2x2 * 2x6 multiplication.
    # Rank(M_{2,2,6})?
    # Similarly for A2 * B2 where B2 = [B21, B22, B23].
    
    # So Rank(M_{2,4,6}) <= Rank(M_{2,2,6}) + Rank(M_{2,2,6}).
    # What is Rank(M_{2,2,6})?
    # Trivial: 2*2*6 = 24.
    # Strassen-based: 3 * Rank(M_{2,2,2}) = 3 * 7 = 21?
    # If we split 2x6 into three 2x2 blocks, we do 3 Strassen multiplications.
    # Total rank 2 * 21 = 42.
    
    # Is Rank(M_{2,2,6}) < 21?
    # Lower bound for M_{2,2,p} is at least 2p - 1? Or something similar?
    # For p=2, rank 7. 2*2-1=3. No.
    # For p=1, M_{2,2,1} is just a linear map, rank 4.
    # For p=3, M_{2,2,3}.
    
    # Let's check if we can find a rank < 42 decomposition.
    # One known result: Rank(M_{n,m,p}) >= max(n*m, m*p, p*n, n+m+p-1).
    # Here max(8, 24, 12, 2+4+6-1=11) = 24.
    # So rank >= 24.
    
    # Let's try to find a rank 30 or 36 decomposition.
    # If we can't find one quickly, we return the rank 42 one.
    
    # Implementation of Rank 42 decomposition:
    # We need to define U, V, W for the 6 Strassen multiplications.
    
    def get_strassen_2x2():
        # Returns U, V, W for M_{2,2,2} with rank 7.
        # Indices:
        # U: 2x2 = 4 entries, indexed by (i,j) -> i*2+j
        # V: 2x2 = 4 entries, indexed by (j,k) -> j*2+k
        # W: 2x2 = 4 entries, indexed by (k,i) -> k*2+i
        # Strassen formulas for C = AB:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[0, 2] = -1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W
    
    U_s, V_s, W_s = get_strassen_2x2()
    
    # Now construct M_{2,4,6}
    # A is 2x4. Split into A1 (cols 0-1) and A2 (cols 2-3).
    # B is 4x6. Split rows into top 2 (rows 0-1) and bottom 2 (rows 2-3).
    # B1 (rows 0-1, cols 0-5) -> B11, B12, B13 (2x2 each)
    # B2 (rows 2-3, cols 0-5) -> B21, B22, B23 (2x2 each)
    
    # We perform 6 Strassen multiplications:
    # 1. A1 * B11 -> C11
    # 2. A2 * B21 -> C11
    # 3. A1 * B12 -> C12
    # 4. A2 * B22 -> C12
    # 5. A1 * B13 -> C13
    # 6. A2 * B23 -> C13
    
    # Total rank = 6 * 7 = 42.
    
    Rank = 42
    U = np.zeros((8, Rank))
    V = np.zeros((24, Rank))
    W = np.zeros((12, Rank))
    
    # Helper to map local 2x2 indices to global indices
    # U local (i,j) -> global (i, j) if A1, or (i, j+2) if A2
    # V local (j,k) -> global (j, k) if B1 (rows 0-1), or (j+2, k) if B2 (rows 2-3)
    # W local (k,i) -> global (k, i)
    
    def add_strassen_block(U_g, V_g, W_g, u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, col_offset):
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j, r]
                    if val != 0:
                        U_g[(u_offset_i + i) * 4 + (u_offset_j + j), col_offset + r] += val
            
            for j in range(2):
                for k in range(2):
                    val = V_s[j*2+k, r]
                    if val != 0:
                        V_g[(v_offset_j + j) * 6 + (v_offset_k + k), col_offset + r] += val
            
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i, r]
                    if val != 0:
                        W_g[(w_offset_k + k) * 2 + (w_offset_i + i), col_offset + r] += val
    
    # Block 1: A1 * B11
    # A1: i in 0..1, j in 0..1
    # B11: j in 0..1, k in 0..1
    # C11: k in 0..1, i in 0..1
    add_strassen_block(U, V, W, 0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A2 * B21
    # A2: i in 0..1, j in 2..3
    # B21: j in 2..3, k in 0..1
    # C11: k in 0..1, i in 0..1
    add_strassen_block(U, V, W, 0, 2, 2, 0, 0, 0, 7)
    
    # Block 3: A1 * B12
    # A1: i in 0..1, j in 0..1
    # B12: j in 0..1, k in 2..3
    # C12: k in 2..3, i in 0..1
    add_strassen_block(U, V, W, 0, 0, 0, 2, 2, 0, 14)
    
    # Block 4: A2 * B22
    # A2: i in 0..1, j in 2..3
    # B22: j in 2..3, k in 2..3
    # C12: k in 2..3, i in 0..1
    add_strassen_block(U, V, W, 0, 2, 2, 2, 2, 0, 21)
    
    # Block 5: A1 * B13
    # A1: i in 0..1, j in 0..1
    # B13: j in 0..1, k in 4..5
    # C13: k in 4..5, i in 0..1
    add_strassen_block(U, V, W, 0, 0, 0, 4, 4, 0, 28)
    
    # Block 6: A2 * B23
    # A2: i in 0..1, j in 2..3
    # B23: j in 2..3, k in 4..5
    # C13: k in 4..5, i in 0..1
    add_strassen_block(U, V, W, 0, 2, 2, 4, 4, 0, 35)
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i,j,k.
    
    # Quick check on a few entries
    # C[0,0] = sum_j A[0,j] B[j,0]
    # T[0,j,0] should be 1 for all j.
    # sum_r U[0*4+j, r] V[j*6+0, r] W[0*2+0, r]
    
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(Rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    error += 1
    
    if error > 0:
        # Fallback to trivial if construction failed
        Rank = n*m*p
        U = np.zeros((n*m, Rank))
        V = np.zeros((m*p, Rank))
        W = np.zeros((p*n, Rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
