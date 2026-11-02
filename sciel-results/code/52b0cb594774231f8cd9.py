"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank  
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]
    
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s kill
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p  # 80
    
    # Try Strassen-based block decomposition
    # Split A (4x4) into 2x2 blocks of 2x2 matrices
    # Split B (4x5) into 2x2 blocks: top-left 2x2, top-right 2x3, bottom-left 2x2, bottom-right 2x3
    # This is asymmetric. Let's use a different approach.
    
    # Alternative: compute 4x4 * 4x5 as:
    # First compute 4x4 * 4x4 (first 4 cols of B) using Strassen
    # Then compute 4x4 * 4x1 (last col of B) using schoolbook
    # 4x4*4x4 via Strassen: rank 2^3 * 7 = 56 (using Strassen recursively)
    # Wait, 4x4*4x4: split into 2x2 blocks, each 2x2*2x2 uses Strassen (rank 7)
    # So 4x4*4x4 needs 4*7 = 28 multiplications? No...
    # Actually for 2x2*2x2, Strassen uses 7 scalar mults.
    # For 4x4*4x4, we have 4 submatrices of size 2x2.
    # Using Strassen at the 4x4 level: 7 "multiplications" of 2x2 matrices.
    # Each 2x2*2x2 multiplication itself can use Strassen (7 scalar mults).
    # So total: 7 * 7 = 49 scalar mults for 4x4*4x4.
    # Plus 4x4*4x1: this is 4 dot products of length 4, each needs 4 mults = 16 mults.
    # But we can be smarter. Let me think...
    
    # Actually, let me try a direct Strassen embedding.
    # 4x4 * 4x5: 
    # Let A be 4x4, B be 4x5.
    # Split A into 4 blocks: A11, A12, A21, A22 each 2x2
    # Split B into blocks: B11 (2x2), B12 (2x3), B21 (2x2), B22 (2x3)
    # C = A*B, C is 4x5
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x3)
    
    # For the 2x2*2x2 products, use Strassen (7 mults each)
    # For the 2x2*2x3 products, we need to figure out the rank.
    # 2x2 * 2x3: schoolbook is 12. Can we do better?
    # Actually, 2x2*2x3: C = A*B where A is 2x2, B is 2x3.
    # C is 2x3. Each entry C[i,j] = sum_k A[i,k]*B[k,j]
    # This is like computing 2 rows of a 2x2*2x2 product (ignoring the 3rd col).
    # The tensor for 2x2*2x3 has rank... let me think.
    # It's a sub-tensor of 2x2*2x2 (rank 7). The 2x2*2x3 tensor is obtained by
    # zeroing out the 3rd column of the output. This doesn't necessarily reduce rank.
    # Actually, 2x2*2x3 can be done in rank 7 as well (just use Strassen and discard one output).
    # But wait, Strassen computes all 4 outputs. If we only need 3 outputs (2x3), 
    # we still need the intermediate computations.
    
    # Let me just try to construct a Strassen-based solution and verify it.
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen formulas for 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11 (k=0,i=0 -> k*2+i=0)
        W[3][0] = 1  # C22 (k=1,i=1 -> k*2+i=3)
        
        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21 (k=1,i=0 -> k*2+i=2)
        
        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1  # A11
        V[1][2] = 1  # B12
        V[3][2] = -1  # B22
        W[1][2] = 1  # C12 (k=0,i=1 -> k*2+i=1)
        
        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1  # A22
        V[2][3] = 1  # B21
        V[0][3] = -1  # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1  # C11
        W[1][4] = 1  # C12
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1  # A21
        U[0][5] = -1  # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1  # A12
        U[3][6] = -1  # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        W[3][6] = 1  # C22
        
        return U, V, W
    
    def test_strassen():
        """Test Strassen 2x2 decomposition."""
        U, V, W = strassen_2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not test_strassen():
        # Strassen implementation is wrong, skip
        pass
    else:
        # Build 4x4*4x5 using Strassen for 2x2*2x2 parts
        # and figure out 2x2*2x3 parts
        
        # For 2x2*2x3, let's use schoolbook for now (rank 12)
        # But we can try to optimize.
        
        # Actually, let me think about this differently.
        # 4x4 * 4x5:
        # We can view this as computing C = A*B where A is 4x4, B is 4x5.
        # Split into 2x2 blocks:
        # A = [A11 A12; A21 A22] (each 2x2)
        # B = [B11 B12; B21 B22] where B11,B21 are 2x2 and B12,B22 are 2x3
        # C = [C11 C12; C21 C22] where C11,C21 are 2x2 and C12,C22 are 2x3
        
        # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x2*2x2)
        # C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x2*2x3)
        # C21 = A21*B11 + A22*B21  (2x2 = 2x2*2x2 + 2x2*2x2)
        # C22 = A21*B12 + A22*B22  (2x3 = 2x2*2x3 + 2x2*2x3)
        
        # For each 2x2*2x2, use Strassen (7 mults)
        # For each 2x2*2x3, we need a decomposition.
        
        # 2x2*2x3: A(2x2) * B(2x3) = C(2x3)
        # C[i,j] = sum_k A[i,k] * B[k,j] for i=0,1 and j=0,1,2
        # This is 6 scalar products of length 2.
        # Schoolbook: 12 mults.
        # Can we do better? The tensor is a "slice" of 2x2*2x2.
        # I believe the rank of 2x2*2x3 is 7 (same as 2x2*2x2).
        # Let me try to construct it.
        
        # Actually, I'll just use schoolbook for 2x2*2x3 for now and see the total.
        # 4 * 7 (for 2x2*2x2) + 4 * 12 (for 2x2*2x3) = 28 + 48 = 76
        # That's better than 80!
        
        # But wait, we need to be careful about how we combine them.
        # The Strassen decomposition gives us the product of two 2x2 matrices.
        # We need to add the results from different block multiplications.
        
        # Let me construct the full decomposition.
        
        def build_strassen_block():
            """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
            U_list = []
            V_list = []
            W_list = []
            
            # Block (1,1): C11 += A11*B11
            # A11 is rows 0-1, cols 0-1 of A -> indices 0,1,2,3 in U (i*m+j)
            # B11 is rows 0-1, cols 0-1 of B -> indices 0,1,2,3 in V (j*p+k)
            # C11 is rows 0-1, cols 0-1 of C -> W indices: k*n+i
            # C11[i,k] = W[k*4+i]
            
            # Use Strassen for A11*B11
            sU, sV, sW = strassen_2x2()
            # Map Strassen indices to full matrix indices
            # A11[i,j] -> U[(i*2+j)*4 + j] ... wait, U index is i_full*m + j_full
            # A11 is at rows 0-1, cols 0-1, so U[i*4+j] for i=0,1 j=0,1
            # B11 is at rows 0-1, cols 0-1, so V[j*5+k] for j=0,1 k=0,1
            # C11[i,k] -> W[k*4+i] for i=0,1 k=0,1
            
            # Strassen U indices: 0=A11, 1=A12, 2=A21, 3=A22
            # Map to full: 0->U[0], 1->U[1], 2->U[4], 3->U[5]
            # Wait, A is 4x4, so A[i,j] is at U[i*4+j]
            # A11 = A[0:2, 0:2], so A11[0,0]=U[0], A11[0,1]=U[1], A11[1,0]=U[4], A11[1,1]=U[5]
            # Strassen U: [A11, A12, A21, A22] -> [U[0], U[1], U[4], U[5]]
            
            # B11 = B[0:2, 0:2], B[j,k] is at V[j*5+k]
            # B11[0,0]=V[0], B11[0,1]=V[1], B11[1,0]=V[5], B11[1,1]=V[6]
            # Strassen V: [B11, B12, B21, B22] -> [V[0], V[1], V[5], V[6]]
            
            # C11[i,k] -> W[k*4+i]
            # C11[0,0]=W[0], C11[0,1]=W[4], C11[1,0]=W[1], C11[1,1]=W[5]
            # Strassen W: [C11, C12, C21, C22] -> [W[0], W[4], W[1], W[5]]
            
            a_map = [0, 1, 4, 5]  # Strassen A index -> full U index
            b_map = [0, 1, 5, 6]  # Strassen B index -> full V index
            c_map = [0, 4, 1, 5]  # Strassen C index -> full W index
            
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
            
            # Block (1,2): C12 += A11*B12
            # A11 same as before
            # B12 = B[0:2, 2:5], B[j,k] for j=0,1 k=2,3,4 -> V[j*5+k]
            # B12[0,0]=V[2], B12[0,1]=V[3], B12[0,2]=V[4]
            # B12[1,0]=V[7], B12[1,1]=V[8], B12[1,2]=V[9]
            # C12[i,k] for i=0,1 k=0,1,2 -> W[k*4+i]
            # C12[0,0]=W[0], C12[0,1]=W[4], C12[0,2]=W[8]
            # C12[1,0]=W[1], C12[1,1]=W[5], C12[1,2]=W[9]
            
            # Schoolbook for 2x2*2x3: 12 terms
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+(2+k)] = 1  # B12 column k is at col 2+k
                        W_new[(2+k)*4+i] = 1  # C12 col k is at col 2+k
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
            
            # Block (1,3): C11 += A12*B21
            # A12 = A[0:2, 2:4], A[i,j] for i=0,1 j=2,3 -> U[i*4+j]
            # A12[0,0]=U[2], A12[0,1]=U[3], A12[1,0]=U[6], A12[1,1]=U[7]
            # B21 = B[2:4, 0:2], B[j,k] for j=2,3 k=0,1 -> V[j*5+k]
            # B21[0,0]=V[10], B21[0,1]=V[11], B21[1,0]=V[15], B21[1,1]=V[16]
            
            a_map2 = [2, 3, 6, 7]
            b_map2 = [10, 11, 15, 16]
            c_map2 = [0, 4, 1, 5]  # Same C11 mapping
            
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map2[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map2[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map2[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
            
            # Block (1,4): C12 += A12*B22
            # A12 same as above
            # B22 = B[2:4, 2:5], B[j,k] for j=2,3 k=2,3,4 -> V[j*5+k]
            # B22[0,0]=V[12], B22[0,1]=V[13], B22[0,2]=V[14]
            # B22[1,0]=V[17], B22[1,1]=V[18], B22[1,2]=V[19]
            
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+(2+j)] = 1  # A12 col j is at col 2+j
                        V_new[(2+j)*5+(2+k)] = 1  # B22 row j, col k
                        W_new[(2+k)*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
            
            # Block (2,1): C21 += A21*B11
            # A21 = A[2:4, 0:2], A[i,j] for i=2,3 j=0,1 -> U[i*4+j]
            # A21[0,0]=U[8], A21[0,1]=U[9], A21[1,0]=U[12], A21[1,1]=U[13]
            # B11 same as before
            
            a_map3 = [8, 9, 12, 13]
            b_map3 = [0, 1, 5, 6]
            c_map3 = [1, 5, 2, 6]  # C21: W[k*4+i] for i=2,3 k=0,1
            # C21[0,0]=W[0*4+2]=W[2], C21[0,1]=W[1*4+2]=W[6]
            # C21[1,0]=W[0*4+3]=W[3], C21[1,1]=W[1*4+3]=W[7]
            # Strassen W: [C11,C12,C21,C22] -> need to map to C21 positions
            # C21 in Strassen is at position 2, which maps to W[2] and W[6] in full
            # But Strassen W[2] corresponds to C21[0,0] and C21[0,1]? No.
            # Strassen W indices: 0=C11, 1=C12, 2=C21, 3=C22
            # In the 2x2 output, C21 is the bottom-left 2x2 block.
            # Wait, I'm confusing myself. In the 2x2*2x2 Strassen, the output is a 2x2 matrix.
            # W[k*2+i] where k is col, i is row.
            # For the full 4x4 output, C21 is at rows 2-3, cols 0-1.
            # So C21[i,j] (local) -> full W[j*4+(2+i)]
            # Local C21[0,0] -> W[0*4+2] = W[2]
            # Local C21[0,1] -> W[1*4+2] = W[6]  
            # Local C21[1,0] -> W[0*4+3] = W[3]
            # Local C21[1,1] -> W[1*4+3] = W[7]
            # Strassen W: index 0=C11[0,0], 1=C11[1,0]? No...
            # Strassen W[k*2+i]: 
            # W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
            # So Strassen W[0] -> local C[0,0], W[1] -> local C[1,0], W[2] -> local C[0,1], W[3] -> local C[1,1]
            # For C21 block: local C[i,j] -> full W[j*4+(2+i)]
            # Strassen W[0] (local C[0,0]) -> full W[0*4+2] = W[2]
            # Strassen W[1] (local C[1,0]) -> full W[0*4+3] = W[3]
            # Strassen W[2] (local C[0,1]) -> full W[1*4+2] = W[6]
            # Strassen W[3] (local C[1,1]) -> full W[1*4+3] = W[7]
            
            c_map3 = [2, 3, 6, 7]
            
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map3[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map3[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map3[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
            
            # Block (2,2): C22 += A21*B12
            # A21 same as above
            # B12 same as before
            
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
            
            # Block (3,1): C21 += A22*B21
            # A22 = A[2:4, 2:4], A[i,j] for i=2,3 j=2,3 -> U[i*4+j]
            # A22[0,0]=U[10], A22[0,1]=U[11], A22[1,0]=U[14], A22[1,1]=U[15]
            # B21 same as before
            
            a_map4 = [10, 11, 14, 15]
            b_map4 = [10, 11, 15, 16]
            c_map4 = [2, 3, 6, 7]
            
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map4[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map4[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map4[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
            
            # Block (3,2): C22 += A22*B22
            # A22 same as above
            # B22 same as before
            
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+(2+j)] = 1
                        V_new[(2+j)*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
            
            U = U_list
            V = V_list
            W = W_list
            return {"U": U, "V": V, "W": W}
        
        strassen_result = build_strassen_block()
        rank = len(strassen_result["U"])
        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
            if rank < best_rank:
                best = strassen_result
                best_rank = rank
    
    # Try other approaches if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try to see if we can do even better
        # For now, return the best we have
        pass
    
    return best
    
# EVOLVE-BLOCK-END
