"""Matrix multiplication 4x4 by 4x6: search for low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    def make_schoolbook():
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
    
    # Try to find a decomposition with rank < 96 using structured approach
    # Use Strassen-style: split into 2x2 blocks
    
    # For 2x2 * 2x3, try to find rank-8 decomposition
    # T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # T = sum_r u_r[i] * v_r[j] * w_r[k]
    
    # Set up linear system for rank R:
    # For each (i,j,k): T[i,j,k] = sum_r U[r,i] * V[r,j] * W[r,k]
    # This is nonlinear. Use iterative approach or known constructions.
    
    # Known: 2x2*2x2 has rank 7 (Strassen). 
    # For 2x2*2x3, let's try to extend Strassen.
    
    # Actually, let me try a different structural approach:
    # Use the fact that 4x4*4x6 can be seen as computing 4x6 output.
    # Each output C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try rank 84 (7*12) with Strassen blocks where each 2x2*2x3 uses rank 12.
    # This gives 84 < 96, an improvement.
    
    def strassen_decomp():
        """Strassen-based decomposition: 7 products of (2x2)*(2x3), each rank 12."""
        sub_rank = 12
        total_rank = 7 * sub_rank  # 84
        
        # Sub-decomposition for 2x2 * 2x3 (schoolbook, rank 12)
        U_sub = [[0]*sub_rank for _ in range(4)]  # 2*2 rows
        V_sub = [[0]*sub_rank for _ in range(6)]  # 2*3 rows
        W_sub = [[0]*sub_rank for _ in range(6)]  # 3*2 rows
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_sub[i*2+j][c] = 1
                    V_sub[j*3+k][c] = 1
                    W_sub[k*2+i][c] = 1
                    c += 1
        
        # Strassen combinations:
        # A blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1) - each 2x2
        # B blocks: B11(0,0:3), B12(0,3:6)... wait, B is 4x6, split into 2x3 blocks
        # B row-blocks: 0 (rows 0-1), 1 (rows 2-3)
        # B col-blocks: 0 (cols 0-2), 1 (cols 3-5) -- but that's 2x2 blocks of 2x3? No.
        
        # Let me re-think. B is 4x6. Split into 2 row-blocks of 2 rows, and 3 col-blocks of 2 cols.
        # So B has 2*3 = 6 blocks, each 2x2? No, 2 rows x 2 cols = 2x2 blocks.
        # But then C is 4x6, split into 2 row-blocks x 3 col-blocks, each 2x2.
        
        # Hmm, but the problem says W[k*n+i,r] which is the trace convention for p*n = 6*4 = 24 entries.
        # So C is 4x6 (4 rows, 6 cols).
        
        # Let me split:
        # A (4x4) into 2x2 blocks of size 2x2: A[ib,jb] for ib,jb in {0,1}
        # B (4x6) into 2x3 blocks of size 2x2: B[ib,kb] for ib in {0,1}, kb in {0,1,2}
        # C (4x6) into 2x3 blocks of size 2x2: C[ib,kb] for ib in {0,1}, kb in {0,1,2}
        
        # C[ib,kb] = sum_jb A[ib,jb] * B[jb,kb]
        # This is a 2x2 matrix = sum of 2 products of 2x2 matrices.
        
        # Total: 2 (ib) * 3 (kb) * 2 (jb) = 12 products of 2x2*2x2.
        # Each 2x2*2x2 can be done in rank 7 (Strassen).
        # Total rank = 12 * 7 = 84.
        
        # This is the same as before! Let me implement this.
        
        # Strassen for 2x2*2x2:
        # M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22), M4=A22(B21-B11)
        # M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
        # C11=M1-M4-M6+M7, C12=M3+M1, C21=M2+M4, C22=M1-M3+M5
        
        def strassen_2x2_2x2():
            """Return (U,V,W) for 2x2*2x2 with rank 7."""
            R = 7
            U = [[0]*R for _ in range(4)]  # 2*2
            V = [[0]*R for _ in range(4)]  # 2*2
            W = [[0]*R for _ in range(4)]  # 2*2, W[k*2+i]
            
            # M1: (A11+A22)(B11+B22)
            # A part: A11 at (0,0), A22 at (1,1)
            # B part: B11 at (0,0), B22 at (1,1)
            # C part: C11 at (0,0), C12 at (0,1), C21 at (1,0), C22 at (1,1)
            # W index: k*2+i, so C11->W[0], C12->W[1], C21->W[2], C22->W[3]
            
            # M1 contributes to all C entries with +1
            U[0][0] = 1  # A11
            U[3][0] = 1  # A22
            V[0][0] = 1  # B11
            V[3][0] = 1  # B22
            W[0][0] = 1  # C11
            W[1][0] = 1  # C12
            W[2][0] = 1  # C21
            W[3][0] = 1  # C22
            
            # M2: (A21+A22)B11
            U[2][1] = 1  # A21
            U[3][1] = 1  # A22
            V[0][1] = 1  # B11
            W[2][1] = 1  # C21
            W[3][1] = 1  # C22
            
            # M3: A11(B12-B22)
            U[0][2] = 1  # A11
            V[1][2] = 1  # B12
            V[3][2] = -1  # B22
            W[1][2] = 1  # C12
            W[3][2] = 1  # C22
            
            # M4: A22(B21-B11)
            U[3][3] = 1  # A22
            V[2][3] = 1  # B21
            V[0][3] = -1  # B11
            W[2][3] = 1  # C21
            W[3][3] = 1  # C22
            
            # M5: (A11+A12)B22
            U[0][4] = 1  # A11
            U[1][4] = 1  # A12
            V[3][4] = 1  # B22
            W[1][4] = 1  # C12
            W[3][4] = 1  # C22
            
            # M6: (A21-A11)(B11+B12)
            U[2][5] = 1  # A21
            U[0][5] = -1  # A11
            V[0][5] = 1  # B11
            V[1][5] = 1  # B12
            W[2][5] = 1  # C21
            W[3][5] = 1  # C22
            
            # M7: (A12-A22)(B21+B22)
            U[1][6] = 1  # A12
            U[3][6] = -1  # A22
            V[2][6] = 1  # B21
            V[3][6] = 1  # B22
            W[0][6] = 1  # C11
            W[1][6] = 1  # C12
            W[2][6] = 1  # C21
            W[3][6] = 1  # C22
            
            return U, V, W
        
        U_s, V_s, W_s = strassen_2x2_2x2()
        
        # Now build the full 4x4*4x6 decomposition
        # A: 2x2 blocks of 2x2, indexed by (ib, jb) for ib,jb in {0,1}
        # B: 2x3 blocks of 2x2, indexed by (jb, kb) for jb in {0,1}, kb in {0,1,2}
        # C: 2x3 blocks of 2x2, indexed by (ib, kb) for ib in {0,1}, kb in {0,1,2}
        
        # For each (ib, kb), C[ib,kb] = A[ib,0]*B[0,kb] + A[ib,1]*B[1,kb]
        # Each product is 2x2*2x2, done with Strassen (rank 7).
        # Total: 2*3*2 = 12 Strassen products, each rank 7, total rank 84.
        
        total_rank = 12 * 7
        U_full = [[0]*total_rank for _ in range(16)]  # 4*4
        V_full = [[0]*total_rank for _ in range(24)]  # 4*6
        W_full = [[0]*total_rank for _ in range(24)]  # 6*4
        
        col_offset = 0
        for ib in range(2):
            for kb in range(3):
                for jb in range(2):
                    # Product: A[ib,jb] * B[jb,kb] contributes to C[ib,kb]
                    # A[ib,jb] is rows 2*ib to 2*ib+1, cols 2*jb to 2*jb+1
                    # B[jb,kb] is rows 2*jb to 2*jb+1, cols 2*kb to 2*kb+1
                    # C[ib,kb] is rows 2*ib to 2*ib+1, cols 2*kb to 2*kb+1
                    # W index: k*n+i = k*4+i where k is col of C, i is row of C
                    
                    for r in range(7):
                        # U: A indices
                        for ai in range(2):
                            for aj in range(2):
                                a_idx = ai*2 + aj  # local index in 2x2
                                if U_s[a_idx][r] != 0:
                                    global_i = 2*ib + ai
                                    global_j = 2*jb + aj
                                    U_full[global_i*4 + global_j][col_offset + r] += U_s[a_idx][r]
                        
                        # V: B indices
                        for bj in range(2):
                            for bk in range(2):
                                b_idx = bj*2 + bk  # local index in 2x2
                                if V_s[b_idx][r] != 0:
                                    global_j = 2*jb + bj
                                    global_k = 2*kb + bk
                                    V_full[global_j*6 + global_k][col_offset + r] += V_s[b_idx][r]
                        
                        # W: C indices
                        for ck in range(2):
                            for ci in range(2):
                                c_idx = ck*2 + ci  # local index, W[ck*2+ci]
                                if W_s[c_idx][r] != 0:
                                    global_k = 2*kb + ck
                                    global_i = 2*ib + ci
                                    W_full[global_k*4 + global_i][col_offset + r] += W_s[c_idx][r]
                    
                    col_offset += 7
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify the Strassen decomposition is correct
    def verify(U, V, W):
        """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,n+i,r] = A[i,j]*B[j,k] for all i,j,k."""
        n, m, p = 4, 4, 6
        rank = len(U[0])
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    # Should be 1 if this is the "diagonal" term, 0 otherwise
                    # Actually, the tensor T[i,j,k] = delta_{i,j,k} is not right.
                    # The matrix multiplication tensor is T[i,j,k] = A[i,j]*B[j,k]
                    # which as a trilinear form is: for fixed A,B, C[i,k] = sum_j A[i,j]B[j,k]
                    # The tensor has T[i,j,k] = 1 for all i,j,k (it's the product tensor)
                    # Wait no. The tensor decomposition means:
                    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[j,r]) * (sum_k B[j,k] V[k,r]) * ... 
                    # No, the correct form is:
                    # C[i,k] = sum_r (sum_j A[i,j] U[j,r]) * V[r,k] ... 
                    
                    # Let me re-read the problem. The tensor is for matrix multiplication:
                    # T[i,j,k] such that C = A*B means C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
                    # where T[i,j,k] = delta_{j,j} = 1 for all i,j,k.
                    # So T is the all-ones tensor of size n*m*p.
                    
                    # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # Wait, the interface says U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
                    # So U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                    # And T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
                    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                    
                    expected = 1
                    if abs(s - expected) > 1e-9:
                        return False, (i,j,k,s,expected)
        return True, None
    
    # First try Strassen (rank 84)
    result = strassen_decomp()
    valid, err = verify(result["U"], result["V"], result["W"])
    
    if valid:
        return result
    
    # If Strassen fails, fall back to schoolbook
    return make_schoolbook()
