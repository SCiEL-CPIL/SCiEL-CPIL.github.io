"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    def verify_decomp(U, V, W):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build schoolbook decomposition as baseline
    U_school = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0.0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Strategy: Use block structure with Strassen for 4x4x4 sub-problem
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [[B1 (4x6)], [B2 (1x6)]]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x4 * 4x6 = 4x4*4x4 + 4x4*4x2
    #   - 4x4*4x4: use Strassen (rank 7 for 2x2x2, so 4 blocks * 7 = 28? No...)
    #     Actually for 4x4*4x4, we can use 2x2 Strassen blocks:
    #     Each 2x2*2x2 has rank 7, and we have 4 such blocks, but they interact.
    #     Better: use the known rank-21 decomposition for 4x4x4.
    #   - 4x4*4x2: schoolbook rank 32
    # A2*B2: 4x1 * 1x6 = outer product, rank 24 schoolbook
    #
    # Let me construct a proper decomposition using Strassen for the 4x4x4 part.
    
    # Strassen's algorithm for 2x2x2 (rank 7):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4*4x4, partition into 2x2 blocks of 2x2 matrices:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = [[C11, C12], [C21, C22]]
    # 
    # Using Strassen: each Mi is a 2x2*2x2 multiplication (rank 7 each)
    # But we need to compute each Mi, which itself requires rank 7.
    # Total rank = 7 * 7 = 49? That's worse than schoolbook 64? No wait.
    # 
    # Actually, the rank of 4x4x4 tensor is the minimum number of rank-1 tensors.
    # Strassen gives rank 7 for 2x2x2. For 4x4x4, we can nest:
    # The 4x4x4 tensor can be decomposed using 4 copies of 2x2x2 (one for each block position)
    # But that gives rank 4*7 = 28, which is better than 64!
    # 
    # Wait, that's not quite right either. Let me think more carefully.
    # 
    # The matrix multiplication tensor for 4x4*4x4 has 64 entries (i,j,k) where i,j,k in {0,1,2,3}.
    # We can partition into 8 sub-tensors of size 2x2x2 (by splitting each index into 2 bits).
    # Each 2x2x2 sub-tensor has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56. Still better than 64!
    # 
    # But actually, we can do better. The known optimal rank for 4x4x4 is 21.
    # Let me use a known rank-21 decomposition.
    
    # Known rank-21 decomposition for 4x4x4 (from literature):
    # This is complex. Let me use a simpler approach: 
    # Use Strassen recursively to get rank 56 for 4x4x4, then try to eliminate.
    
    # Actually, let me just use schoolbook for now and focus on the block structure.
    # The key insight is:
    # C = A*B where A is 4x5, B is 5x6
    # 
    # Partition A as [A1 | A2] where A1 is 4x4, A2 is 4x1
    # Partition B as [[B1], [B2]] where B1 is 4x6, B2 is 1x6
    # 
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x4 * 4x6. Partition B1 = [B1a | B1b] where B1a is 4x4, B1b is 4x2
    # A1*B1 = A1*B1a + A1*B1b
    #   - A1*B1a: 4x4*4x4, rank 64 schoolbook
    #   - A1*B1b: 4x4*4x2, rank 32 schoolbook
    # A2*B2: 4x1*1x6, rank 24 schoolbook (outer product)
    # 
    # Total schoolbook: 64 + 32 + 24 = 120 (same as before, no savings)
    # 
    # To get savings, we need to use a lower-rank decomposition for 4x4*4x4.
    # Let me implement Strassen for 4x4*4x4 with rank 56.
    
    def build_strassen_4x4x4():
        """Build Strassen decomposition for 4x4*4x4 with rank 56."""
        # 4x4*4x4: partition each index into 2 bits: i = i0*2 + i1, etc.
        # 8 sub-tensors of 2x2*2x2, each with rank 7 (Strassen)
        # Total rank = 8 * 7 = 56
        
        U = [[0.0] * 56 for _ in range(16)]  # 4*4 = 16 entries
        V = [[0.0] * 56 for _ in range(16)]
        W = [[0.0] * 56 for _ in range(16)]
        
        # Strassen for 2x2*2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For each of the 8 sub-tensors (i0,j0,k0 in {0,1}^3):
        for i0 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    base = (i0 * 4 + j0) * 8 + k0 * 7  # Not quite right, let me fix indexing
                    
                    # Actually, let me index differently.
                    # U index: (i, j) where i in 0-3, j in 0-3 -> index = i*4 + j
                    # V index: (j, k) where j in 0-3, k in 0-3 -> index = j*4 + k  
                    # W index: (k, i) where k in 0-3, i in 0-3 -> index = k*4 + i
                    # 
                    # For sub-tensor (i0, j0, k0):
                    # i = i0*2 + i1, j = j0*2 + j1, k = k0*2 + k1
                    # U[i,j] = U[i0*2+i1, j0*2+j1]
                    # V[j,k] = V[j0*2+j1, k0*2+k1]
                    # W[k,i] = W[k0*2+k1, i0*2+i1]
                    
                    sub_base = (i0 * 2 + j0) * 8 + k0 * 7  # This is getting complicated
                    
                    # Let me just do it explicitly for each of the 8 sub-tensors.
                    # Sub-tensor (i0, j0, k0) uses columns 7*(i0*4 + j0*2 + k0) to 7*(i0*4 + j0*2 + k0)+6
                    
                    col_base = 7 * (i0 * 4 + j0 * 2 + k0)
                    
                    # M1 = (A[i0*2+i1, j0*2+j1] for i1,j1 in {0,1} with pattern) 
                    # Actually, Strassen combines A entries. For the sub-tensor, the A block is 2x2:
                    # A_sub[i1, j1] = A[i0*2+i1, j0*2+j1]
                    # Similarly for B and C.
                    
                    # M1: (A_sub + A_sub_shifted) * (B_sub + B_sub_shifted)
                    # This is getting very complex. Let me use a different approach.
                    
                    # For simplicity, let me just use schoolbook for now and try elimination.
                    pass
        
        return U, V, W
    
    # Given the complexity, let me use a simpler strategy:
    # 1. Start with schoolbook (rank 120)
    # 2. Try greedy elimination
    # 3. Try a few structured decompositions
    
    def try_eliminate(U, V, W, max_rounds=10):
        """Try to eliminate redundant terms"""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        eliminated = 0
        
        for _round in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            found = False
            for r in range(rank):
                if time.time() > deadline - 5:
                    break
                
                # Check if column r is needed
                U_save = [row[:] for row in U]
                V_save = [row[:] for row in V]
                W_save = [row[:] for row in W]
                
                for row in U:
                    row[r] = 0.0
                for row in V:
                    row[r] = 0.0
                for row in W:
                    row[r] = 0.0
                
                if verify_decomp(U, V, W):
                    U = [row[:r] + row[r+1:] for row in U]
                    V = [row[:r] + row[r+1:] for row in V]
                    W = [row[:r] + row[r+1:] for row in W]
                    rank -= 1
                    eliminated += 1
                    found = True
                    break
                else:
                    U, V, W = U_save, V_save, W_save
            
            if not found:
                break
        
        return U, V, W, eliminated
    
    # Try elimination on schoolbook
    U_elim, V_elim, W_elim, elim_count = try_eliminate(U_school, V_school, W_school)
    
    if elim_count > 0:
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = len(U_elim[0])
    
    # Try a structured decomposition: partition A and B to enable Strassen-like savings
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [[B1 (4x6)], [B2 (1x6)]]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (4x4 * 4x6):
    # Partition B1 = [B1a (4x4) | B1b (4x2)]
    # A1*B1 = A1*B1a + A1*B1b
    
    # Use Strassen for A1*B1a (4x4*4x4):
    # The 4x4*4x4 tensor can be decomposed using 8 sub-tensors of 2x2*2x2 (rank 7 each)
    # Total rank for this part: 8 * 7 = 56
    
    # Let me construct this properly.
    
    def build_4x4x4_strassen():
        """Build a rank-56 decomposition for 4x4*4x4 using Strassen."""
        U = [[0.0] * 56 for _ in range(16)]
        V = [[0.0] * 56 for _ in range(16)]
        W = [[0.0] * 56 for _ in range(16)]
        
        # Strassen 2x2x2 coefficients:
        # M1 = (A00+A11)*(B00+B11): U coeff for A00=1, A11=1; V coeff for B00=1, B11=1; W coeff for C00=1, C11=1
        # M2 = (A10+A11)*B00: U: A10=1, A11=1; V: B00=1; W: C10=1
        # M3 = A00*(B01-B11): U: A00=1; V: B01=1, B11=-1; W: C01=1
        # M4 = A11*(B10-B00): U: A11=1; V: B10=1, B00=-1; W: C00=1, C10=1
        # M5 = (A00+A01)*B11: U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
        # M6 = (A10-A00)*(B00+B01): U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
        # M7 = (A01-A11)*(B10+B11): U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1, C11=1
        
        strassen_coeffs = [
            # (U_coeffs for 2x2 A block, V_coeffs for 2x2 B block, W_coeffs for 2x2 C block)
            # Each is a dict mapping (local_i, local_j) to coefficient
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # M2
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),  # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M7
        ]
        
        for i0 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    col_base = 7 * (i0 * 4 + j0 * 2 + k0)
                    
                    for s in range(7):
                        u_coeffs, v_coeffs, w_coeffs = strassen_coeffs[s]
                        col = col_base + s
                        
                        for (li, lj), coeff in u_coeffs.items():
                            i = i0 * 2 + li
                            j = j0 * 2 + lj
                            U[i * 4 + j][col] += coeff
                        
                        for (lj, lk), coeff in v_coeffs.items():
                            j = j0 * 2 + lj
                            k = k0 * 2 + lk
                            V[j * 4 + k][col] += coeff
                        
                        for (lk, li), coeff in w_coeffs.items():
                            k = k0 * 2 + lk
                            i = i0 * 2 + li
                            W[k * 4 + i][col] += coeff
        
        return U, V, W
    
    # Build the full 4x5*5x6 decomposition using Strassen for 4x4*4x4 part
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [[B1 (4x6)], [B2 (1x6)]]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x4 * 4x6 = A1*B1a + A1*B1b where B1a is 4x4, B1b is 4x2
    #   - A1*B1a: use Strassen rank 56
    #   - A1*B1b: schoolbook rank 32
    # A2*B2: 4x1 * 1x6, schoolbook rank 24
    # 
    # Total: 56 + 32 + 24 = 112
    
    if time.time() < deadline - 10:
        U_strassen_444, V_strassen_444, W_strassen_444 = build_4x4x4_strassen()
        
        # Now build the full decomposition
        # Part 1: A1*B1a (4x4*4x4) using Strassen, rank 56
        # Part 2: A1*B1b (4x4*4x2) using schoolbook, rank 32
        # Part 3: A2*B2 (4x1*1x6) using schoolbook, rank 24
        
        total_rank = 56 + 32 + 24
        U_full = [[0.0] * total_rank for _ in range(20)]  # n*m = 4*5 = 20
        V_full = [[0.0] * total_rank for _ in range(30)]  # m*p = 5*6 = 30
        W_full = [[0.0] * total_rank for _ in range(24)]  # p*n = 6*4 = 24
        
        # Part 1: A1*B1a (4x4*4x4), columns 0-55
        # A1 is rows 0-3, cols 0-3 of A -> U indices 0-15 (i*4+j for i in 0-3, j in 0-3)
        # B1a is rows 0-3, cols 0-3 of B -> V indices 0-15 (j*4+k for j in 0-3, k in 0-3)
        # C part is rows 0-3, cols 0-3 -> W indices 0-15 (k*4+i for k in 0-3, i in 0-3)
        
        for col in range(56):
            for i in range(4):
                for j in range(4):
                    u_val = U_strassen_444[i * 4 + j][col]
                    if u_val != 0:
                        U_full[i * 5 + j][col] = u_val  # A index: i*5+j
            
            for j in range(4):
                for k in range(4):
                    v_val = V_strassen_444[j * 4 + k][col]
                    if v_val != 0:
                        V_full[j * 6 + k][col] = v_val  # B index: j*6+k
            
            for k in range(4):
                for i in range(4):
                    w_val = W_strassen_444[k * 4 + i][col]
                    if w_val != 0:
                        W_full[k * 4 + i][col] = w_val  # C index: k*4+i
        
        # Part 2: A1*B1b (4x4*4x2), columns 56-87
        # A1 is rows 0-3, cols 0-3 of A
        # B1b is rows 0-3, cols 4-5 of B
        # C part is rows 0-3, cols 4-5
        
        col_offset = 56
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    col = col_offset + i * 8 + j * 2 + k
                    U_full[i * 5 + j][col] = 1.0
                    V_full[j * 6 + (4 + k)][col] = 1.0  # B col 4+k
                    W_full[(4 + k) * 4 + i][col] = 1.0  # C col 4+k
        
        # Part 3: A2*B2 (4x1*1x6), columns 88-111
        # A2 is rows 0-3, col 4 of A
        # B2 is row 4, cols 0-5 of B
        # C part is rows 0-3, cols 0-5
        
        col_offset = 88
        for i in range(4):
            for k in range(6):
                col = col_offset + i * 6 + k
                U_full[i * 5 + 4][col] = 1.0  # A col 4
                V_full[4 * 6 + k][col] = 1.0  # B row 4
                W_full[k * 4 + i][col] = 1.0
        
        # Verify
        if verify_decomp(U_full, V_full, W_full):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = total_rank
        
        # Try elimination on the Strassen-based decomposition
        if time.time() < deadline - 5:
            U_elim2, V_elim2, W_elim2, elim_count2 = try_eliminate(U_full, V_full, W_full, max_rounds=5)
            if elim_count2 > 0:
                best_U, best_V, best_W = U_elim2, V_elim2, W_elim2
                best_rank = len(U_elim2[0])
    
    # Return best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
