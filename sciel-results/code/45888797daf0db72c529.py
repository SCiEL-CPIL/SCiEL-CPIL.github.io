"""Optimized program: block Strassen decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = 4, 5, 6
    
    # Schoolbook fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Strassen 2x2*2x2 decomposition: rank 7
    # Standard Strassen formulas for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # m1 = (b)(e-g)
    # m2 = (a+b)(g+h)  
    # m3 = (a-d)(e+f)
    # m4 = (c+d)(e)
    # m5 = (a+b)(h)
    # m6 = (c-d)(f+h)
    # m7 = (a-c)(e-g)
    # C[0,0] = m2 + m4 - m5 - m7
    # C[0,1] = m1 + m5
    # C[1,0] = m3 + m7
    # C[1,1] = m2 + m6 - m1 - m4
    
    # Build a decomposition using block Strassen
    # Split: rows i: {0,1}, {2,3}; cols k: {0,1,2}, {3,4,5}; mid j: {0,1,2}, {3,4}
    
    # For each 2x2 output block (2 rows x 2 cols from the 4x6 output),
    # we need to compute sum over j of A_block(2x5) * B_block(5x2)
    # Split j into {0,1,2} and {3,4}:
    #   sum_{j=0}^{2} A[:,0:3] * B[0:3,:] + sum_{j=3}^{4} A[:,3:5] * B[3:5,:]
    
    # For the 2x3 * 3x2 part: we can further split
    # For the 2x2 * 2x2 part: use Strassen (rank 7)
    # For 2x1 * 1x2: rank 2
    # For 2x2 * 1x2: rank 4 (schoolbook)
    # For 2x1 * 2x2: rank 4 (schoolbook)
    
    # Let me build a complete decomposition step by step
    # I'll use a systematic approach: enumerate all terms and try to apply Strassen where possible
    
    # Alternative: use a known low-rank construction
    # The rank of 4x5*5x6 is at most:
    # Using block decomposition with Strassen on 2x2 blocks:
    
    # Let's try a direct construction with rank < 120
    # Split into 2x2 blocks of the output (4x6 = 2x3 blocks of 2x2, but 6/2=3, so 2x3=6 blocks)
    # Wait, 4x6 output: split into 2x2 blocks: 2 row-blocks x 3 col-blocks = 6 blocks of 2x2
    
    # For each 2x2 output block C_block, we need:
    # C_block[i',k'] = sum_j A_block[i',j] * B_block[j,k']
    # where A_block is 2x5, B_block is 5x2
    
    # Split j into {0,1,2} and {3,4}:
    # C_block = A_2x3 * B_3x2 + A_2x2 * B_2x2
    
    # For A_2x3 * B_3x2: split further
    #   j in {0,1,2}: A_2x3 * B_3x2
    #   Can split j={0,1} and j={2}:
    #   A_2x2 * B_2x2 (Strassen: rank 7) + A_2x1 * B_1x2 (schoolbook: rank 2)
    #   Total: 7 + 2 = 9
    
    # For A_2x2 * B_2x2: Strassen rank 7
    
    # So per 2x2 output block: 9 + 7 = 16 terms
    # 6 blocks * 16 = 96 terms? But there's overlap in the j-sums...
    
    # Actually, the j-sums are shared across blocks. Let me think differently.
    
    # Better approach: decompose the entire sum
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # = sum_{j=0}^{2} A[i,j] * B[j,k] + sum_{j=3}^{4} A[i,j] * B[j,k]
    
    # First part: 4x3 * 3x6
    # Second part: 4x2 * 2x6
    
    # For 4x3 * 3x6: split i into {0,1},{2,3}, k into {0,1,2},{3,4,5}
    # Each 2x3 * 3x3 block: can we do better than schoolbook (18)?
    # 2x3 * 3x3: split j={0,1},{2}: 2x2*2x3 + 2x1*1x3
    # 2x2*2x3: split k={0,1},{2}: 2x2*2x2 (Strassen 7) + 2x2*2x1 (schoolbook 4) = 11
    # 2x1*1x3: rank 2
    # Total for 2x3*3x3: 11 + 2 = 13
    
    # 4 blocks of 2x3*3x3: 4 * 13 = 52
    
    # For 4x2 * 2x6: split i into {0,1},{2,3}, k into {0,1,2},{3,4,5}
    # Each 2x2 * 2x3 block: split k={0,1},{2}: 2x2*2x2 (Strassen 7) + 2x2*2x1 (4) = 11
    # 4 blocks: 4 * 11 = 44
    
    # Total: 52 + 44 = 96
    
    # Let me construct this explicitly!
    
    def build_decomposition():
        """Build a rank-96 decomposition using block Strassen."""
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            """Add a rank-1 term."""
            U_list.append((u_idx, u_val))
            V_list.append((v_idx, v_val))
            W_list.append((w_idx, w_val))
        
        # We'll build the full U, V, W matrices
        # First, let's figure out the total rank
        # Part 1: 4x3 * 3x6, split into 4 blocks of 2x3*3x3
        #   Each 2x3*3x3 = 13 terms (Strassen on 2x2*2x2 + schoolbook on rest)
        # Part 2: 4x2 * 2x6, split into 4 blocks of 2x2*2x3
        #   Each 2x2*2x3 = 11 terms
        
        # Total rank = 4*13 + 4*11 = 52 + 44 = 96
        
        total_rank = 96
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        # Helper: add Strassen 2x2*2x2 terms
        # For A(2x2) * B(2x2) -> C(2x2)
        # A rows: i1, i2; B cols: k1, k2; mid: j1, j2
        # The 7 Strassen terms:
        # m1 = A[0,1] * (B[1,0] - B[0,0])
        # m2 = (A[0,0]+A[1,0]) * (B[0,0]+B[1,1])
        # m3 = (A[0,0]-A[1,1]) * (B[0,0]+B[0,1])
        # m4 = (A[1,0]-A[1,1]) * (B[1,0]+B[1,1])
        # m5 = (A[0,0]+A[0,1]) * (B[1,0]-B[1,1])
        # m6 = (A[0,1]+A[1,1]) * (B[0,0]-B[1,0])
        # m7 = (A[0,0]-A[1,0]) * (B[0,1]-B[1,1])
        # C[0,0] = m2 + m4 - m5 - m7
        # C[0,1] = m1 + m5
        # C[1,0] = m3 + m7
        # C[1,1] = m2 + m6 - m1 - m4
        
        # In tensor form, each Strassen term m_c contributes:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For m1 = A[0,1] * (B[1,0] - B[0,0]):
        # This means: u[i,j] is nonzero only at (i=0,j=1) with value 1
        # v[j,k] is nonzero at (j=1,k=0) with +1 and (j=0,k=0) with -1
        # w[k,i] = 1 for the output position
        
        # Let me define a helper that adds a Strassen block
        def add_strassen_block(i_base, k_base, j_base, offset):
            """Add Strassen terms for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]"""
            nonlocal col
            # The 7 Strassen combinations
            # Each is: (linear combo of A entries) * (linear combo of B entries)
            # In tensor form, each product term becomes a set of U,V,W entries
            
            # m1: A[0,1] * (B[1,0] - B[0,0])
            # U: i=i_base, j=j_base+1 -> 1; i=i_base+1, j=j_base+1 -> 0 (only i_base)
            # V: j=j_base+1, k=k_base -> +1; j=j_base, k=k_base -> -1
            # W: k=k_base, i=i_base -> 1; k=k_base, i=i_base+1 -> 1 (both output rows)
            # Wait, W[k*n+i, c] - for each (k,i) pair in the output block
            
            # Actually, for a 2x2*2x2 Strassen block:
            # Output is C[i_base:i_base+2, k_base:k_base+2]
            # W[k*n+i, c] should be 1 for all (k,i) in the 2x2 output block
            
            # m1 contributes to C[0,0] and C[1,0] (i.e., W[k_base, i_base] and W[k_base, i_base+1])
            # No wait, m1 = A[0,1]*(B[1,0]-B[0,0])
            # C[0,0] = m2 + m4 - m5 - m7 (no m1)
            # C[0,1] = m1 + m5
            # C[1,0] = m3 + m7
            # C[1,1] = m2 + m6 - m1 - m4
            # So m1 contributes to C[0,1] with +1 and C[1,1] with -1
            
            # Let me redo: for each Strassen multiplier m_c, which output entries does it contribute to?
            # m1: C[0,1] += m1, C[1,1] -= m1
            # m2: C[0,0] += m2, C[1,1] += m2
            # m3: C[1,0] += m3
            # m4: C[0,0] += m4, C[1,1] -= m4
            # m5: C[0,1] += m5, C[0,0] -= m5
            # m6: C[1,1] += m6
            # m7: C[1,0] += m7, C[0,0] -= m7
            
            # For each m_c, the U and V are fixed, and W has specific values for each (k,i)
            
            # m1: U[i_base, j_base+1]=1, V[j_base+1, k_base]=1, V[j_base, k_base]=-1
            #      W[k_base, i_base]=0, W[k_base+1, i_base]=1, W[k_base, i_base+1]=0, W[k_base+1, i_base+1]=-1
            # Wait, I need to be more careful.
            
            # m1 = A[i_base, j_base+1] * (B[j_base+1, k_base] - B[j_base, k_base])
            # This contributes to output C[i_base, k_base+1] and C[i_base+1, k_base+1]
            # C[i_base, k_base+1] += m1 (coefficient +1)
            # C[i_base+1, k_base+1] -= m1 (coefficient -1)
            
            # So W[k_base+1, i_base, c] = +1, W[k_base+1, i_base+1, c] = -1
            # U[i_base*m + j_base+1, c] = 1
            # V[j_base+1*p + k_base, c] = +1, V[j_base*p + k_base, c] = -1
            
            # Let me code all 7:
            
            # m1: A[0,1] * (B[1,0] - B[0,0])
            # U: (i_base, j_base+1) = 1
            # V: (j_base+1, k_base) = 1, (j_base, k_base) = -1
            # W: (k_base+1, i_base) = 1, (k_base+1, i_base+1) = -1
            U[i_base*m + j_base+1][col] = 1
            V[j_base+1*p + k_base][col] = 1
            V[j_base*p + k_base][col] = -1
            W[k_base*n + i_base][col] = 1
            W[k_base+1*n + i_base][col] = 1  # Wait, this should be -1 for i_base+1
            # Let me fix: W[k*n+i] where k=k_base+1, i=i_base -> +1; k=k_base+1, i=i_base+1 -> -1
            W[k_base+1*n + i_base][col] = 1
            W[k_base+1*n + i_base+1][col] = -1
            col += 1
            
            # m2: (A[0,0]+A[1,0]) * (B[0,0]+B[1,1])
            # U: (i_base, j_base) = 1, (i_base+1, j_base) = 1
            # V: (j_base, k_base) = 1, (j_base+1, k_base+1) = 1
            # W: (k_base, i_base) = 1, (k_base+1, i_base+1) = 1
            U[i_base*m + j_base][col] = 1
            U[(i_base+1)*m + j_base][col] = 1
            V[j_base*p + k_base][col] = 1
            V[(j_base+1)*p + k_base+1][col] = 1
            W[k_base*n + i_base][col] = 1
            W[(k_base+1)*n + i_base+1][col] = 1
            col += 1
            
            # m3: (A[0,0]-A[1,1]) * (B[0,0]+B[0,1])
            # U: (i_base, j_base) = 1, (i_base+1, j_base+1) = -1
            # V: (j_base, k_base) = 1, (j_base, k_base+1) = 1
            # W: (k_base, i_base+1) = 1, (k_base+1, i_base+1) = 1
            U[i_base*m + j_base][col] = 1
            U[(i_base+1)*m + j_base+1][col] = -1
            V[j_base*p + k_base][col] = 1
            V[j_base*p + k_base+1][col] = 1
            W[k_base*n + i_base+1][col] = 1
            W[(k_base+1)*n + i_base+1][col] = 1
            col += 1
            
            # m4: (A[1,0]-A[1,1]) * (B[1,0]+B[1,1])
            # U: (i_base+1, j_base) = 1, (i_base+1, j_base+1) = -1
            # V: (j_base+1, k_base) = 1, (j_base+1, k_base+1) = 1
            # W: (k_base, i_base) = 1, (k_base+1, i_base+1) = -1
            U[(i_base+1)*m + j_base][col] = 1
            U[(i_base+1)*m + j_base+1][col] = -1
            V[(j_base+1)*p + k_base][col] = 1
            V[(j_base+1)*p + k_base+1][col] = 1
            W[k_base*n + i_base][col] = 1
            W[(k_base+1)*n + i_base+1][col] = -1
            col += 1
            
            # m5: (A[0,0]+A[0,1]) * (B[1,0]-B[1,1])
            # U: (i_base, j_base) = 1, (i_base, j_base+1) = 1
            # V: (j_base+1, k_base) = 1, (j_base+1, k_base+1) = -1
            # W: (k_base, i_base) = -1, (k_base+1, i_base) = 1
            U[i_base*m + j_base][col] = 1
            U[i_base*m + j_base+1][col] = 1
            V[(j_base+1)*p + k_base][col] = 1
            V[(j_base+1)*p + k_base+1][col] = -1
            W[k_base*n + i_base][col] = -1
            W[(k_base+1)*n + i_base][col] = 1
            col += 1
            
            # m6: (A[0,1]+A[1,1]) * (B[0,0]-B[1,0])
            # U: (i_base, j_base+1) = 1, (i_base+1, j_base+1) = 1
            # V: (j_base, k_base) = 1, (j_base+1, k_base) = -1
            # W: (k_base+1, i_base+1) = 1
            U[i_base*m + j_base+1][col] = 1
            U[(i_base+1)*m + j_base+1][col] = 1
            V[j_base*p + k_base][col] = 1
            V[(j_base+1)*p + k_base][col] = -1
            W[(k_base+1)*n + i_base+1][col] = 1
            col += 1
            
            # m7: (A[0,0]-A[1,0]) * (B[0,1]-B[1,1])
            # U: (i_base, j_base) = 1, (i_base+1, j_base) = -1
            # V: (j_base, k_base+1) = 1, (j_base+1, k_base+1) = -1
            # W: (k_base, i_base+1) = 1, (k_base, i_base) = -1
            U[i_base*m + j_base][col] = 1
            U[(i_base+1)*m + j_base][col] = -1
            V[j_base*p + k_base+1][col] = 1
            V[(j_base+1)*p + k_base+1][col] = -1
            W[k_base*n + i_base+1][col] = 1
            W[k_base*n + i_base][col] = -1
            col += 1
            
            return col
        
        # Helper: add schoolbook terms for a small block
        def add_schoolbook_block(i_base, i_size, k_base, k_size, j_base, j_size, offset):
            """Add schoolbook terms for A[i_base:i_base+i_size, j_base:j_base+j_size] * B[j_base:j_base+j_size, k_base:k_base+k_size]"""
            nonlocal col
            for di in range(i_size):
                for dk in range(k_size):
                    for dj in range(j_size):
                        i = i_base + di
                        k = k_base + dk
                        j = j_base + dj
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return col
        
        # Now build the decomposition:
        # Part 1: j in {0,1,2} (4x3 * 3x6)
        #   Split into 4 blocks of 2x3 * 3x3:
        #   Block (0,0): i={0,1}, k={0,1,2}
        #   Block (0,1): i={0,1}, k={3,4,5}
        #   Block (1,0): i={2,3}, k={0,1,2}
        #   Block (1,1): i={2,3}, k={3,4,5}
        
        # For each 2x3 * 3x3 block:
        #   Split j into {0,1} and {2}:
        #   2x2 * 2x3 (for j={0,1}): further split k into pairs
        #   2x1 * 1x3 (for j={2}): schoolbook rank 2
        
        # For 2x2 * 2x3: split k into {0,1} and {2}:
        #   2x2 * 2x2: Strassen (7 terms)
        #   2x2 * 2x1: schoolbook (4 terms)
        
        # So per 2x3 * 3x3 block:
        #   j={0,1}: 2x2*2x3 = 7 + 4 = 11 terms
        #   j={2}: 2x1*1x3 = 2 terms
        #   Total: 13 terms
        
        # Part 2: j in {3,4} (4x2 * 2x6)
        #   Split into 4 blocks of 2x2 * 2x3:
        #   Each: 2x2 * 2x2 (Strassen 7) + 2x2 * 2x1 (schoolbook 4) = 11 terms
        #   Total: 4 * 11 = 44 terms
        
        # Grand total: 4 * 13 + 4 * 11 = 52 + 44 = 96
        
        # Let's build it:
        
        # Part 1: j in {0,1,2}
        for i_block in range(2):  # i={0,1} or {2,3}
            for k_block in range(2):  # k={0,1,2} or {3,4,5}
                i_base = i_block * 2
                k_base = k_block * 3
                
                # j={0,1}: 2x2 * 2x3
                # Split k into {k_base, k_base+1} and {k_base+2}
                # 2x2 * 2x2: Strassen
                add_strassen_block(i_base, k_base, 0, 0)
                
                # 2x2 * 2x1: schoolbook
                add_schoolbook_block(i_base, 2, k_base+2, 1, 0, 2, 0)
                
                # j={2}: 2x1 * 1x3: schoolbook
                add_schoolbook_block(i_base, 2, k_base, 3, 2, 1, 0)
        
        # Part 2: j in {3,4}
        for i_block in range(2):
            for k_block in range(2):
                i_base = i_block * 2
                k_base = k_block * 3
                
                # 2x2 * 2x3: split k into {k_base, k_base+1} and {k_base+2}
                # 2x2 * 2x2: Strassen
                add_strassen_block(i_base, k_base, 3, 0)
                
                # 2x2 * 2x1: schoolbook
                add_schoolbook_block(i_base, 2, k_base+2, 1, 3, 2, 0)
        
        return {"U": U, "V": V, "W": W}
    
    # Try the block Strassen decomposition
    try:
        result = build_decomposition()
        if verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = 96
            print(f"Block Strassen decomposition verified with rank {best_rank}")
    except Exception as e:
        print(f"Block Strassen failed: {e}")
    
    elapsed = time.time() - start_time
    return best_result

DIMENSIONS = [4, 5, 6]
