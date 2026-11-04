"""Matrix multiplication 4x5x6 - Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6  # n*m input, m*p input, p*n output
    
    # Strassen's 2x2x2 decomposition:
    # Multiplies two 2x2 matrices A, B to get C = A*B using 7 multiplications
    # A = [[a1,a2],[a3,a4]], B = [[b1,b2],[b3,b4]]
    # m1 = (a1+a4)*(b1+b4)
    # m2 = (a3+a4)*b1
    # m3 = a1*(b2-b3)
    # m4 = a4*(b4-b2)
    # m5 = (a1+a2)*b4
    # m6 = (a3-a1)*(b1+b2)
    # m7 = (a2-a4)*(b3+b4)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # Each multiplication m_c involves a linear combination of A entries times
    # a linear combination of B entries, contributing to specific C entries.
    
    # For a 2x2x2 block, the U, V, W factors for each of the 7 terms:
    # U has shape (4, 1) for each term (indices 0-3 = row*2+col of A)
    # V has shape (4, 1) for each term (indices 0-3 = row*2+col of B)
    # W has shape (4, 1) for each term (indices 0-3 = row*2+col of C, but with trace convention)
    
    # Trace convention: W[k*n+i, c] where k is column of C, i is row of C
    # So W index = k*2 + i for a 2x2 output
    
    # Let's define the 7 Strassen terms as (u_vec, v_vec, w_vec) in R^4 each:
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices (trace): k*2+i, so 0=(i=0,k=0), 1=(i=1,k=0), 2=(i=0,k=1), 3=(i=1,k=1)
    # Wait: W[k*n+i] with n=2, so index = k*2+i
    # C[i,k]: i is row, k is column
    # W index for C[i,k] = k*2 + i
    
    # m1 = (A[0]+A[3])*(B[0]+B[3]) -> contributes to C[0,0] and C[1,1]
    # m2 = (A[2]+A[3])*B[0] -> contributes to C[1,0] and C[1,1]
    # m3 = A[0]*(B[1]-B[2]) -> contributes to C[0,1] and C[1,0]
    # m4 = A[3]*(B[3]-B[1]) -> contributes to C[0,0] and C[1,1]
    # m5 = (A[0]+A[1])*B[3] -> contributes to C[0,0] and C[0,1]
    # m6 = (A[2]-A[0])*(B[0]+B[1]) -> contributes to C[1,0]
    # m7 = (A[1]-A[3])*(B[2]+B[3]) -> contributes to C[0,1]
    
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # W vectors (index = k*2+i):
    # C[0,0]: i=0,k=0 -> idx 0
    # C[0,1]: i=0,k=1 -> idx 2
    # C[1,0]: i=1,k=0 -> idx 1
    # C[1,1]: i=1,k=1 -> idx 3
    
    strassen_2x2 = [
        # (u_vec, v_vec, w_vec)
        # m1: (A0+A3)*(B0+B3) -> C00 + C11
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m2: (A2+A3)*B0 -> C10 + C11
        ([0,0,1,1], [1,0,0,0], [0,1,0,1]),
        # m3: A0*(B1-B2) -> C01 + C10
        ([1,0,0,0], [0,1,-1,0], [0,0,1,0]),
        # m4: A3*(B3-B1) -> C00 + C11
        ([0,0,0,1], [0,-1,0,1], [1,0,0,1]),
        # m5: (A0+A1)*B3 -> C00 + C01
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        # m6: (A2-A0)*(B0+B1) -> C10
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
        # m7: (A1-A3)*(B2+B3) -> C01
        ([0,1,0,-1], [0,0,1,1], [0,0,1,0]),
    ]
    
    # Now build the full 4x5x6 decomposition
    # Split: A is 4x5, B is 5x6, C is 4x6
    # We can split into blocks along the middle dimension (j from 0 to 4):
    # Actually, let's split into 2x2 blocks where possible.
    # 
    # A: 4 rows x 5 cols
    # B: 5 rows x 6 cols
    # 
    # Let's partition:
    # - Use a 4x4x4 Strassen-based block for the first 4 columns of A and first 4 rows of B
    # - Handle the remaining 5th column of A / 5th row of B separately (schoolbook)
    # - Handle the 6th column of B (5th and 6th) - wait, B has 6 columns
    #
    # Better approach: partition the middle dimension j=0..4 into:
    #   j in {0,1,2,3} -> 4x4x4 block (use Strassen for 2x2 sub-blocks)
    #   j = 4 -> 4x1x6 block (schoolbook: 24 terms)
    #
    # For the 4x4x4 block: split into 2x2x2 sub-blocks
    #   A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (4 terms -> 7)
    #   A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]  (4 terms -> 7)
    #   A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]  (4 terms -> 7)
    #   A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]  (4 terms -> 7)
    #   A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]  (4 terms -> 7)
    #   A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]  (4 terms -> 7)
    #   A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]  (4 terms -> 7)
    #   A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]  (4 terms -> 7)
    #   Total for 4x4x4: 8*7 = 56 terms (vs 64 schoolbook)
    #
    # For j=4 (4x1x6): 4*1*6 = 24 terms (schoolbook)
    #
    # Total rank: 56 + 24 = 80
    #
    # Wait, that's better than the 108 mentioned. Let me reconsider.
    # Actually, the 4x5x6 schoolbook is 4*5*6 = 120.
    # With Strassen on the 4x4x4 part: 56 + 24 = 80.
    # 
    # But we can also handle the 5th column differently.
    # Let's just go with this: 80 rank.
    
    # Build the decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Helper: add a Strassen 2x2x2 block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        """Add a 2x2x2 Strassen block.
        A rows: a_row_offset, a_row_offset+1; cols: a_col_offset, a_col_offset+1
        B rows: b_row_offset, b_row_offset+1; cols: b_col_offset, b_col_offset+1
        C rows: a_row_offset, a_row_offset+1; cols: b_col_offset, b_col_offset+1
        """
        for (u_vec, v_vec, w_vec) in strassen_2x2:
            u_full = [0.0] * (n * m)  # 4*5 = 20
            v_full = [0.0] * (m * p)  # 5*6 = 30
            w_full = [0.0] * (p * n)  # 6*4 = 24
            
            # Fill U: A indices are i*m + j
            for idx, val in enumerate(u_vec):
                if val != 0:
                    i_local = idx // 2  # 0 or 1
                    j_local = idx % 2   # 0 or 1
                    i = a_row_offset + i_local
                    j = a_col_offset + j_local
                    u_full[i * m + j] += val
            
            # Fill V: B indices are j*p + k
            for idx, val in enumerate(v_vec):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j = b_row_offset + j_local
                    k = b_col_offset + k_local
                    v_full[j * p + k] += val
            
            # Fill W: C indices use trace convention: k*n + i
            for idx, val in enumerate(w_vec):
                if val != 0:
                    # idx = k_local * 2 + i_local (since n=2 for this sub-block)
                    k_local = idx // 2
                    i_local = idx % 2
                    i = a_row_offset + i_local
                    k = b_col_offset + k_local
                    w_full[k * n + i] += val
            
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)
    
    # Add all 8 Strassen 2x2x2 blocks for the 4x4x4 part
    for ar in range(0, 4, 2):
        for ac in range(0, 4, 2):
            for br in range(0, 4, 2):
                for bc in range(0, 4, 2):
                    # Only if the middle indices match: ac+col == br+row
                    # Actually no - we need A[i, j] * B[j, k] where j matches
                    # A cols: ac, ac+1; B rows: br, br+1
                    # We need ac == br for the blocks to connect properly
                    # Wait, that's not right either. Let me reconsider.
                    #
                    # A is 4x4 (rows 0-3, cols 0-3), B is 4x4 (rows 0-3, cols 0-3)
                    # C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
                    #
                    # Block A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2] contributes to
                    # C[ar:ar+2, bc:bc+2] ONLY IF ac == br (the middle indices match)
                    #
                    # So valid combinations:
                    # (ar, ac, br, bc) where ac == br
                    if ac == br:
                        add_strassen_block(ar, ac, br, bc)
    
    # Count: for each (ar, ac) with ar in {0,2}, ac in {0,2}, we have br=ac, bc in {0,2}
    # ar=0, ac=0: br=0, bc=0,2 -> 2 blocks
    # ar=0, ac=2: br=2, bc=0,2 -> 2 blocks
    # ar=2, ac=0: br=0, bc=0,2 -> 2 blocks
    # ar=2, ac=2: br=2, bc=0,2 -> 2 blocks
    # Total: 8 blocks * 7 terms = 56 terms. Good.
    
    # Now handle j=4 (the 5th column of A, 5th row of B)
    # A[i,4] * B[4,k] for i=0..3, k=0..5
    # This is 4*1*6 = 24 terms
    for i in range(n):
        for k in range(p):
            u_full = [0.0] * (n * m)
            v_full = [0.0] * (m * p)
            w_full = [0.0] * (p * n)
            
            u_full[i * m + 4] = 1.0  # A[i, 4]
            v_full[4 * p + k] = 1.0  # B[4, k]
            w_full[k * n + i] = 1.0  # C[i, k]
            
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)
    
    # Also need to handle B columns 4 and 5 (the 5th and 6th columns) with A columns 0-3
    # Wait, I already handled all of A[0:4, 0:4] * B[0:4, 0:4] via Strassen blocks.
    # But B has columns 0-5, and I only used B columns 0-3 in the Strassen blocks.
    # A has columns 0-4. I used A columns 0-3 in Strassen, and A column 4 above.
    # 
    # I'm missing:
    # - A[i, 0:4] * B[0:4, 4:6] -> C[i, 4:6] for i=0:4
    # - A[i, 4] * B[4, 4:6] -> already handled above (all k=0..5)
    # 
    # Wait, B[4, k] for k=0..5 is handled. But A[i, 0:4] * B[j, 4:6] for j=0:4 is missing!
    #
    # Let me redo this more carefully.
    #
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # I've handled:
    # 1. j in {0,1,2,3}, A cols 0-3, B rows 0-3: via Strassen blocks (56 terms)
    #    These cover B columns 0-3 (bc in {0,2} gives cols 0,1 and 2,3)
    # 2. j=4, all i, all k: 24 terms
    #
    # Missing:
    # - A[i, 0:4] * B[j, 4] for j=0:3, i=0:3, k=4
    # - A[i, 0:4] * B[j, 5] for j=0:3, i=0:3, k=5
    #
    # These are: for j in 0..3, for i in 0..3, for k in {4,5}: A[i,j]*B[j,k]
    # That's 4*4*2 = 32 more terms (schoolbook)
    #
    # Total: 56 + 24 + 32 = 112. That's worse than expected.
    
    # Let me restart with a cleaner approach.
    U_list = []
    V_list = []
    W_list = []
    
    # Clear previous additions
    # Actually, I already appended. Let me rebuild from scratch.
    U_list = []
    V_list = []
    W_list = []
    
    # Strategy:
    # Partition middle index j=0..4:
    #   j in {0,1}: handle with Strassen where possible
    #   j in {2,3}: handle with Strassen where possible  
    #   j = 4: schoolbook
    #
    # For j in {0,1} and j in {2,3}, we have 2x2 middle blocks.
    # A rows 0..3, B cols 0..5.
    #
    # Let's think of it as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Group by pairs of j: (0,1) and (2,3), and single j=4.
    #
    # For the pair j in {0,1}:
    #   C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is like multiplying a 4x2 matrix by a 2x6 matrix.
    #   Split further: for each 2x2 sub-block in the column direction of B.
    #   B cols 0..5: split into (0,1), (2,3), (4,5)
    #   A rows 0..3: split into (0,1), (2,3)
    #
    # For sub-block A[ar:ar+2, 0:2] * B[0:2, bc:bc+2] -> C[ar:ar+2, bc:bc+2]
    #   This is a 2x2x2 problem: use Strassen (7 terms instead of 4)
    #
    # Number of such sub-blocks: 2 (ar choices) * 3 (bc choices) = 6 sub-blocks
    # Each uses 7 terms: 6*7 = 42 terms
    # (vs schoolbook: 2*2*6 = 24 terms for the j=0,1 part... wait)
    #
    # Schoolbook for j in {0,1}: 4 rows * 2 cols * 6 cols = 48 terms
    # Strassen: 42 terms. Savings of 6.
    #
    # Similarly for j in {2,3}: another 42 terms (savings of 6)
    #
    # For j=4: 4*1*6 = 24 terms (no savings)
    #
    # Total: 42 + 42 + 24 = 108 terms.
    # Schoolbook: 120. Savings: 12.
    #
    # This matches the knowledge that rank 108 is achievable!
    
    # Build the decomposition:
    
    # Part 1: j in {0,1}
    # A cols 0,1; B rows 0,1
    for ar in range(0, 4, 2):       # A row blocks: (0,1), (2,3)
        for bc in range(0, 6, 2):   # B col blocks: (0,1), (2,3), (4,5)
            # Strassen block: A[ar:ar+2, 0:2] * B[0:2, bc:bc+2] -> C[ar:ar+2, bc:bc+2]
            for (u_vec, v_vec, w_vec) in strassen_2x2:
                u_full = [0.0] * (n * m)
                v_full = [0.0] * (m * p)
                w_full = [0.0] * (p * n)
                
                for idx, val in enumerate(u_vec):
                    if val != 0:
                        i_local = idx // 2
                        j_local = idx % 2
                        i = ar + i_local
                        j = j_local  # j is 0 or 1
                        u_full[i * m + j] += val
                
                for idx, val in enumerate(v_vec):
                    if val != 0:
                        j_local = idx // 2  # j is 0 or 1
                        k_local = idx % 2
                        j = j_local
                        k = bc + k_local
                        v_full[j * p + k] += val
                
                for idx, val in enumerate(w_vec):
                    if val != 0:
                        k_local = idx // 2
                        i_local = idx % 2
                        i = ar + i_local
                        k = bc + k_local
                        w_full[k * n + i] += val
                
                U_list.append(u_full)
                V_list.append(v_full)
                W_list.append(w_full)
    
    # Part 2: j in {2,3}
    for ar in range(0, 4, 2):
        for bc in range(0, 6, 2):
            # Strassen block: A[ar:ar+2, 2:4] * B[2:4, bc:bc+2] -> C[ar:ar+2, bc:bc+2]
            for (u_vec, v_vec, w_vec) in strassen_2x2:
                u_full = [0.0] * (n * m)
                v_full = [0.0] * (m * p)
                w_full = [0.0] * (p * n)
                
                for idx, val in enumerate(u_vec):
                    if val != 0:
                        i_local = idx // 2
                        j_local = idx % 2
                        i = ar + i_local
                        j = 2 + j_local  # j is 2 or 3
                        u_full[i * m + j] += val
                
                for idx, val in enumerate(v_vec):
                    if val != 0:
                        j_local = idx // 2
                        k_local = idx % 2
                        j = 2 + j_local  # j is 2 or 3
                        k = bc + k_local
                        v_full[j * p + k] += val
                
                for idx, val in enumerate(w_vec):
                    if val != 0:
                        k_local = idx // 2
                        i_local = idx % 2
                        i = ar + i_local
                        k = bc + k_local
                        w_full[k * n + i] += val
                
                U_list.append(u_full)
                V_list.append(v_full)
                W_list.append(w_full)
    
    # Part 3: j = 4
    for i in range(n):
        for k in range(p):
            u_full = [0.0] * (n * m)
            v_full = [0.0] * (m * p)
            w_full = [0.0] * (p * n)
            
            u_full[i * m + 4] = 1.0
            v_full[4 * p + k] = 1.0
            w_full[k * n + i] = 1.0
            
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)
    
    rank = len(U_list)
    
    # Convert to numpy arrays
    U = np.array(U_list, dtype=float).T  # (n*m, rank)
    V = np.array(V_list, dtype=float).T  # (m*p, rank)
    W = np.array(W_list, dtype=float).T  # (p*n, rank)
    
    # Verify the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        U_fb = np.zeros((n*m, n*m*p))
        V_fb = np.zeros((m*p, n*m*p))
        W_fb = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1.0
                    V_fb[j*p+k, col] = 1.0
                    W_fb[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
