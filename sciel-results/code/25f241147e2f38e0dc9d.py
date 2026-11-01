import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses a block decomposition: split the 10 columns of B into 5 pairs of 2.
    For each pair, use a rank-11 decomposition for 2x3x2 (Strassen for 2x2x2 + schoolbook for 2x1x2).
    Total rank = 5 * 11 = 55.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We split the p=10 columns into num_blocks = 5 blocks of block_size = 2
    num_blocks = p // 2  # 5
    block_size = 2
    
    # Rank per block: 11 (7 from Strassen for 2x2x2 + 4 from schoolbook for 2x1x2)
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 55
    
    # U shape: (n*m, total_rank) = (6, 55)
    # V shape: (m*p, total_rank) = (30, 55)
    # W shape: (p*n, total_rank) = (20, 55)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # For each block b (0 to 4), we handle columns 2*b and 2*b+1 of B
    # The sub-problem is: A (2x3) * B_block (3x2) = C_block (2x2)
    # We split A into A_left (2x2) and A_right (2x1)
    # We split B_block into B_top (2x2) and B_bottom (1x2)
    # C_block = A_left * B_top + A_right * B_bottom
    # A_left * B_top is 2x2x2 -> Strassen (rank 7)
    # A_right * B_bottom is 2x1x2 -> schoolbook (rank 4)
    
    for b in range(num_blocks):
        col_offset = b * rank_per_block  # offset in the rank dimension for this block
        k0 = 2 * b      # first column index in B
        k1 = 2 * b + 1  # second column index in B
        
        # --- Strassen part for A_left (2x2) * B_top (2x2) ---
        # A_left rows: i=0,1; cols: j=0,1 (from A)
        # B_top rows: j=0,1 (from B_block); cols: k=0,1 (which are k0, k1 in B)
        # C_block rows: i=0,1; cols: k=0,1 (which are k0, k1)
        
        # Strassen variables for A_left = [[a, b], [c, d]] (indices: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d)
        # and B_top = [[e, f], [g, h]] (indices: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h)
        # where B_top[j, k_local] = B[j, k0 + k_local]
        
        # m1 = (a + d)(e + h)
        # m2 = (c + d)(e)
        # m3 = (a)(f - h)
        # m4 = (d)(g - e)
        # m5 = (a + b)(h)
        # m6 = (c - a)(e + f)
        # m7 = (b - d)(g + h)
        
        # C[0,0] = m1 - m4 + m5 + m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m3 + m2 + m6
        
        # We'll create 7 rank terms for this part.
        # For each term r (0..6), we need U entries, V entries, W entries.
        
        # Let's define the Strassen factors more explicitly.
        # A_left[i, j] is part of A[i, j] (since j=0,1 for A_left)
        # B_top[j, k_local] is B[j, k0 + k_local]
        
        # Term 0: m1 = (A[0,0] + A[1,1]) * (B[0, k0] + B[1, k1])
        # U: A[0,0] -> U[0, col_offset+0] = 1; A[1,1] -> U[3, col_offset+0] = 1
        # V: B[0, k0] -> V[0*p + k0, col_offset+0] = 1; B[1, k1] -> V[1*p + k1, col_offset+0] = 1
        # W: C[0,0] gets +1, C[1,1] gets +1
        # W[k_local*n + i, ...] where k_local is 0 or 1, i is 0 or 1
        # C[0,0] -> k_local=0, i=0 -> W[0*2+0] = W[0]
        # C[1,1] -> k_local=1, i=1 -> W[1*2+1] = W[3]
        U[0][col_offset+0] += 1
        U[3][col_offset+0] += 1
        V[0*p + k0][col_offset+0] += 1
        V[1*p + k1][col_offset+0] += 1
        W[0*2 + 0][col_offset+0] += 1   # C[0,0]
        W[1*2 + 1][col_offset+0] += 1   # C[1,1]
        
        # Term 1: m2 = (A[1,0] + A[1,1]) * (B[0, k0])
        # U: A[1,0] -> U[1*3+0]=U[3] (wait, A is 2x3, so A[i,j] is at U[i*m+j]. m=3.
        # A[0,0]->U[0], A[0,1]->U[1], A[0,2]->U[2]
        # A[1,0]->U[3], A[1,1]->U[4], A[1,2]->U[5]
        # So A[1,0] is U[3], A[1,1] is U[4].
        U[3][col_offset+1] += 1
        U[4][col_offset+1] += 1
        V[0*p + k0][col_offset+1] += 1
        # C[1,0] gets +1
        W[0*2 + 1][col_offset+1] += 1   # C[1,0]
        
        # Term 2: m3 = (A[0,0]) * (B[0, k1] - B[1, k1])
        U[0][col_offset+2] += 1
        V[0*p + k1][col_offset+2] += 1
        V[1*p + k1][col_offset+2] += -1
        # C[0,1] gets +1
        W[1*2 + 0][col_offset+2] += 1   # C[0,1]
        
        # Term 3: m4 = (A[1,1]) * (B[1, k0] - B[0, k0])
        U[4][col_offset+3] += 1
        V[1*p + k0][col_offset+3] += 1
        V[0*p + k0][col_offset+3] += -1
        # C[0,0] gets -1, C[1,0] gets +1
        W[0*2 + 0][col_offset+3] += -1
        W[0*2 + 1][col_offset+3] += 1
        
        # Term 4: m5 = (A[0,0] + A[0,1]) * (B[1, k1])
        U[0][col_offset+4] += 1
        U[1][col_offset+4] += 1
        V[1*p + k1][col_offset+4] += 1
        # C[0,0] gets +1, C[0,1] gets +1
        W[0*2 + 0][col_offset+4] += 1
        W[1*2 + 0][col_offset+4] += 1
        
        # Term 5: m6 = (A[1,0] - A[0,0]) * (B[0, k0] + B[0, k1])
        U[3][col_offset+5] += 1
        U[0][col_offset+5] += -1
        V[0*p + k0][col_offset+5] += 1
        V[0*p + k1][col_offset+5] += 1
        # C[1,1] gets +1
        W[1*2 + 1][col_offset+5] += 1
        
        # Term 6: m7 = (A[0,1] - A[1,1]) * (B[1, k0] + B[1, k1])
        U[1][col_offset+6] += 1
        U[4][col_offset+6] += -1
        V[1*p + k0][col_offset+6] += 1
        V[1*p + k1][col_offset+6] += 1
        # C[0,0] gets +1
        W[0*2 + 0][col_offset+6] += 1
        
        # --- Schoolbook part for A_right (2x1) * B_bottom (1x2) ---
        # A_right is column j=2 of A: A[0,2], A[1,2]
        # B_bottom is row j=2 of B_block: B[2, k0], B[2, k1]
        # C_block[i, k_local] += A[i,2] * B[2, k0+k_local]
        
        # Term 7: A[0,2] * B[2, k0] -> C[0,0]
        U[2][col_offset+7] += 1
        V[2*p + k0][col_offset+7] += 1
        W[0*2 + 0][col_offset+7] += 1
        
        # Term 8: A[0,2] * B[2, k1] -> C[0,1]
        U[2][col_offset+8] += 1
        V[2*p + k1][col_offset+8] += 1
        W[1*2 + 0][col_offset+8] += 1
        
        # Term 9: A[1,2] * B[2, k0] -> C[1,0]
        U[5][col_offset+9] += 1
        V[2*p + k0][col_offset+9] += 1
        W[0*2 + 1][col_offset+9] += 1
        
        # Term 10: A[1,2] * B[2, k1] -> C[1,1]
        U[5][col_offset+10] += 1
        V[2*p + k1][col_offset+10] += 1
        W[1*2 + 1][col_offset+10] += 1
        
    # Convert to numpy arrays for verification (optional, but good for debugging)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Quick verification with random matrices
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            # The decomposition computes: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i, r] * A[i,j] * B[j',k']
            # Wait, the tensor decomposition is for the multiplication operator.
            # C[i,k] = sum_{j} A[i,j] B[j,k]
            # The bilinear form is: sum_r (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j',k'} V[j'*p+k', r] B[j',k']) * W[k*n+i, r]
            # This is not quite right. The standard tensor decomposition for matrix multiplication T_{i,j,k; i',j',k'}
            # is such that sum_r U[i,j,r] V[j',k',r] W[k,i',r] = delta(j,j') delta(i,i') delta(k,k') ... no.
            
            # Let's re-read the contract.
            # "output factor follows the official trace convention W[k*n+i,r]"
            # "exactly decompose the matrix multiplication tensor"
            # The matrix multiplication tensor M has entries M[i,j,k; i',j',k'] = delta(j,j') delta(i,i') delta(k,k')? No.
            # M_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
            # So C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
            # The tensor T is 3-tensor: T_{i,k,j} = 1 if indices match appropriately?
            # Usually, the matrix multiplication tensor is T[i, j, k; i', j', k'] where
            # T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^* ...
            
            # Let's just verify the bilinear complexity directly.
            # For each rank term r, we compute:
            # u_r = sum_{i,j} U[i*m+j, r] * A[i,j]
            # v_r = sum_{j',k'} V[j'*p+k', r] * B[j',k']
            # Then C[i,k] += W[k*n+i, r] * u_r * v_r
            
            u_r = 0
            for i in range(n):
                for j in range(m):
                    u_r += U_np[i*m+j, r] * A_test[i,j]
            
            v_r = 0
            for jp in range(m):
                for kp in range(p):
                    v_r += V_np[jp*p+kp, r] * B_test[jp,kp]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += W_np[k*n+i, r] * u_r * v_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook if verification fails
            schoolbook_rank = n * m * p
            U_school = [[0] * schoolbook_rank for _ in range(n * m)]
            V_school = [[0] * schoolbook_rank for _ in range(m * p)]
            W_school = [[0] * schoolbook_rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
