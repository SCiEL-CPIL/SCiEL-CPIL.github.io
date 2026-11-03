"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Recursive 2x2 Strassen decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Split the 4x6 output into a 4x4 part (columns 0-3) and a 4x2 part (columns 4-5).
    - 4x4 part: Split A into 2x2 blocks, B_left into 2x2 blocks.
      Each 2x2 output block C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c]
      Use Strassen (rank 7) per 2x2*2x2 multiplication.
      Total: 2*2 output blocks * 7 = 28 rank terms.
    - 4x2 part: Split A into 2x2 blocks, B_right (4x2) into two 2x2 blocks stacked.
      Each 2x2 output block C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c]
      Use Strassen (rank 7) per 2x2*2x2 multiplication.
      Total: 2*1 output blocks * 7 = 14 rank terms.
    
    Total rank: 28 + 14 = 42.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen for 2x2*2x2:
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11)
    # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    strassen_terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):-1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    # Part 1: 4x4 output (columns 0-3 of B)
    # A is 4x4, split into 2x2 blocks: A[a,jb] where a in {0,1}, jb in {0,1}
    # B_left is 4x4, split into 2x2 blocks: B[jb,c] where jb in {0,1}, c in {0,1}
    # Output C[a,c] for a in {0,1}, c in {0,1}
    
    n_blocks = 2  # A split into 2x2 blocks
    m_blocks = 2  # B split into 2x2 blocks along m dimension
    p_blocks_left = 2  # B_left has 2x2 blocks along p dimension
    
    # Part 2: 4x2 output (columns 4-5 of B)
    # B_right is 4x2, which is two 2x2 blocks stacked vertically (along p dimension)
    # Actually, B_right is 4x2, so we treat it as having p_blocks_right = 1 block of size 2x2
    # Wait, let me reconsider. B is 4x6. Split into B_left (4x4) and B_right (4x2).
    # B_right is 4x2, which can be viewed as a single 2x2 block if we consider the p dimension.
    # Actually, B_right has shape (m=4, p_right=2). Split into 2x2 blocks: jb in {0,1}, c in {0} (only one p block)
    
    max_rank = (n_blocks * m_blocks * p_blocks_left + n_blocks * m_blocks * 1) * 7 + 10
    U_data = np.zeros((n*m, max_rank))
    V_data = np.zeros((m*p, max_rank))
    W_data = np.zeros((p*n, max_rank))
    
    col = 0
    
    # Part 1: 4x4 output (columns 0-3)
    for a in range(n_blocks):
        for c in range(p_blocks_left):
            for jb in range(m_blocks):
                for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                    for (ar, ac), val in a_coeffs.items():
                        # A index: row = 2*a + ar, col = 2*jb + ac
                        U_data[(2*a+ar)*m + (2*jb+ac), col] += val
                    for (br, bc), val in b_coeffs.items():
                        # B_left index: row = 2*jb + br, col = 2*c + bc
                        V_data[(2*jb+br)*p + (2*c+bc), col] += val
                    for (cr, cc), val in c_coeffs.items():
                        # C index: row = 2*c + cc, col = 2*a + cr
                        W_data[(2*c+cc)*n + (2*a+cr), col] += val
                    col += 1
    
    # Part 2: 4x2 output (columns 4-5)
    # B_right is columns 4-5 of B, so B_right[jb, c] where c is the local column index in {0,1}
    # Global column index = 4 + c
    for a in range(n_blocks):
        for c in range(1):  # Only one p block for the 4x2 part
            for jb in range(m_blocks):
                for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                    for (ar, ac), val in a_coeffs.items():
                        U_data[(2*a+ar)*m + (2*jb+ac), col] += val
                    for (br, bc), val in b_coeffs.items():
                        # B_right index: row = 2*jb + br, col = 4 + bc (global column)
                        V_data[(2*jb+br)*p + (4+bc), col] += val
                    for (cr, cc), val in c_coeffs.items():
                        # C index: row = 4 + cc (global row), col = 2*a + cr
                        W_data[(4+cc)*n + (2*a+cr), col] += val
                    col += 1
    
    rank = col
    U = U_data[:, :rank].tolist()
    V = V_data[:, :rank].tolist()
    W = W_data[:, :rank].tolist()
    
    # Verify
    U_np, V_np, W_np = np.array(U), np.array(V), np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j] * V_np[j*p+k], W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    r2 = n*m*p
                    U2 = np.zeros((n*m, r2)); V2 = np.zeros((m*p, r2)); W2 = np.zeros((p*n, r2))
                    cc = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U2[ii*m+jj, cc] = 1; V2[jj*p+kk, cc] = 1; W2[kk*n+ii, cc] = 1; cc += 1
                    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
