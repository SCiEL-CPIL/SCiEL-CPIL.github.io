"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U_r ⊗ V_r ⊗ W_r = ones tensor."""
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    rank = U.shape[1]
    # Check tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For efficiency, check a few random entries if rank is large
    # But for exactness, check all entries for small rank
    if rank <= 200:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True
    else:
        # Check a sample
        for _ in range(100):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            s = 0
            for r in range(rank):
                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if s != 1:
                return False
        return True

def _strassen_2x2_block():
    """
    Return the Strassen decomposition for 2x2 x 2x2 multiplication.
    This gives rank 7 instead of 8.
    
    For 2x2 matrices:
    C[0,0] = M1 - M2 + M4 - M7
    C[0,1] = M3 + M7
    C[1,0] = M5 + M6
    C[1,1] = M1 - M3 + M4 + M7
    
    Where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    """
    # We need to express this as factor matrices.
    # For the full 4x5x5 problem, we'll use this for the top-left 2x2 block
    # and schoolbook for the rest.
    
    # Actually, let's think about this differently.
    # The 4x5 matrix has rows 0,1,2,3 and cols 0,1,2,3,4
    # The 5x5 matrix has rows 0,1,2,3,4 and cols 0,1,2,3,4
    # The output 4x5 matrix has rows 0,1,2,3 and cols 0,1,2,3,4
    
    # Block decomposition:
    # Split A into 2x2 blocks of size 2x2 (top-left), and the rest
    # Split B into 2x2 blocks of size 2x2 (top-left), and the rest
    # Split C into 2x2 blocks of size 2x2 (top-left), and the rest
    
    # Top-left 2x2 block of C:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:5] * B[2:5, 0:2]
    # The first term is 2x2 x 2x2, which can use Strassen (rank 7)
    # The second term involves 2x3 x 3x2, which is schoolbook (rank 18)
    
    # Let me just do a simple block approach:
    # Use Strassen for the (0,0) 2x2 block contribution
    # Use schoolbook for everything else
    
    # Actually, the cleanest approach: 
    # Partition the computation into blocks.
    # For each output element C[i,k] = sum_j A[i,j] B[j,k]
    # Group by (i_block, j_block, k_block)
    
    # Let's use a simpler block strategy:
    # Split j into two groups: j in {0,1} and j in {2,3,4}
    # Split i into two groups: i in {0,1} and i in {2,3}
    # Split k into two groups: k in {0,1} and k in {2,3,4}
    
    # Block (i={0,1}, j={0,1}, k={0,1}): 2x2 x 2x2 -> Strassen rank 7
    # All other blocks: schoolbook
    
    # Total rank = 7 + (total schoolbook - 8 for that block)
    # Total schoolbook = 4*5*5 = 100
    # Block (i={0,1}, j={0,1}, k={0,1}) has 2*2*2=8 terms in schoolbook
    # So rank = 7 + 100 - 8 = 99
    
    # Hmm, that's only a savings of 1. Let me think more carefully.
    
    # Actually, I can apply Strassen more broadly. Let me use a different approach.
    
    # For the 4x5 x 5x5 case, let me try to use Strassen on overlapping blocks.
    
    # Alternative: use the known result that M(4,5,5) <= 98 or less.
    # Let me construct a valid rank-98 decomposition using Strassen.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook rank 100
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook(n, m, p)
    
    # Try to improve using Strassen for the 2x2 x 2x2 sub-block
    # Partition: i-block {0,1}, j-block {0,1}, k-block {0,1}
    # This block contributes 8 terms in schoolbook, but only 7 in Strassen
    
    # Build the full decomposition with one Strassen block
    # and schoolbook for all other (i,j,k) triples
    
    i_block = {0, 1}
    j_block = {0, 1}
    k_block = {0, 1}
    
    # Strassen factors for 2x2 x 2x2:
    # We need A_r (2x2), B_r (2x2), C_r (2x2) for r=1..7
    # such that sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] = 1 for all i,j,k in {0,1}
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C[0,0] = M1 - M2 + M4 - M7
    # C[0,1] = M3 + M7
    # C[1,0] = M5 + M6
    # C[1,1] = M1 - M3 + M4 + M7
    
    # For the tensor decomposition, we need:
    # sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] = 1
    
    # Let me define the Strassen factors explicitly.
    # A_r are 2x2 matrices, B_r are 2x2, C_r are 2x2.
    
    # r=1: M1 = (A00+A11)(B00+B11)
    # A_1[i,j]: coefficient of A[i,j] in (A00+A11)
    #   (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1
    # B_1[j,k]: coefficient of B[j,k] in (B00+B11)
    #   (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1
    # C_1[k,i]: contributes to C[0,0] with +1
    #   C[0,0] gets +M1, so C_1[0,0] = 1
    #   All other C entries: 0
    
    # Let me build these properly.
    
    # I'll represent each factor as a flat array of size 4 for 2x2.
    # For A_r: flat[i*2+j]
    # For B_r: flat[j*2+k]
    # For C_r: flat[k*2+i]
    
    strassen_A = []
    strassen_B = []
    strassen_C = []
    
    # M1 = (A00+A11)(B00+B11)
    A1 = [1,0,0,1]  # A00=1, A01=0, A10=0, A11=1
    B1 = [1,0,0,1]  # B00=1, B01=0, B10=0, B11=1
    C1 = [1,0,0,0]  # C00=1 (flat: k*2+i, so C[0,0]=1)
    strassen_A.append(A1); strassen_B.append(B1); strassen_C.append(C1)
    
    # M2 = (A10+A11)B00
    A2 = [0,0,1,1]
    B2 = [1,0,0,0]  # B00=1
    C2 = [0,1,0,0]  # C[0,0] gets -M2, so C_2[0,0] = -1
    # Wait, C[0,0] = M1 - M2 + M4 - M7
    # So M2 contributes -1 to C[0,0]
    C2 = [-1,0,0,0]
    strassen_A.append(A2); strassen_B.append(B2); strassen_C.append(C2)
    
    # M3 = A00(B01-B11)
    A3 = [1,0,0,0]
    B3 = [0,1,0,-1]  # B01=1, B11=-1
    C3 = [0,0,0,1]  # C[0,1] gets +M3, C[1,1] gets -M3
    # C[0,1]: flat index = k*2+i = 0*2+1 = 1 -> +1
    # C[1,1]: flat index = k*2+i = 1*2+1 = 3 -> -1
    C3 = [0,1,0,-1]
    strassen_A.append(A3); strassen_B.append(B3); strassen_C.append(C3)
    
    # M4 = A11(B10-B00)
    A4 = [0,0,0,1]
    B4 = [0,0,-1,1]  # B10=-1, B11=1... wait
    # B10-B00: B[1,0] - B[0,0]
    # flat: B[0,0]=index0, B[0,1]=index1, B[1,0]=index2, B[1,1]=index3
    # So B4 = [-1, 0, 1, 0]
    B4 = [-1, 0, 1, 0]
    C4 = [1,0,0,1]  # C[0,0] gets +M4, C[1,1] gets +M4
    strassen_A.append(A4); strassen_B.append(B4); strassen_C.append(C4)
    
    # M5 = (A00+A01)B11
    A5 = [1,1,0,0]
    B5 = [0,0,0,1]
    C5 = [0,0,1,0]  # C[1,0] gets +M5, flat: k=1,i=0 -> index 2
    strassen_A.append(A5); strassen_B.append(B5); strassen_C.append(C5)
    
    # M6 = (A10-A00)(B00+B10)
    A6 = [-1,0,1,0]
    B6 = [1,0,1,0]  # B00=1, B10=1
    C6 = [0,0,1,0]  # C[1,0] gets +M6
    strassen_A.append(A6); strassen_B.append(B6); strassen_C.append(C6)
    
    # M7 = (A01-A11)(B01+B11)
    A7 = [0,1,0,-1]
    B7 = [0,1,0,1]  # B01=1, B11=1
    C7 = [0,1,0,-1]  # C[0,0] gets -M7, C[0,1] gets +M7, C[1,1] gets +M7
    # C[0,0]: index 0 -> -1
    # C[0,1]: index 1 -> +1
    # C[1,1]: index 3 -> +1
    C7 = [-1,1,0,1]
    strassen_A.append(A7); strassen_B.append(B7); strassen_C.append(C7)
    
    # Verify Strassen decomposition
    # For all i,j,k in {0,1}: sum_r A_r[i,j]*B_r[j,k]*C_r[k,i] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += strassen_A[r][i*2+j] * strassen_B[r][j*2+k] * strassen_C[r][k*2+i]
                if s != 1:
                    # Fallback to schoolbook
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Now build the full 4x5x5 decomposition
    # Use Strassen for the (i,j,k) block where i in {0,1}, j in {0,1}, k in {0,1}
    # Use schoolbook for all other (i,j,k) triples
    
    total_rank = 7 + (n*m*p - 8)
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col = 0
    # Add Strassen factors for the 2x2x2 block
    for r in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii*m + jj, col] = strassen_A[r][ii*2+jj]
        for jj in range(2):
            for kk in range(2):
                V[jj*p + kk, col] = strassen_B[r][jj*2+kk]
        for kk in range(2):
            for ii in range(2):
                W[kk*n + ii, col] = strassen_C[r][kk*2+ii]
        col += 1
    
    # Add schoolbook factors for all other (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in i_block and j in j_block and k in k_block:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to pure schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
