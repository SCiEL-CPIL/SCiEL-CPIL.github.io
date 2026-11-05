"""Search-based program for matrix multiplication tensor decomposition 4x4 * 4x5."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20

# Strassen's 7-term decomposition for 2x2 * 2x2
# For C = A*B where A,B are 2x2:
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A10)*B11
# M6 = (A11-A00)*(B00+B10)
# M7 = (A01-A11)*(B11+B00)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# Strassen factors: U_str[i,j] for 2x2 input A (4 entries), V_str[j,k] for 2x2 input B (4 entries), W_str[k,i] for 2x2 output C (4 entries)
# Indexing: A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]], C = [[c00, c01],[c10, c11]]
# U indices: 0=a00, 1=a01, 2=a10, 3=a11
# V indices: 0=b00, 1=b01, 2=b10, 3=b11
# W indices: 0=c00, 1=c01, 2=c10, 3=c11

def _strassen_2x2_factors():
    """Return U, V, W factor matrices (4x7 each) for 2x2*2x2 Strassen."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A10)*B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A11-A00)*(B00+B10)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B11+B00)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[0, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_4x4_4x5():
    """
    Build a rank-65 decomposition for 4x4 * 4x5.
    
    Strategy: Split the 4x4 matrix into 2x2 blocks and use Strassen for the 4x4*4x4 part (rank 49),
    then handle the 4x4*4x1 part with schoolbook (rank 16). Total: 65.
    
    The 4x4*4x5 multiplication: C = A * B where A is 4x4, B is 4x5.
    Split B into B[:, :4] (4x4) and B[:, 4:] (4x1).
    C[:, :4] = A * B[:, :4]  (4x4 * 4x4, use Strassen, rank 49)
    C[:, 4:] = A * B[:, 4:]  (4x4 * 4x1, use schoolbook, rank 16)
    """
    # Strassen for 4x4 * 4x4: rank 49 = 7^2
    # This is the Kronecker product of 2x2 Strassen with itself.
    # U_4x4[i,j] = sum over Strassen terms...
    # Actually, for 4x4*4x4, we can use the 4x4 Strassen decomposition which has rank 49.
    # The 4x4*4x4 multiplication can be done with 49 multiplications using Strassen recursively.
    
    # Let's build it properly.
    # 4x4 A split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # 4x4 B split into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # C = A*B, C00 = A00*B00 + A01*B10, etc.
    # Using Strassen on the 2x2 "super-matrices":
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B10)
    # M7 = (A01-A11)*(B11+B00)
    
    # Each Mi is a 2x2 * 2x2 multiplication, which itself uses 7 Strassen terms.
    # So total rank = 7 * 7 = 49 for the 4x4*4x4 part.
    
    # For the U, V, W factors of the 4x4*4x4 part:
    # U has 16 rows (4x4 input A), 49 columns
    # V has 16 rows (4x4 input B), 49 columns  
    # W has 16 rows (4x4 output C), 49 columns
    
    # Let's construct using the Kronecker product structure.
    # For each of the 7 outer Strassen terms (say term t), we have:
    #   u_outer[t, i] for the 2x2 block index i (0-3)
    #   v_outer[t, j] for the 2x2 block index j (0-3)
    #   w_outer[t, k] for the 2x2 output block index k (0-3)
    
    # And for each inner 2x2*2x2 Strassen (term s, 0-6):
    #   u_inner[s, a] for the element index a within the 2x2 block
    #   v_inner[s, b] for the element index b within the 2x2 block
    #   w_inner[s, c] for the element index c within the 2x2 output block
    
    # The combined factor for (t, s) has:
    #   U row (block_i * 4 + elem_a): u_outer[t, block_i] * u_inner[s, elem_a]
    #   V row (block_j * 4 + elem_b): v_outer[t, block_j] * v_inner[s, elem_b]
    #   W row (block_k * 4 + elem_c): w_outer[t, block_k] * w_inner[s, elem_c]
    
    U_inner, V_inner, W_inner = _strassen_2x2_factors()
    
    # Outer Strassen for 2x2 block matrices
    # Same structure as inner but for block indices
    U_outer = U_inner.copy()
    V_outer = V_inner.copy()
    W_outer = W_inner.copy()
    
    rank_4x4_4x4 = 49
    U_44 = np.zeros((16, rank_4x4_4x4), dtype=np.int64)
    V_44 = np.zeros((16, rank_4x4_4x4), dtype=np.int64)
    W_44 = np.zeros((16, rank_4x4_4x4), dtype=np.int64)
    
    for t in range(7):  # outer Strassen term
        for s in range(7):  # inner Strassen term
            col = t * 7 + s
            for block_i in range(4):  # 2x2 block index for A
                for elem_a in range(4):  # element index within block
                    row = block_i * 4 + elem_a
                    U_44[row, col] = U_outer[block_i, t] * U_inner[elem_a, s]
            
            for block_j in range(4):  # 2x2 block index for B
                for elem_b in range(4):  # element index within block
                    row = block_j * 4 + elem_b
                    V_44[row, col] = V_outer[block_j, t] * V_inner[elem_b, s]
            
            for block_k in range(4):  # 2x2 block index for C
                for elem_c in range(4):  # element index within block
                    row = block_k * 4 + elem_c
                    W_44[row, col] = W_outer[block_k, t] * W_inner[elem_c, s]
    
    # Now for the 4x4 * 4x1 part (schoolbook, rank 16)
    # C_col4[i] = sum_j A[i,j] * B[j,4]
    # This is just 16 scalar multiplications.
    rank_44_41 = 16
    U_41 = np.zeros((16, rank_44_41), dtype=np.int64)
    V_41 = np.zeros((20, rank_44_41), dtype=np.int64)  # V has 20 rows (4*5)
    W_41 = np.zeros((20, rank_44_41), dtype=np.int64)  # W has 20 rows (5*4)
    
    # Wait, I need to be more careful about the indexing.
    # The full problem: A is 4x4 (NM=16), B is 4x5 (MP=20), C is 5x4 (PN=20)
    # C[k*n+i] where k is column of C (0-4), i is row of C (0-3)
    # B[j*p+k] where j is row of B (0-3), k is column of B (0-4)
    # A[i*m+j] where i is row of A (0-3), j is column of A (0-3)
    
    # For the 4x4*4x4 part (columns 0-3 of B, columns 0-3 of C):
    # U has 16 rows (A entries), V has 16 rows (B[:,0:4] entries), W has 16 rows (C[:,0:4] entries)
    # But in the full problem, V has 20 rows and W has 20 rows.
    # V rows 0-15 correspond to B[:,0:4], rows 16-19 correspond to B[:,4]
    # W rows 0-15 correspond to C[:,0:4], rows 16-19 correspond to C[:,4]
    
    # So for the 4x4*4x4 part, we use:
    # U_44: 16 x 49 (full U)
    # V_44_full: 20 x 49, where rows 0-15 are V_44, rows 16-19 are 0
    # W_44_full: 20 x 49, where rows 0-15 are W_44, rows 16-19 are 0
    
    # For the 4x4*4x1 part (column 4 of B, column 4 of C):
    # C[:,4][i] = sum_j A[i,j] * B[j,4]
    # U_41: 16 x 16, where U_41[i*4+j, col] = 1 if col = i*4+j
    # V_41: 20 x 16, where V_41[j*5+4, col] = 1 if col = i*4+j... 
    # Wait, this doesn't work directly because each product A[i,j]*B[j,4] contributes to C[i,4].
    # We need: for each (i,j) pair, a term with U[i,j]=1, V[j,4]=1, W[i,4]=1
    # That's 16 terms.
    
    U_41 = np.zeros((16, rank_44_41), dtype=np.int64)
    V_41 = np.zeros((20, rank_44_41), dtype=np.int64)
    W_41 = np.zeros((20, rank_44_41), dtype=np.int64)
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_41[i * 4 + j, col] = 1
            V_41[j * 5 + 4, col] = 1  # B[j, 4]
            W_41[4 * 4 + i, col] = 1   # C[i, 4] -> index 4*4+i = 16+i
    
    # Combine
    total_rank = rank_4x4_4x4 + rank_44_41  # 49 + 16 = 65
    
    U_full = np.zeros((16, total_rank), dtype=np.int64)
    V_full = np.zeros((20, total_rank), dtype=np.int64)
    W_full = np.zeros((20, total_rank), dtype=np.int64)
    
    # Fill in the 4x4*4x4 part (columns 0-48)
    U_full[:, :49] = U_44
    V_full[:16, :49] = V_44
    W_full[:16, :49] = W_44
    # Rows 16-19 of V_full and W_full are already 0 for the first 49 columns
    
    # Fill in the 4x4*4x1 part (columns 49-64)
    U_full[:, 49:] = U_41
    V_full[:, 49:] = V_41
    W_full[:, 49:] = W_41
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def _build_tensor_entries(U, V, W):
    """Compute the tensor from factors and check against all-ones."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                tensor[i, j, k] = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
    return tensor

def _is_valid(U, V, W, tol=1e-9):
    """Check if the factors exactly decompose the all-ones tensor."""
    try:
        t = _build_tensor_entries(U, V, W)
        if np.allclose(t, np.ones((n, m, p)), atol=tol):
            return True
    except Exception:
        pass
    return False

def construct():
    start_time = time.time()
    
    # Build the structured rank-65 decomposition
    U, V, W = _build_strassen_4x4_4x5()
    
    # Verify it's valid
    if _is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-80 decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(NM)]
    V_sb = [[0] * rank_schoolbook for _ in range(MP)]
    W_sb = [[0] * rank_schoolbook for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
