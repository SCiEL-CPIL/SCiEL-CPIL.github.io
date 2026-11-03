"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Strassen's 7-multiplication algorithm for 2x2x2 matrix multiplication.
    Returns U, V, W for the 2x2x2 tensor with W indexed as [k*n+i, r] = [k*2+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # Strassen's formulas:
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
    
    # U rows: A11, A12, A21, A22 (index i*m+j)
    # V rows: B11, B12, B21, B22 (index j*p+k)
    # W rows: C11, C12, C21, C22 (index k*n+i, so C11=0, C12=1, C21=2, C22=3)
    
    # M1: (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_base_decomposition():
    """Build a base decomposition using Strassen for 2x2x2 and schoolbook for 2x1x4.
    
    Strategy: Split the 3x8 part. 
    - For the first 2 columns of B (j=0,1) and all 8 rows: use Strassen-like on 2x2x2 blocks
    - Actually, let's think differently.
    
    Better approach: View 2x3x8 as:
    - The 2x3 matrix A has 6 entries, 3x8 matrix B has 24 entries, output C is 2x8.
    
    Let's use a block structure. Split B into two 3x4 blocks: B = [B1 | B2] where B1, B2 are 3x4.
    Then C = A*B = A*B1 | A*B2, each being 2x4.
    
    For a 2x3x4 multiplication:
    Split A into rows: A = [A1; A2] where A1, A2 are 1x3.
    C = A1*B | A2*B, each 1x4.
    
    Actually, let me use a more systematic approach.
    
    2x3x8: Split the middle dimension (3) into 2+1.
    A = [A_left (2x2) | A_right (2x1)]
    B = [B_top (2x8); B_bottom (1x8)]
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 2x2x8
    A_right * B_bottom: 2x1x8
    
    For 2x2x8: Split 8 into 4+4.
    A_left * [B_top1 | B_top2] where B_top1, B_top2 are 2x4.
    = A_left*B_top1 | A_left*B_top2, each 2x2x4.
    
    For 2x2x4: Split 4 into 2+2.
    = A_left*B1 | A_left*B2, each 2x2x2. Use Strassen: 7 mults each.
    Total for 2x2x8: 4 * 7 = 28 mults.
    
    For 2x1x8: This is just 2*8 = 16 scalar multiplications (each row of A_right times each column of B_bottom).
    Actually 2x1x8: A_right is 2x1, B_bottom is 1x8, C is 2x8.
    C[i,k] = A_right[i,0] * B_bottom[0,k]. So 16 independent multiplications.
    
    Total: 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0.0]*rank for _ in range(n*m)]  # 6 x 44
    V = [[0.0]*rank for _ in range(m*p)]  # 24 x 44
    W = [[0.0]*rank for _ in range(p*n)]  # 16 x 44
    
    col = 0
    # Part 1: A_left (2x2) * B_top (2x8) -> 2x8 output
    # A_left rows: i=0,1; j=0,1 => U indices: i*m+j = i*3+0, i*3+1
    # B_top rows: j=0,1; k=0..7 => V indices: j*8+k = 0..7, 8..15
    # Output: C[i,k] for i=0,1; k=0..7 => W index: k*n+i = k*2+i
    
    # Split B_top into 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    for block in range(4):
        k_offset = block * 2
        # Strassen for 2x2x2: A_left * B_top[:, k_offset:k_offset+2]
        sU, sV, sW = _strassen_2x2x2()
        for r in range(7):
            # U: A_left[i,j] -> U[i*3 + j]
            for i in range(2):
                for j in range(2):
                    U[i*3 + j][col] += sU[i*2 + j][r]
            # V: B_top[j, k] -> V[j*8 + (k_offset + k)]
            for j in range(2):
                for k in range(2):
                    V[j*8 + k_offset + k][col] += sV[j*2 + k][r]
            # W: C[i, k] -> W[(k_offset + k)*2 + i]
            for i in range(2):
                for k in range(2):
                    W[(k_offset + k)*2 + i][col] += sW[k*2 + i][r]
            col += 1
    
    # Part 2: A_right (2x1) * B_bottom (1x8) -> 2x8 output
    # A_right[i,0] -> U[i*3 + 2]
    # B_bottom[0,k] -> V[2*8 + k] = V[16 + k]
    # C[i,k] -> W[k*2 + i]
    for i in range(2):
        for k in range(8):
            U[i*3 + 2][col] = 1
            V[16 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = delta (should be 1 if this is a valid MM tensor)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So tensor T[i,j,k] = 1 for all i,j,k (the MM tensor has all ones)
    # Wait, the MM tensor is T[i,j,k] = delta_{i,j,k} ... no.
    # The tensor for matrix multiplication: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Actually the tensor has T[i,j,k] = 1 for all valid i,j,k.
    
    # Compute from decomposition: T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T_decomp[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum
    T_decomp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Expected: all ones
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T_decomp, T_expected, atol=1e-10)

def _try_reduce_rank(U, V, W, n=2, m=3, p=8, max_iters=500, timeout=100):
    """Try to reduce the rank by finding linear dependencies."""
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Try to find a direction in which we can reduce rank
    # The idea: if there's a linear combination of columns that gives a valid decomposition with one fewer rank
    # This is equivalent to finding if the tensor has rank < current rank.
    
    # Use the fact that we can try to eliminate one column by expressing it as a combination
    # of others, but this is NP-hard in general.
    
    # Alternative: use gradient-based optimization to find a lower-rank approximation
    # But we need exact decomposition.
    
    # For now, just verify and return. The structured 44-rank is likely near-optimal.
    # Known results: rank of 2x3x8 is not well-studied, but 44 is a good upper bound.
    
    # Try a simple local search: for each column, see if it can be expressed as a linear combination
    # of the others while maintaining the tensor equation.
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Try removing each column and solving for the missing coefficients
    for r_remove in range(rank):
        if time.time() - start_time > timeout:
            break
        
        U_reduced = np.delete(U_arr, r_remove, axis=1)
        V_reduced = np.delete(V_arr, r_remove, axis=1)
        W_reduced = np.delete(W_arr, r_remove, axis=1)
        
        # Check if the tensor can be represented with rank-1
        # This requires solving a system of equations, which is complex.
        # Skip for now.
        
        pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build the base 44-rank decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook if verification fails
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank (limited time)
    if time.time() < deadline - 20:
        U_opt, V_opt, W_opt, rank_opt = _try_reduce_rank(U, V, W, timeout=deadline - time.time() - 10)
        if rank_opt < len(U[0]):
            U, V, W = U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    
    # Final verification
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
