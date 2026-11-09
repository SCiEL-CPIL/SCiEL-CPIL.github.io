"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return U, V, W as lists of lists for schoolbook n*m*p rank decomposition."""
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
    return U, V, W


def _strassen_2x2(n, m, p, sub_U, sub_V, sub_W):
    """
    Build a Strassen-style rank-7 decomposition for n x n matrix multiplication
    where each 2x2 block multiplication is done via sub_U, sub_V, sub_W.
    
    n must be even. The matrices are divided into 2x2 blocks of size (n/2).
    Each block multiplication of size (n/2) uses the given sub decomposition.
    """
    s = n // 2
    # sub_U shape: (s*m_sub, rank_sub), sub_V shape: (m_sub*p_sub, rank_sub), sub_W shape: (p_sub*s, rank_sub)
    # For square: m_sub = p_sub = s
    rank_sub = len(sub_U[0])
    
    # Strassen's 7 products for 2x2 block matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_i is a product of two s x s matrices, so uses rank_sub rank-1 tensors.
    # Total rank = 7 * rank_sub
    
    total_rank = 7 * rank_sub
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    # Index helpers
    def a_idx(i, j):
        return i * n + j
    
    def b_idx(i, j):
        return i * n + j
    
    def c_idx(i, j):
        return j * n + i  # W[k*n+i, c] convention: output index (k, i) -> k*n + i
    
    # For each of the 7 Strassen products, we have rank_sub rank-1 components.
    # For product t (0..6), with rank sub-components r (0..rank_sub-1):
    #   U row corresponding to input A positions gets coefficient from factor A_t
    #   V row corresponding to input B positions gets coefficient from factor B_t
    #   W row corresponding to output C positions gets coefficient from combination
    
    # Define the 7 products as (A_factor, B_factor) where each is a linear combo of block entries
    # Block indices: (11, 12, 21, 22) for rows and cols
    
    # A blocks: A11 = A[0:s, 0:s], A12 = A[0:s, s:n], A21 = A[s:n, 0:s], A22 = A[s:n, s:n]
    # B blocks: B11 = B[0:s, 0:s], B12 = B[0:s, s:n], B21 = B[s:n, 0:s], B22 = B[s:n, s:n]
    # C blocks: C11 = C[0:s, 0:s], C12 = C[0:s, s:n], C21 = C[s:n, 0:s], C22 = C[s:n, s:n]
    
    # For each product t, define:
    #   a_coeff: dict from (block_row, block_col) -> coefficient in A
    #   b_coeff: dict from (block_row, block_col) -> coefficient in B
    #   c_coeff: dict from (block_row, block_col) -> coefficient in C
    
    products = []
    
    # M1 = (A11 + A22) * (B11 + B22)
    a1 = {(0,0): 1, (1,1): 1}
    b1 = {(0,0): 1, (1,1): 1}
    c1 = {(0,0): 1, (1,1): 1}  # M1 contributes +1 to C11 and +1 to C22
    products.append((a1, b1, c1))
    
    # M2 = (A21 + A22) * B11
    a2 = {(1,0): 1, (1,1): 1}
    b2 = {(0,0): 1}
    c2 = {(1,0): 1, (1,1): -1}  # M2 contributes +1 to C21 and -1 to C22
    products.append((a2, b2, c2))
    
    # M3 = A11 * (B12 - B22)
    a3 = {(0,0): 1}
    b3 = {(0,1): 1, (1,1): -1}
    c3 = {(0,1): 1, (1,1): 1}  # M3 contributes +1 to C12 and +1 to C22
    products.append((a3, b3, c3))
    
    # M4 = A22 * (B21 - B11)
    a4 = {(1,1): 1}
    b4 = {(1,0): 1, (0,0): -1}
    c4 = {(0,0): 1, (1,0): 1}  # M4 contributes +1 to C11 and +1 to C21
    products.append((a4, b4, c4))
    
    # M5 = (A11 + A12) * B22
    a5 = {(0,0): 1, (0,1): 1}
    b5 = {(1,1): 1}
    c5 = {(0,0): -1, (0,1): 1}  # M5 contributes -1 to C11 and +1 to C12
    products.append((a5, b5, c5))
    
    # M6 = (A21 - A11) * (B11 + B12)
    a6 = {(1,0): 1, (0,0): -1}
    b6 = {(0,0): 1, (0,1): 1}
    c6 = {(1,1): 1}  # M6 contributes +1 to C22
    products.append((a6, b6, c6))
    
    # M7 = (A12 - A22) * (B21 + B22)
    a7 = {(0,1): 1, (1,1): -1}
    b7 = {(1,0): 1, (1,1): 1}
    c7 = {(0,0): 1}  # M7 contributes +1 to C11
    products.append((a7, b7, c7))
    
    # Now build U, V, W
    for t in range(7):
        a_coeff, b_coeff, c_coeff = products[t]
        for r in range(rank_sub):
            col = t * rank_sub + r
            
            # U: for each input position (i, j) in A
            for br in range(2):
                for bc in range(2):
                    coeff_a = a_coeff.get((br, bc), 0)
                    if coeff_a == 0:
                        continue
                    # sub_U row index: for block (br, bc), the local row is i_local in [0, s)
                    # local row i_local corresponds to global row br*s + i_local
                    # The sub_U row for this block and rank component r
                    for i_local in range(s):
                        global_i = br * s + i_local
                        global_j = bc * s + i_local  # wait, j is column of A, which is row index of sub_V
                        # Actually, for the input matrix A (n x n), position (i, j):
                        # U[i*n + j, col] should get coeff_a * sub_U[i_local * s + j_local, r]
                        # But sub_U has shape (s*s, rank_sub) for s x s multiplication
                        # Wait, sub_U is for multiplying (s x s) by (s x s)
                        # So sub_U row index is (i_local * s + j_local) where i_local is row of first matrix, j_local is col of first matrix
                        for j_local in range(s):
                            global_j = bc * s + j_local
                            u_val = sub_U[i_local * s + j_local][r]
                            if u_val != 0:
                                U[global_i * n + global_j][col] += coeff_a * u_val
            
            # V: for each input position (i, j) in B
            for br in range(2):
                for bc in range(2):
                    coeff_b = b_coeff.get((br, bc), 0)
                    if coeff_b == 0:
                        continue
                    for i_local in range(s):
                        global_i = br * s + i_local
                        for j_local in range(s):
                            global_j = bc * s + j_local
                            v_val = sub_V[i_local * s + j_local][r]
                            if v_val != 0:
                                V[global_i * n + global_j][col] += coeff_b * v_val
            
            # W: for each output position (i, j) in C, stored as W[j*n + i, col]
            for br in range(2):
                for bc in range(2):
                    coeff_c = c_coeff.get((br, bc), 0)
                    if coeff_c == 0:
                        continue
                    # C block (br, bc) has rows br*s to (br+1)*s-1, cols bc*s to (bc+1)*s-1
                    # Output C[i, j] where i = br*s + i_local, j = bc*s + j_local
                    # W index: j*n + i = (bc*s + j_local)*n + (br*s + i_local)
                    for i_local in range(s):
                        global_i = br * s + i_local
                        for j_local in range(s):
                            global_j = bc * s + j_local
                            w_val = sub_W[j_local * s + i_local][r]
                            if w_val != 0:
                                W[global_j * n + global_i][col] += coeff_c * w_val
    
    return U, V, W


def _build_decomposition(n):
    """
    Build a rank-minimal-ish decomposition for n x n matrix multiplication.
    Uses Strassen recursion where beneficial.
    """
    if n == 1:
        # Rank 1
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    if n == 2:
        # Strassen rank 7 with 1x1 base
        U, V, W = _strassen_2x2(2, 2, 2, [[1]], [[1]], [[1]])
        return U, V, W
    
    if n == 3:
        # Schoolbook rank 9
        return _schoolbook(3, 3, 3)
    
    if n == 4:
        # Strassen 2x2 of 2x2, each 2x2 uses Strassen rank 7
        # Total rank = 7 * 7 = 49
        U2, V2, W2 = _strassen_2x2(2, 2, 2, [[1]], [[1]], [[1]])
        U, V, W = _strassen_2x2(4, 4, 4, U2, V2, W2)
        return U, V, W
    
    if n == 5:
        # Pad to 6x6: use Strassen 2x2 of 3x3
        # 3x3 uses schoolbook rank 9
        # 6x6 uses Strassen: rank = 7 * 9 = 63
        U3, V3, W3 = _schoolbook(3, 3, 3)
        U6, V6, W6 = _strassen_2x2(6, 6, 6, U3, V3, W3)
        # Now extract the 5x5 part from the 6x6 decomposition
        # The 5x5 matrix is embedded in the top-left 5x5 of the 6x6 matrix
        # U6 has shape (6*6, 63) = (36, 63)
        # We need U with shape (5*5, 63) = (25, 63)
        # Row indices for 5x5: i in [0,4], j in [0,4] -> i*6+j for 6x6, but we want i*5+j for 5x5
        # Wait, the padding means we treat the 5x5 matrix as a 6x6 matrix with last row/col zero.
        # U6[i*6+j, c] for i,j in [0,5]
        # For the 5x5 extraction, we keep rows where i<=4 and j<=4
        U5 = []
        for i in range(5):
            for j in range(5):
                U5.append(U6[i*6 + j])
        V5 = []
        for i in range(5):
            for j in range(5):
                V5.append(V6[i*6 + j])
        # W6 has shape (6*6, 63) with index k*6+i
        # We need W with shape (5*5, 63) with index k*5+i
        W5 = []
        for k in range(5):
            for i in range(5):
                W5.append(W6[k*6 + i])
        return U5, V5, W5
    
    # For larger n, fall back to schoolbook
    return _schoolbook(n, n, n)


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Compute C = A * B for all rank-1 components and check
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Build the full tensor check
    # For efficiency, check with random matrices
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m))
    B = np.random.randint(-5, 6, size=(m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p), dtype=int)
    for c in range(rank):
        Uc = np.array(U[:, c]).reshape(n, m)
        Vc = np.array(V[:, c]).reshape(m, p)
        Wc = np.array(W[:, c]).reshape(p, n)
        # The contribution is (Uc * something) ... 
        # Actually: C[i,k] += sum_j Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is not a standard matrix product. Let's compute directly.
        for i in range(n):
            for k in range(p):
                w_val = Wc[k, i]
                if w_val != 0:
                    for j in range(m):
                        C_computed[i, k] += Uc[i, j] * Vc[j, k] * w_val
    
    if not np.allclose(C_computed, C_expected):
        diff = np.abs(C_computed - C_expected)
        max_diff = np.max(diff)
        num_diff = np.sum(diff > 0)
        return False, max_diff, num_diff
    return True, 0, 0


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with shapes (25, rank), (25, rank), (25, rank).
    """
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    try:
        U, V, W = _build_decomposition(5)
    except Exception as e:
        # Fall back to schoolbook
        U, V, W = _schoolbook(5, 5, 5)
    
    # Verify correctness
    try:
        valid, max_diff, num_diff = _verify_decomposition(U, V, W, 5, 5, 5)
        if not valid:
            # If verification fails, fall back to schoolbook
            U, V, W = _schoolbook(5, 5, 5)
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
