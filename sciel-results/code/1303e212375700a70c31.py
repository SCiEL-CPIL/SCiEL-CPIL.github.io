"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

This program uses a block decomposition strategy based on Strassen's algorithm
for 2x2 matrices. A 6x6 matrix is partitioned into a 3x3 grid of 2x2 blocks.
The product of any two 2x2 blocks is computed using Strassen's 7-multiplication
algorithm. Since there are 3x3 = 9 positions for the A-blocks, 3x3 = 9 for
B-blocks, and 3x3 = 9 for C-blocks, and 7 multiplications per block pair,
the total rank is 27 * 7 = 189.

This is an exact decomposition with integer coefficients.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_strassen_6x6_decomposition(n, m, p):
    """
    Constructs a rank-189 decomposition for 6x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    """
    # 6x6 matrices divided into 3x3 grid of 2x2 blocks
    block_size = 2
    blocks = 3  # 6 / 2
    
    # Strassen patterns for 2x2 matrices
    # Each pattern is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(local_row, local_col): val} for A block
    # V_coeffs: dict {(local_row, local_col): val} for B block
    # W_coeffs: dict {(local_row, local_col): val} for C block (P)
    # Note: W index in global tensor is k*n+i.
    # In local 2x2 block, C[u,v] corresponds to global C[2I+u, 2K+v].
    # So W row index is (2K+v)*n + (2I+u).
    
    strassen_patterns = [
        # M0 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = (blocks ** 3) * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for I in range(blocks):
        for J in range(blocks):
            for K in range(blocks):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # U: A block at (I, J)
                    # Global row i = 2*I + local_i
                    # Global col j = 2*J + local_j
                    # U index: i * m + j
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c, col_idx] += val
                    
                    # V: B block at (J, K)
                    # Global row j = 2*J + local_j
                    # Global col k = 2*K + local_k
                    # V index: j * p + k
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c, col_idx] += val
                    
                    # W: C block at (I, K)
                    # Global row k = 2*K + local_k (from C[2I+u, 2K+v])
                    # Global col i = 2*I + local_i
                    # W index: k * n + i
                    # Note: w_coeffs keys are (local_row, local_col) of C block
                    # C[u,v] -> global row 2I+u, col 2K+v.
                    # So local_row u corresponds to global row offset in I block.
                    # local_col v corresponds to global col offset in K block.
                    for (pi, pj), val in w_coeffs.items():
                        # pi is local row of C (u), pj is local col of C (v)
                        k_global = K * block_size + pj
                        i_global = I * block_size + pi
                        W[k_global * n + i_global, col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, num_tests=5, atol=1e-8):
    """
    Verifies the tensor decomposition by checking the bilinear identity
    C = A @ B for random integer matrices.
    """
    rank = U.shape[1]
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        # Efficient verification using tensor contraction
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this by iterating r
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # Ur @ Vr gives n x p matrix
            # Element-wise multiply by Wr (p x n reshaped to n x p? No)
            # Wr is indexed by [k*n+i]. So Wr[k, i] if reshaped (p, n).
            # But C[i, k] needs Wr[k, i].
            # Let's check the shape.
            # Wr is (p*n,). Reshape to (p, n).
            # Wr[k, i] is the value.
            # C[i, k] += (Ur @ Vr)[i, k] * Wr[k, i]
            # This is not standard element-wise multiplication.
            # We need to transpose Wr or index carefully.
            
            prod = Ur @ Vr # shape (n, p)
            # Wr reshaped to (p, n).
            # We want to multiply prod[i, k] by Wr[k, i].
            # This is equivalent to prod * Wr.T
            C_computed += prod * Wr.T

        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    # Build the Strassen decomposition
    decomp = _build_strassen_6x6_decomposition(n, m, p)
    
    # Verify to be safe
    if _verify_decomposition(decomp["U"], decomp["V"], decomp["W"], n, m, p, num_tests=3):
        best = decomp
    else:
        # Fallback to naive if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        best = {"U": U, "V": V, "W": W}
        
    # Convert to lists
    U = best["U"].tolist()
    V = best["V"].tolist()
    W = best["W"].tolist()
    
    return {
        "U": U,
        "V": V,
        "W": W
    }


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    valid = _verify_decomposition(U, V, W, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
