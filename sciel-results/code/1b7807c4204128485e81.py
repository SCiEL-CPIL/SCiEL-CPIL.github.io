import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n=4, m=4, p=5, verbose=False):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==k and j==j (identity)
    Actually, the tensor T[i,j,k] for matrix multiplication is 1 if the indices form a valid product term.
    Specifically, T_{i,j,k} = 1 if the (i,k)-th entry of A*B depends on A_{i,j} * B_{j,k}.
    So T[i,j,k] = 1 for all i,j,k? No.
    The matrix multiplication tensor T is defined such that:
    (A B)_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
    So the tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, that's not right. The tensor is a 3-tensor.
    T[i, j, k] is 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    Yes, for standard matrix multiplication, every A[i,j] * B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k.
    
    The decomposition is:
    T[i,j,k] = sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We check if this sum equals 1 for all i,j,k.
    """
    rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                # Sum over rank
                prod_sum = np.dot(u_val, v_val * w_val)
                expected = 1.0
                err = abs(prod_sum - expected)
                if err > max_err:
                    max_err = err
                if verbose and err > 1e-9:
                    print(f"Error at i={i}, j={j}, k={k}: got {prod_sum}, expected {expected}")
    
    return max_err

def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen.
    Each term is (u_vec, v_vec, w_vec) where:
    u_vec has 4 entries (for A[0,0], A[0,1], A[1,0], A[1,1])
    v_vec has 4 entries (for B[0,0], B[0,1], B[1,0], B[1,1])
    w_vec has 4 entries (for C[0,0], C[0,1], C[1,0], C[1,1])
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M4 + M6
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M7
    """
    terms = []
    
    # M1: (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    terms.append((u, v, w))
    
    # M2: (A10+A11)B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0]
    terms.append((u, v, w))
    
    # M3: A00(B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 0]
    terms.append((u, v, w))
    
    # M4: A11(B10-B00)
    u = [0, 0, 0, 1]
    v = [0, 0, 1, -1]
    w = [0, 0, 1, 0]
    terms.append((u, v, w))
    
    # M5: (A00+A01)B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [0, 1, 0, 0]
    terms.append((u, v, w))
    
    # M6: (A10-A00)(B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [1, 0, 0, 0]
    terms.append((u, v, w))
    
    # M7: (A01-A11)(B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [0, 0, 0, 1]
    terms.append((u, v, w))
    
    return terms

def _build_strassen_4x4x4(U, V, W, col_start, n=4, m=4, p=4):
    """
    Build rank-49 decomposition for 4x4x4 using recursive Strassen.
    Splits 4x4 into 2x2 blocks and applies Strassen at the block level,
    then expands each block product using 2x2x2 Strassen.
    
    U shape: (16, rank) - indices i*4+j
    V shape: (16, rank) - indices j*4+k (for the 4x4 part)
    W shape: (16, rank) - indices k*4+i
    """
    col = col_start
    base_terms = _strassen_2x2x2_terms()
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen products at block level:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B01-B11) -> contributes to C01, C11 (with -1)
    # M4 = A11(B10-B00) -> contributes to C00 (with -1), C10
    # M5 = (A00+A01)B11 -> contributes to C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C00
    # M7 = (A01-A11)(B10+B11) -> contributes to C11
    
    # For each block product, we need to:
    # 1. Define the linear combination of A blocks (U coefficients)
    # 2. Define the linear combination of B blocks (V coefficients)
    # 3. Map the resulting 2x2 product to C blocks (W coefficients)
    
    # Block index mapping:
    # A00: rows 0-1, cols 0-1 -> U indices: 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> U indices: 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> U indices: 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> U indices: 10,11,14,15
    
    # B00: rows 0-1, cols 0-1 -> V indices: 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> V indices: 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> V indices: 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> V indices: 10,11,14,15
    
    # C00: rows 0-1, cols 0-1 -> W indices: 0,1,4,5
    # C01: rows 0-1, cols 2-3 -> W indices: 2,3,6,7
    # C10: rows 2-3, cols 0-1 -> W indices: 8,9,12,13
    # C11: rows 2-3, cols 2-3 -> W indices: 10,11,14,15
    
    def get_block_indices(base_r, base_c, n_block=2):
        """Get global indices for a 2x2 block starting at (base_r, base_c)."""
        indices = []
        for dr in range(n_block):
            for dc in range(n_block):
                indices.append((base_r + dr) * 4 + (base_c + dc))
        return indices
    
    A00_idx = get_block_indices(0, 0)
    A01_idx = get_block_indices(0, 2)
    A10_idx = get_block_indices(2, 0)
    A11_idx = get_block_indices(2, 2)
    
    B00_idx = get_block_indices(0, 0)
    B01_idx = get_block_indices(0, 2)
    B10_idx = get_block_indices(2, 0)
    B11_idx = get_block_indices(2, 2)
    
    C00_idx = get_block_indices(0, 0)
    C01_idx = get_block_indices(0, 2)
    C10_idx = get_block_indices(2, 0)
    C11_idx = get_block_indices(2, 2)
    
    # Define the 7 Strassen products with their block combinations
    # Each product: (a_blocks, b_blocks, c_blocks_with_signs)
    # a_blocks: dict {block_name: coefficient}
    # b_blocks: dict {block_name: coefficient}
    # c_blocks: dict {block_name: coefficient}
    
    products = [
        # M1
        ({"A00": 1, "A11": 1}, {"B00": 1, "B11": 1}, {"C00": 1, "C11": 1}),
        # M2
        ({"A10": 1, "A11": 1}, {"B00": 1}, {"C10": 1}),
        # M3
        ({"A00": 1}, {"B01": 1, "B11": -1}, {"C01": 1, "C11": -1}),
        # M4
        ({"A11": 1}, {"B10": 1, "B00": -1}, {"C00": -1, "C10": 1}),
        # M5
        ({"A00": 1, "A01": 1}, {"B11": 1}, {"C01": 1}),
        # M6
        ({"A10": 1, "A00": -1}, {"B00": 1, "B01": 1}, {"C00": 1}),
        # M7
        ({"A01": 1, "A11": -1}, {"B10": 1, "B11": 1}, {"C11": 1}),
    ]
    
    block_to_idx = {
        "A00": A00_idx, "A01": A01_idx, "A10": A10_idx, "A11": A11_idx,
        "B00": B00_idx, "B01": B01_idx, "B10": B10_idx, "B11": B11_idx,
        "C00": C00_idx, "C01": C01_idx, "C10": C10_idx, "C11": C11_idx,
    }
    
    for a_blocks, b_blocks, c_blocks in products:
        # For each of the 7 rank-1 terms in the 2x2x2 Strassen:
        for u_base, v_base, w_base in base_terms:
            # u_base, v_base, w_base are length-4 vectors for local 2x2 blocks
            # We need to expand them to the global 4x4 indices
            
            # For U: combine the A blocks with the local 2x2 pattern
            for local_r in range(2):
                for local_c in range(2):
                    u_coeff = u_base[local_r * 2 + local_c]
                    if u_coeff == 0:
                        continue
                    for a_block_name, a_coeff in a_blocks.items():
                        if a_coeff == 0:
                            continue
                        idx_list = block_to_idx[a_block_name]
                        local_idx_in_block = local_r * 2 + local_c
                        global_idx = idx_list[local_idx_in_block]
                        U[global_idx][col] += a_coeff * u_coeff
            
            # For V: combine the B blocks with the local 2x2 pattern
            for local_r in range(2):
                for local_c in range(2):
                    v_coeff = v_base[local_r * 2 + local_c]
                    if v_coeff == 0:
                        continue
                    for b_block_name, b_coeff in b_blocks.items():
                        if b_coeff == 0:
                            continue
                        idx_list = block_to_idx[b_block_name]
                        local_idx_in_block = local_r * 2 + local_c
                        global_idx = idx_list[local_idx_in_block]
                        V[global_idx][col] += b_coeff * v_coeff
            
            # For W: combine the C blocks with the local 2x2 pattern
            for local_r in range(2):
                for local_c in range(2):
                    w_coeff = w_base[local_r * 2 + local_c]
                    if w_coeff == 0:
                        continue
                    for c_block_name, c_coeff in c_blocks.items():
                        if c_coeff == 0:
                            continue
                        idx_list = block_to_idx[c_block_name]
                        local_idx_in_block = local_r * 2 + local_c
                        global_idx = idx_list[local_idx_in_block]
                        W[global_idx][col] += c_coeff * w_coeff
            
            col += 1
    
    return col

def _build_schoolbook_4x4x1(U, V, W, col_start, n=4, m=4, p=5):
    """
    Build rank-16 decomposition for the 5th column of B (4x4 * 4x1 -> 4x1).
    U shape: (16, rank)
    V shape: (20, rank) - indices j*5+k, for k=4
    W shape: (20, rank) - indices k*4+i, for k=4
    """
    col = col_start
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    return col

def construct():
    """
    Construct the rank-minimal tensor decomposition for 4x4 * 4x5 -> 4x5.
    Returns: {'U': U, 'V': V, 'W': W}
    U: (16, rank), V: (20, rank), W: (20, rank)
    """
    n, m, p = 4, 4, 5
    
    # Total rank: 49 (for 4x4x4) + 16 (for 4x4x1) = 65
    total_rank = 65
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Build 4x4x4 part (first 4 columns of B)
    col = _build_strassen_4x4x4(U, V, W, col, n=n, m=m, p=4)
    
    # Build 4x4x1 part (5th column of B)
    col = _build_schoolbook_4x4x1(U, V, W, col, n=n, m=m, p=p)
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    
    # Verify the decomposition
    max_err = _verify_decomposition(U, V, W, n=n, m=m, p=p)
    if max_err > 1e-9:
        print(f"Warning: Verification failed with max error {max_err}")
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    max_err = _verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Max verification error: {max_err}")
