import numpy as np
import time

def construct():
    """
    Computes a rank-minimal (or near-minimal) tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy:
    Decompose the problem into 16 independent 2x2 * 2x2 = 2x2 matrix multiplications.
    Use Strassen's algorithm (rank 7) for each 2x2 multiplication instead of the naive rank 8.
    Total rank = 16 * 7 = 112.
    
    The naive rank is 4*4*8 = 128.
    Score improvement: 128 / 112 = 1.1428...
    
    Interface:
    U: shape (16, 112) - indices (i, j) for A
    V: shape (32, 112) - indices (j, k) for B
    W: shape (32, 112) - indices (k, i) for C (trace convention W[k*4+i, r])
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = 4, 4, 8
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A * B
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 rank-1 terms for a single 2x2x2 block.
    # Each term is a tuple of (U_vec, V_vec, W_vec) where vectors are length 4 (flattened 2x2).
    # Indices in 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Helper to create a vector of length 4 with a single 1 or -1
    def e(idx, val=1):
        v = np.zeros(4)
        v[idx] = val
        return v
    
    # The 7 Strassen terms (U, V, W) for one 2x2x2 block
    # Term 1: (A00+A11)(B00+B11) -> contributes to C00(+1), C11(-1)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: C00=+1, C11=-1 -> [1, 0, 0, -1]
    strassen_terms = [
        (
            np.array([1, 0, 0, 1]),  # U: A00+A11
            np.array([1, 0, 0, 1]),  # V: B00+B11
            np.array([1, 0, 0, -1])  # W: C00 - C11
        ),
        (
            np.array([0, 0, 1, 1]),  # U: A10+A11
            np.array([1, 0, 0, 0]),  # V: B00
            np.array([0, 0, 1, 0])   # W: C10
        ),
        (
            np.array([1, 0, 0, 0]),  # U: A00
            np.array([0, 1, 0, -1]), # V: B01-B11
            np.array([0, 1, 0, 0])   # W: C01
        ),
        (
            np.array([0, 0, 0, 1]),  # U: A11
            np.array([0, 0, 1, -1]), # V: B10-B00
            np.array([1, 0, 0, 0])   # W: C00
        ),
        (
            np.array([1, 1, 0, 0]),  # U: A00+A01
            np.array([0, 0, 0, 1]),  # V: B11
            np.array([-1, 0, 0, 0])  # W: -C00
        ),
        (
            np.array([0, 0, 1, -1]), # U: A10-A00
            np.array([1, 1, 0, 0]),  # V: B00+B01
            np.array([0, 0, 0, 1])   # W: C11
        ),
        (
            np.array([0, 1, 0, -1]), # U: A01-A11
            np.array([0, 0, 1, 1]),  # V: B10+B11
            np.array([0, 0, 0, -1])  # W: -C11
        ),
    ]
    
    # Verify Strassen correctness locally (optional but good for debugging)
    # For random 2x2 A, B, check if sum(U*V*W) == A*B
    # We trust the standard formulation.
    
    rank = 112
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We have 4 row-blocks of A (2x2) and 4 col-blocks of B (2x2) for each of the 4 "half" columns of B?
    # No, B is 4x8.
    # A is 4x4.
    # C is 4x8.
    
    # Let's map the global indices to local 2x2 block indices.
    # A[i, j]: i in 0..3, j in 0..3.
    # Block A[r, c] corresponds to rows 2r..2r+1, cols 2c..2c+1.
    # Local indices in A_block: 00->0, 01->1, 10->2, 11->3.
    
    # B[j, k]: j in 0..3, k in 0..7.
    # Block B[r, c] corresponds to rows 2r..2r+1, cols 2c..2c+1.
    # Here r is row-block (0..1), c is col-block (0..3).
    # Local indices in B_block: 00->0, 01->1, 10->2, 11->3.
    
    # C[i, k]: i in 0..3, k in 0..7.
    # Block C[r, c] corresponds to rows 2r..2r+1, cols 2c..2c+1.
    # Here r is row-block (0..1), c is col-block (0..3).
    # Local indices in C_block: 00->0, 01->1, 10->2, 11->3.
    # W index convention: W[k*n + i, rank_idx].
    # Global C index for block (r, c) local (li, lk) is:
    # i = 2*r + li_row (0 or 1)
    # k = 2*c + lk_col (0 or 1)
    # Wait, local indices 0,1,2,3 map to:
    # 0: (0,0) -> i_off=0, k_off=0
    # 1: (0,1) -> i_off=0, k_off=1
    # 2: (1,0) -> i_off=1, k_off=0
    # 3: (1,1) -> i_off=1, k_off=1
    
    t = 0
    
    # Iterate over all 16 block products: A_row_block, A_col_block, B_row_block, B_col_block
    # A is 4x4 -> 2x2 blocks: A_br in {0,1}, A_bc in {0,1}
    # B is 4x8 -> 2x2 blocks: B_br in {0,1}, B_bc in {0,1,2,3}
    # The product A[A_br, A_bc] * B[B_br, B_bc] contributes to C[A_br, B_bc]
    # Note: For the product to be valid, the column index of A must match the row index of B.
    # A has cols 0..3. B has rows 0..3.
    # A_block(A_br, A_bc) uses cols 2*A_bc .. 2*A_bc+1.
    # B_block(B_br, B_bc) uses rows 2*B_br .. 2*B_br+1.
    # So we need A_bc == B_br.
    
    # So we iterate:
    # A_row_block (ar) in 0..1
    # A_col_block (ac) in 0..1
    # B_row_block (br) must equal ac. So br = ac.
    # B_col_block (bc) in 0..3
    
    for ar in range(2):
        for ac in range(2):
            br = ac
            for bc in range(4):
                # This block product contributes to C_block (ar, bc)
                
                # Add 7 Strassen terms
                for u_vec, v_vec, w_vec in strassen_terms:
                    # Fill U
                    # A_block (ar, ac) local indices 0..3 map to global A indices:
                    # local 0: i=2*ar, j=2*ac
                    # local 1: i=2*ar, j=2*ac+1
                    # local 2: i=2*ar+1, j=2*ac
                    # local 3: i=2*ar+1, j=2*ac+1
                    
                    for local_idx in range(4):
                        val = u_vec[local_idx]
                        if val != 0:
                            # Map local 2x2 index to (i, j)
                            li = local_idx // 2
                            lj = local_idx % 2
                            gi = 2 * ar + li
                            gj = 2 * ac + lj
                            U[gi * m + gj, t] = val
                    
                    # Fill V
                    # B_block (br, bc) local indices 0..3 map to global B indices:
                    # local 0: j=2*br, k=2*bc
                    # local 1: j=2*br, k=2*bc+1
                    # local 2: j=2*br+1, k=2*bc
                    # local 3: j=2*br+1, k=2*bc+1
                    
                    for local_idx in range(4):
                        val = v_vec[local_idx]
                        if val != 0:
                            li = local_idx // 2
                            lk = local_idx % 2
                            gj = 2 * br + li
                            gk = 2 * bc + lk
                            V[gj * p + gk, t] = val
                    
                    # Fill W
                    # C_block (ar, bc) local indices 0..3 map to global C indices:
                    # local 0: i=2*ar, k=2*bc
                    # local 1: i=2*ar, k=2*bc+1
                    # local 2: i=2*ar+1, k=2*bc
                    # local 3: i=2*ar+1, k=2*bc+1
                    # W index: k*n + i
                    
                    for local_idx in range(4):
                        val = w_vec[local_idx]
                        if val != 0:
                            li = local_idx // 2
                            lk = local_idx % 2
                            gi = 2 * ar + li
                            gk = 2 * bc + lk
                            W[gk * n + gi, t] = val
                    
                    t += 1
                    
                    # Time check
                    if time.time() > deadline:
                        # If we run out of time, we have a partial result which is invalid.
                        # However, 112 terms is very fast to construct.
                        # We should ensure we finish.
                        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Quick verification
    n, m, p = 4, 4, 8
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Random test
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-way contraction.
    # Efficient way:
    # For each r, add U_r * V_r^T * W_r^T to C?
    # C = sum_r outer(U[:, r], V[:, r]) * W[:, r] ? No.
    # C[i, k] = sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    C_computed = np.zeros((n, p))
    rank = U.shape[1]
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n) # W[k, i]
        
        # Product is sum_j u_r[i, j] v_r[j, k] w_r[k, i]
        # This is trace( u_r * v_r * w_r )? No.
        # Let M = u_r @ v_r. M[i, k].
        # Then term is sum_k M[i, k] w_r[k, i].
        # This is element-wise product of M and w_r^T (since w_r is p x n, w_r^T is n x p).
        # So term = M * w_r.T
        # Sum over r.
        
        M = u_r @ v_r
        C_computed += M * w_r.T
        
    error = np.max(np.abs(C_expected - C_computed))
    print("Max Error:", error)
    if error > 1e-9:
        print("ERROR: Decomposition is invalid")
    else:
        print("OK: Decomposition is valid")
