"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Construct Strassen-based rank-49 decomposition for 4x4x4
    # Then extend to 4x4x5 with 4 additional terms = rank 53
    
    rank_444 = 49
    additional_rank = 4
    total_rank = rank_444 + additional_rank  # 53
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 53
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 53
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 53
    
    # Build Strassen 4x4x4 decomposition
    _build_strassen_4x4x4(U, V, W, rank_444)
    
    # Add 4 terms for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # Term r (r=0..3): contributes to all i for the j=r term
    for r in range(4):
        col = rank_444 + r
        for i in range(n):
            U[i * m + r][col] = 1      # A[i,r]
        V[r * p + 4][col] = 1          # B[r,4]
        for i in range(n):
            W[4 * n + i][col] = 1      # C[i,4]
    
    decomp = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_decomposition(decomp, n, m, p):
        return decomp
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)


def _build_strassen_4x4x4(U, V, W, rank_start=0):
    """
    Build Strassen 4x4x4 decomposition into U, V, W starting at column rank_start.
    
    4x4 matrices split into 2x2 blocks of 2x2 submatrices:
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    C = A*B
    
    Block-level Strassen:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B10-B00)
    P4 = A11*(B01-B11)
    P5 = (A00+A01)*B11
    P6 = (A01-A11)*(B00+B01)
    P7 = (A00-A10)*(B10+B11)
    
    Each Pi is a 2x2 matrix product, computed with Strassen (rank 7).
    Total: 7 * 7 = 49 terms.
    
    Block C:
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    """
    
    # Strassen 2x2x2 coefficients
    # For 2x2 matrices with indices:
    # A[i,j] -> row i*2+j in U
    # B[j,k] -> row j*2+k in V  
    # C[i,k] -> row k*2+i in W
    
    # Strassen P factors for 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    #   U: A00(0)+A11(3), V: B00(0)+B11(3), W: C00(0)+C11(3)
    # P2 = (A10+A11)B00
    #   U: A10(2)+A11(3), V: B00(0), W: C10(1)+C11(3)*(-1)
    # P3 = A00(B10-B00)
    #   U: A00(0), V: B10(2)-B00(0), W: C01(2)+C11(3)
    # P4 = A11(B01-B11)
    #   U: A11(3), V: B01(1)-B11(3), W: C00(0)+C10(1)
    # P5 = (A00+A01)B11
    #   U: A00(0)+A01(1), V: B11(3), W: C01(2)+C00(0)*(-1)
    # P6 = (A01-A11)(B00+B01)
    #   U: A01(1)-A11(3), V: B00(0)+B01(1), W: C11(3)
    # P7 = (A00-A10)(B10+B11)
    #   U: A00(0)-A10(2), V: B10(2)+B11(3), W: C00(0)
    
    strassen_2x2 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is list of (local_index, coefficient)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),    # P1
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (1, 1), (3, -1) ]),              # P2
        ([ (0, 1) ], [ (2, 1), (0, -1) ], [ (2, 1), (3, 1) ]),              # P3
        ([ (3, 1) ], [ (1, 1), (3, -1) ], [ (0, 1), (1, 1) ]),              # P4
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (2, 1), (0, -1) ]),              # P5
        ([ (1, 1), (3, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),              # P6
        ([ (0, 1), (2, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),              # P7
    ]
    
    # Block-level Strassen: 7 block products
    # Each block product Pi = Ai * Bj where Ai, Bj are 2x2 matrices
    # We need to map block indices to local 2x2 indices
    
    # Block layout for 4x4:
    # A[0:2, 0:2] = A00, A[0:2, 2:4] = A01
    # A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    # Similarly for B and C
    
    # For a 2x2 submatrix at block position (bi, bj) in a 4x4 matrix:
    # The local indices (li, lj) map to global index (bi*2+li, bj*2+lj)
    # In U (flattened A[i,j]): global row = (bi*2+li)*4 + (bj*2+lj)
    # In V (flattened B[j,k]): global row = (bi*2+lj)*5 + (bk*2+lk) -- wait, B is 4x5
    # Actually for the 4x4x4 part, B is 4x4, so global row = (bi*2+lj)*4 + (bk*2+lk)
    # But we're building for 4x4x5, so B is 4x5. For the first 4 columns, same as 4x4x4.
    # In W (flattened C[k,i]): global row = (bk*2+lk)*4 + (bi*2+li)
    
    # Block-level Strassen combinations:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B00)
    # P4 = A11*(B01-B11)
    # P5 = (A00+A01)*B11
    # P6 = (A01-A11)*(B00+B01)
    # P7 = (A00-A10)*(B10+B11)
    
    # For each block product, we need to combine the Strassen 2x2 factors
    # with the block-level coefficients
    
    # Block-level coefficient for each Pi in terms of (A_block, B_block):
    block_products = [
        # (A_block_coeffs, B_block_coeffs, C_block_result_coeffs)
        # A_block_coeffs: list of (block_idx, coeff) where block_idx in {(0,0),(0,1),(1,0),(1,1)}
        # B_block_coeffs: list of (block_idx, coeff)
        # C_block_result_coeffs: list of (block_idx, coeff) for the result
        ([ ((0,0), 1), ((1,1), 1) ], [ ((0,0), 1), ((1,1), 1) ], [ ((0,0), 1), ((1,1), 1) ]),  # P1
        ([ ((1,0), 1), ((1,1), 1) ], [ ((0,0), 1) ], [ ((1,0), 1), ((1,1), -1) ]),              # P2
        ([ ((0,0), 1) ], [ ((1,0), 1), ((0,0), -1) ], [ ((0,1), 1), ((1,1), 1) ]),              # P3
        ([ ((1,1), 1) ], [ ((0,1), 1), ((1,1), -1) ], [ ((0,0), 1), ((1,0), 1) ]),              # P4
        ([ ((0,0), 1), ((0,1), 1) ], [ ((1,1), 1) ], [ ((0,1), 1), ((0,0), -1) ]),              # P5
        ([ ((0,1), 1), ((1,1), -1) ], [ ((0,0), 1), ((0,1), 1) ], [ ((1,1), 1) ]),              # P6
        ([ ((0,0), 1), ((1,0), -1) ], [ ((1,0), 1), ((1,1), 1) ], [ ((0,0), 1) ]),              # P7
    ]
    
    # For each of the 7 block products, and for each of the 7 Strassen 2x2 factors within it,
    # we create one rank-1 term in the overall decomposition
    
    col = rank_start
    for bp_idx in range(7):
        a_block_coeffs, b_block_coeffs, c_block_coeffs = block_products[bp_idx]
        
        for s_idx in range(7):
            u_local, v_local, w_local = strassen_2x2[s_idx]
            
            # For each combination of block coefficients and local coefficients,
            # add to U, V, W
            
            for (ab, ac) in a_block_coeffs:
                for (li, lu) in u_local:
                    # Global A index: block (ab[0], ab[1]), local (li, lj)
                    # But u_local gives us the local row index in the 2x2 A submatrix
                    # Local index li means row li in the 2x2 block
                    # We need to figure out which column of A this is
                    # The Strassen factor specifies which entries of A are used
                    
                    # Actually, u_local gives (local_flat_idx, coeff) where local_flat_idx
                    # is the flat index in the 2x2 A block (0=A00, 1=A01, 2=A10, 3=A11)
                    # So local_flat_idx = li*2 + lj
                    
                    li2 = ab[0] * 2 + (u_local[0][0] // 2)  # This isn't right
                    # Let me redo: u_local is a list of (flat_idx, coeff)
                    # flat_idx in {0,1,2,3} corresponds to (row, col) in 2x2:
                    # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                    pass
            
            # Let me restructure: for each Strassen factor, we have specific local indices
            # For the U part: u_local is a list of (flat_idx, coeff)
            # flat_idx maps to local (r, c) = (flat_idx // 2, flat_idx % 2)
            # Global A row = ab[0]*2 + r, global A col = ab[1]*2 + c
            # Global U row = (ab[0]*2 + r) * 4 + (ab[1]*2 + c)
            
            for (uf, uc) in u_local:
                ur = uf // 2
                uc_local = uf % 2
                for (ab, ac) in a_block_coeffs:
                    global_row = (ab[0] * 2 + ur) * 4 + (ab[1] * 2 + uc_local)
                    U[global_row][col] += ac * uc
            
            for (vf, vc) in v_local:
                vr = vf // 2
                vc_local = vf % 2
                for (bb, bc) in b_block_coeffs:
                    # V is B[j,k], flattened as j*4+k for 4x4 (first 4 cols)
                    # Block (bb, bc) means B rows bb*2..bb*2+1, cols bc*2..bc*2+1
                    # Local (vr, vc_local) -> global row = bb*2+vr, global col = bc*2+vc_local
                    # V row = (bb*2+vr) * 4 + (bc*2+vc_local)  [for 4x4 part]
                    global_row = (bb[0] * 2 + vr) * 4 + (bb[1] * 2 + vc_local)
                    V[global_row][col] += bc * vc
            
            for (wf, wc) in w_local:
                wr = wf // 2
                wc_local = wf % 2
                for (cb, cc) in c_block_coeffs:
                    # W is C[k,i], flattened as k*4+i
                    # Block (cb, cc) means C rows cb*2..cb*2+1 (k), cols cc*2..cc*2+1 (i)
                    # Local (wr, wc_local) -> global k = cb*2+wr, global i = cc*2+wc_local
                    # W row = (cb*2+wr) * 4 + (cc*2+wc_local)
                    global_row = (cb[0] * 2 + wr) * 4 + (cb[1] * 2 + wc_local)
                    W[global_row][col] += cc * wc
            
            col += 1
    
    # col should now be rank_start + 49
    assert col == rank_start + 49, f"Expected col={rank_start+49}, got {col}"


def _schoolbook_decomposition(n, m, p):
    """Create a schoolbook (trivial) tensor decomposition for n x m by m x p matrix multiplication."""
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


def _verify_decomposition(decomp, n, m, p, num_checks=100):
    """Verify the decomposition by checking all valid entries and some invalid ones."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
