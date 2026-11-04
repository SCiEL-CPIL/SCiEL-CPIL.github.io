import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(n, m, p, U, V, W, num_checks=50):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    Uses random sampling for speed.
    """
    rank = U.shape[1]
    # Convert to numpy arrays if not already
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
    
    # Full check is 6*6*6 = 216 checks, which is very fast in numpy
    # Let's do a full check to be safe, it's O(216 * rank)
    # rank is ~189, so 216*189 ~ 40k ops, negligible.
    
    # We need to check T[i,j,k] = 1
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indexing:
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # We can vectorize this.
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    # Get the rows
    U_sel = U[u_flat_idx, :] # shape (n, m, p, rank)
    V_sel = V[v_flat_idx, :] # shape (n, m, p, rank)
    W_sel = W[w_flat_idx, :] # shape (n, m, p, rank)
    
    # Compute sum over rank
    prod = U_sel * V_sel * W_sel
    sum_rank = np.sum(prod, axis=3) # shape (n, m, p)
    
    # Check if all are 1
    if not np.allclose(sum_rank, 1.0, atol=1e-9):
        return False
        
    return True

def _build_strassen_189(n, m, p):
    """
    Construct rank-189 decomposition for 6x6x6.
    Structure: 2x2 Strassen blocks of size 3x3.
    Inner 3x3 multiplication uses schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    """
    bs = 3 # Block size
    nb = 2 # Number of blocks (2x2 grid)
    rank = 7 * (bs ** 3)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # Each M corresponds to a set of contributions to U, V, W.
    # We iterate over the 7 Strassen steps.
    
    # Define the Strassen steps as (u_coeff, v_coeff, w_coeff)
    # where coeff is a dict mapping (block_row, block_col) to scalar.
    # Note: U corresponds to A, V to B, W to C.
    # A is n x m, split into nb x nb blocks of bs x bs.
    # B is m x p, split into nb x nb blocks of bs x bs.
    # C is p x n, split into nb x nb blocks of bs x bs.
    
    steps = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col_offset = 0
    for u_coeffs, v_coeffs, w_coeffs in steps:
        # For this Strassen step, we need to perform a bs x bs x bs schoolbook multiplication
        # for every combination of block indices involved in the coefficients?
        # No, the Strassen algorithm computes M = (sum u A) * (sum v B).
        # The result is added to C with weights w.
        # So, for each Strassen step, we effectively perform one matrix multiplication of size bs x bs.
        # We decompose this bs x bs multiplication into bs^3 rank-1 terms (schoolbook).
        
        # The "A" part of this step is a linear combination of A blocks.
        # The "B" part is a linear combination of B blocks.
        # The "C" part is a linear combination of C blocks.
        
        # We need to construct the rank-1 tensors for this specific "virtual" multiplication.
        # The virtual A matrix A_virt has entries:
        # A_virt[i, j] = sum_{(bi, bj) in u_coeffs} u_coeffs[(bi, bj)] * A[bi*bs + i, bj*bs + j]
        # Similarly for B_virt and C_virt.
        
        # We want to express A_virt * B_virt = C_virt (conceptually) using rank-1 terms.
        # Actually, we are constructing the decomposition of the *original* matrix multiplication.
        # The term in the decomposition for this Strassen step and inner index (i,j,k) is:
        # U[row_A, col] * V[row_B, col] * W[row_C, col]
        
        # Let's iterate over the inner indices i, j, k for the bs x bs block.
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    # This corresponds to the scalar multiplication A_virt[i, j] * B_virt[j, k] contributing to C_virt[k, i]
                    
                    # U contribution:
                    # U[row_A, col] should be such that summing over the virtual blocks gives the correct A_virt[i,j]
                    # A_virt[i, j] = sum_{(bi, bj)} u[(bi, bj)] A[bi*bs+i, bj*bs+j]
                    # So for a fixed (i, j), U[row_A, col] is non-zero only for row_A corresponding to (bi, bj) in u_coeffs.
                    # Specifically, U[(bi*bs+i)*m + (bj*bs+j), col] += u[(bi, bj)]
                    
                    for (bi, bj), val_u in u_coeffs.items():
                        r_u = (bi * bs + i) * m + (bj * bs + j)
                        U[r_u, col_offset] += val_u
                    
                    # V contribution:
                    # B_virt[j, k] = sum_{(bj, bk)} v[(bj, bk)] B[bj*bs+j, bk*bs+k]
                    # So V[(bj*bs+j)*p + (bk*bs+k), col] += v[(bj, bk)]
                    for (bj, bk), val_v in v_coeffs.items():
                        r_v = (bj * bs + j) * p + (bk * bs + k)
                        V[r_v, col_offset] += val_v
                    
                    # W contribution:
                    # C_virt[k, i] = sum_{(bk, bi)} w[(bk, bi)] C[bk*bs+k, bi*bs+i]
                    # So W[(bk*bs+k)*n + (bi*bs+i), col] += w[(bk, bi)]
                    for (bk, bi), val_w in w_coeffs.items():
                        r_w = (bk * bs + k) * n + (bi * bs + i)
                        W[r_w, col_offset] += val_w
                        
                    col_offset += 1
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Build the rank-189 decomposition
    U, V, W = _build_strassen_189(n, m, p)
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 216 if something went wrong (should not happen)
        rank = n * m * p
        U_fb = np.zeros((n * m, rank), dtype=float)
        V_fb = np.zeros((m * p, rank), dtype=float)
        W_fb = np.zeros((p * n, rank), dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_fb[u_idx, idx] = 1.0
                    V_fb[v_idx, idx] = 1.0
                    W_fb[w_idx, idx] = 1.0
                    idx += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
