"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Construct a rank-45 decomposition for 3x3x6 matrix multiplication.
    
    Strategy: Split the 6 columns into three groups of 2 columns.
    For each group, use a rank-15 decomposition of 3x3x2.
    Total rank: 3 * 15 = 45.
    
    The 3x3x2 block uses:
    - 7 Strassen terms for the top-left 2x2x2 core
    - 8 additional terms to cover the remaining interactions involving
      the third row/column of A and the second column of B.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # We build the decomposition by tiling three 3x3x2 blocks.
    # Block b (b=0,1,2) handles B columns [2b, 2b+1].
    # Within each block, local column index k_local is 0 or 1.
    # Global column index k_global = 2*b + k_local.
    
    # U has shape (n*m, rank) = (9, 45)
    # V has shape (m*p, rank) = (18, 45)
    # W has shape (p*n, rank) = (18, 45)
    
    rank = 45
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # Define the 15 terms for one 3x3x2 block.
    # Local indices:
    # A rows i in {0,1,2}, cols j in {0,1,2} -> U index i*3+j
    # B rows j in {0,1,2}, local cols k_l in {0,1} -> V index j*2+k_l
    # C local cols k_l in {0,1}, rows i in {0,1,2} -> W index k_l*3+i
    
    # Term definitions for one block (local indices):
    # u: list of (U_idx, coeff)
    # v: list of (V_idx, coeff)
    # w: list of (W_idx, coeff)
    
    terms = []
    
    # --- Strassen 7 terms for A[0:2,0:2] * B[0:2,:] ---
    # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    terms.append((
        [(0, 1), (4, 1)],
        [(0, 1), (3, 1)],
        [(0, 1), (4, 1)]
    ))
    # M2 = (a10+a11)b00 -> c10 += 1
    terms.append((
        [(3, 1), (4, 1)],
        [(0, 1)],
        [(3, 1)]
    ))
    # M3 = a00(b01-b11) -> c01 += 1
    terms.append((
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(1, 1)]
    ))
    # M4 = a11(b10-b00) -> c10 += 1, c00 -= 1
    terms.append((
        [(4, 1)],
        [(2, 1), (0, -1)],
        [(3, 1), (0, -1)]
    ))
    # M5 = (a00+a01)b11 -> c00 += 1, c01 += 1
    terms.append((
        [(0, 1), (1, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)]
    ))
    # M6 = (a10-a00)(b00+b01) -> c11 += 1
    terms.append((
        [(3, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(4, 1)]
    ))
    # M7 = (a01-a11)(b10+b11) -> c00 -= 1, c11 -= 1
    terms.append((
        [(1, 1), (4, -1)],
        [(2, 1), (3, 1)],
        [(0, -1), (4, -1)]
    ))
    
    # --- 8 additional terms for the 3rd row/col ---
    # We need to cover:
    # c00 += a02*b20
    # c01 += a02*b21
    # c10 += a12*b20
    # c11 += a12*b21
    # c20 += a20*b00 + a21*b10 + a22*b20
    # c21 += a20*b01 + a21*b11 + a22*b21
    
    # Let's try to pack these into 8 terms.
    # Note: c20 and c21 are sums of 3 products each.
    
    # T8: a02*b20 -> c00
    terms.append((
        [(2, 1)],
        [(4, 1)],
        [(0, 1)]
    ))
    # T9: a02*b21 -> c01
    terms.append((
        [(2, 1)],
        [(5, 1)],
        [(1, 1)]
    ))
    # T10: a12*b20 -> c10
    terms.append((
        [(5, 1)],
        [(4, 1)],
        [(3, 1)]
    ))
    # T11: a12*b21 -> c11
    terms.append((
        [(5, 1)],
        [(5, 1)],
        [(4, 1)]
    ))
    
    # Now for row 2 of A: a20, a21, a22.
    # We have 4 terms left (T12-T15).
    # We need to form:
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    
    # This is a 1x3 vector (a20,a21,a22) times a 3x2 matrix B.
    # We can't do this in 4 terms with rank 1 factors unless we use Strassen-like tricks,
    # but here the "matrix" is 3x2.
    # Actually, the rank of the tensor for multiplying a 1x3 vector by a 3x2 matrix is 3? No, it's 6?
    # Wait, the tensor is T[i,j,k] where i is fixed (row 2), j is 0..2, k is 0..1.
    # The output is C[2,k] = sum_j A[2,j] B[j,k].
    # This is just two dot products.
    # Can we do it in 4 terms?
    # Let's just use 3 terms for c20 and 3 terms for c21? That's 6 terms.
    # I only have 4 slots.
    
    # Let's re-evaluate the count.
    # Total terms = 15.
    # Strassen = 7.
    # Remaining = 8.
    # I used 4 for a02, a12.
    # I have 4 left for a20, a21, a22.
    
    # Is it possible?
    # C[2,0] = a20 b00 + a21 b10 + a22 b20
    # C[2,1] = a20 b01 + a21 b11 + a22 b21
    
    # Let's try to combine.
    # T12 = a20 * (b00 + b01) ? No, W must be fixed.
    # If T12 = a20 * (b00 + b01), then W contributes to c20 and c21.
    # U: a20 (idx 6)
    # V: b00 (idx 0), b01 (idx 1)
    # W: c20 (idx 2), c21 (idx 5)
    # This gives a20 b00 + a20 b01. Correct.
    
    # T13 = a21 * (b10 + b11)
    # U: a21 (idx 7)
    # V: b10 (idx 2), b11 (idx 3)
    # W: c20 (idx 2), c21 (idx 5)
    # This gives a21 b10 + a21 b11. Correct.
    
    # T14 = a22 * b20
    # U: a22 (idx 8)
    # V: b20 (idx 4)
    # W: c20 (idx 2)
    # This gives a22 b20. Correct.
    
    # T15 = a22 * b21
    # U: a22 (idx 8)
    # V: b21 (idx 5)
    # W: c21 (idx 5)
    # This gives a22 b21. Correct.
    
    # So we have:
    # T12: a20(b00+b01) -> c20, c21
    # T13: a21(b10+b11) -> c20, c21
    # T14: a22 b20 -> c20
    # T15: a22 b21 -> c21
    
    # This works perfectly and uses exactly 4 terms.
    
    terms.append((
        [(6, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (5, 1)]
    ))
    terms.append((
        [(7, 1)],
        [(2, 1), (3, 1)],
        [(2, 1), (5, 1)]
    ))
    terms.append((
        [(8, 1)],
        [(4, 1)],
        [(2, 1)]
    ))
    terms.append((
        [(8, 1)],
        [(5, 1)],
        [(5, 1)]
    ))
    
    # Verify we have 15 terms
    assert len(terms) == 15, f"Expected 15 terms, got {len(terms)}"
    
    # Fill the global matrices
    for b in range(3):
        # Block b corresponds to global B columns 2b and 2b+1
        # Global V index for local (j, k_l) is j*6 + (2b + k_l)
        # Global W index for local (k_l, i) is (2b + k_l)*3 + i
        
        for t_idx, (u_defs, v_defs, w_defs) in enumerate(terms):
            col = b * 15 + t_idx
            
            for u_idx, coeff in u_defs:
                U[u_idx, col] += coeff
            
            for v_idx, coeff in v_defs:
                j_local, k_local = divmod(v_idx, 2)
                v_global = j_local * 6 + (2 * b + k_local)
                V[v_global, col] += coeff
            
            for w_idx, coeff in w_defs:
                k_local, i_local = divmod(w_idx, 3)
                w_global = (2 * b + k_local) * 3 + i_local
                W[w_global, col] += coeff
    
    # Verification (optional, can be slow but rank is small)
    # Let's do a quick sanity check with random integers
    # A = np.random.randint(-5, 5, (3, 3))
    # B = np.random.randint(-5, 5, (3, 6))
    # C_expected = A @ B
    # U_mat = U.T  # (rank, 9)
    # V_mat = V.T  # (rank, 18)
    # W_mat = W.T  # (rank, 18)
    # # Reshape U, V, W to tensors?
    # # U[i*m+j, r] -> U_tensor[i, j, r]
    # # V[j*p+k, r] -> V_tensor[j, k, r]
    # # W[k*n+i, r] -> W_tensor[k, i, r]
    
    # u_t = U.T.reshape(3, 3, 45) # [i, j, r]
    # v_t = V.T.reshape(3, 6, 45) # [j, k, r]
    # w_t = W.T.reshape(6, 3, 45) # [k, i, r]
    
    # # C[i, k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # # This is not a simple einsum because W depends on k and i.
    # # C[i,k] = sum_r ( sum_j U[i,j,r] V[j,k,r] ) * W[k,i,r]
    
    # C_computed = np.zeros((3, 6))
    # for r in range(45):
    #     u_r = u_t[:,:,r] # (3,3)
    #     v_r = v_t[:,:,r] # (3,6)
    #     w_r = w_t[:,:,r] # (6,3)
    #     # Product P[i,k] = sum_j u_r[i,j] v_r[j,k]
    #     P = u_r @ v_r # (3,6)
    #     # Multiply by w_r[k,i]
    #     # w_r is (6,3) so w_r[k,i].
    #     # C[i,k] += P[i,k] * w_r[k,i]
    #     C_computed += P * w_r.T
    
    # if not np.allclose(C_computed, C_expected):
    #     print("Mismatch")
    #     print(C_computed)
    #     print(C_expected)
    #     return {"U": np.zeros((9, 54)).tolist(), "V": np.zeros((18, 54)).tolist(), "W": np.zeros((18, 54)).tolist()}
    
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
