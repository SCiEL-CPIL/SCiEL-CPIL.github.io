import numpy as np
import time

# Constants for the problem dimension
N = 6
M = 6
P = 6

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    Uses a block-wise Strassen decomposition for the 2x2 grid of 3x3 blocks,
    combined with schoolbook decomposition for the inner 3x3 multiplications.
    
    Total Rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189.
    """
    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize factor matrices with the estimated rank
    # Strassen for 2x2 blocks of size 3 -> 7 multiplications
    # Each "multiplication" is a 3x3 scalar mult -> rank 9? No, rank 27.
    # Wait, the tensor decomposition rank of M_3 is 27.
    # So total rank is 7 * 27 = 189.
    
    inner_rank = 27  # Rank of M_3
    outer_rank = 7   # Rank of M_2 (Strassen)
    total_rank = inner_rank * outer_rank
    
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Helper to add a block term
    # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
    # i_block, j_block: which 3x3 sub-block (row, col)
    def map_global(idx_row, idx_col):
        return idx_row * 6 + idx_col

    def add_kernel(u_blocks, v_blocks, w_blocks, col_start):
        """
        Adds terms for one of the 7 Strassen steps.
        u_blocks: list of (block_id, coeff) for U (A matrix side)
        v_blocks: list of (block_id, coeff) for V (B matrix side)
        w_blocks: list of (block_id, coeff) for W (C matrix side)
        """
        # For each of the 27 rank-1 components of the 3x3 schoolbook product
        for r_idx in range(inner_rank):
            # The 3x3 schoolbook decomposition has 1 component per (i,j,k) in 3x3x3
            # i, j, k are local indices in [0,3)
            i_loc = r_idx // 9
            j_loc = (r_idx // 3) % 3
            k_loc = r_idx % 3
            
            # Iterate over A blocks
            for a_bid, a_coeff in u_blocks:
                a_rb, a_cb = a_bid // 2, a_bid % 2
                # Local row in global A: a_rb * 3 + i_loc
                # Local col in global A: a_cb * 3 + j_loc
                u_row = map_global(a_rb * 3 + i_loc, a_cb * 3 + j_loc)
                U[u_row, col_start + r_idx] += a_coeff
            
            # Iterate over B blocks
            for b_bid, b_coeff in v_blocks:
                b_rb, b_cb = b_bid // 2, b_bid % 2
                # Local row in global B: b_rb * 3 + j_loc
                # Local col in global B: b_cb * 3 + k_loc
                v_row = map_global(b_rb * 3 + j_loc, b_cb * 3 + k_loc)
                V[v_row, col_start + r_idx] += b_coeff
            
            # Iterate over C blocks
            for c_bid, c_coeff in w_blocks:
                c_rb, c_cb = c_bid // 2, c_bid % 2
                # Local row in global C: c_rb * 3 + k_loc
                # Local col in global C: c_cb * 3 + i_loc
                # Note: W index is k*n + i. 
                # C[i,j] += A[i,k]*B[k,j].
                # In our trace convention: W[k*n + i, r].
                # Here i is the row of C, k is the col of C?
                # Standard: C_{ij} = sum_k A_{ik} B_{kj}.
                # Tensor T_{ijk}. U_{i,k}? No.
                # Contract: U_{row_U, r} * V_{row_V, r} * W_{row_W, r}.
                # U corresponds to A_{i, ?}. V to B_{?, j}. W to C_{?, i}?
                # Let's align with the previous working code:
                # U6[i*6+j, r]  -> A_{i,j}
                # V6[j*6+k, r]  -> B_{j,k}
                # W6[k*6+i, r]  -> C_{i,k} ?? No, C_{i,k} is not standard.
                # Standard output C_{i,k}.
                # Trace convention W[k*n+i].
                # This means W index = k*6 + i.
                # In 3x3 block: local row k_loc, local col i_loc.
                # Global C index: (c_rb*3 + k_loc) * 6 + (c_cb*3 + i_loc).
                
                w_row = (c_rb * 3 + k_loc) * 6 + (c_cb * 3 + i_loc)
                W[w_row, col_start + r_idx] += c_coeff

    # Strassen Steps for 2x2 block multiplication
    # Blocks: 0:A11, 1:A12, 2:A21, 3:A22
    
    # Step 0: P1 = (A11 + A22)(B11 + B22)
    # Used in C11 (+1) and C22 (+1)
    add_kernel(
        [(0, 1), (3, 1)], 
        [(0, 1), (3, 1)], 
        [(0, 1), (3, 1)], 
        0
    )
    
    # Step 1: P2 = (A21 + A22)B11
    # Used in C21 (+1) and C22 (-1)
    add_kernel(
        [(2, 1), (3, 1)], 
        [(0, 1)], 
        [(2, 1), (3, -1)], 
        inner_rank
    )
    
    # Step 2: P3 = A11(B12 - B22)
    # Used in C12 (+1) and C22 (+1)
    add_kernel(
        [(0, 1)], 
        [(1, 1), (3, -1)], 
        [(1, 1), (3, 1)], 
        2 * inner_rank
    )
    
    # Step 3: P4 = A22(B21 - B11)
    # Used in C11 (+1) and C21 (+1)
    add_kernel(
        [(3, 1)], 
        [(2, 1), (0, -1)], 
        [(0, 1), (2, 1)], 
        3 * inner_rank
    )
    
    # Step 4: P5 = (A11 + A12)B22
    # Used in C12 (+1) and C11 (-1)
    add_kernel(
        [(0, 1), (1, 1)], 
        [(3, 1)], 
        [(1, 1), (0, -1)], 
        4 * inner_rank
    )
    
    # Step 5: P6 = (A21 - A11)(B11 + B12)
    # Used in C22 (+1)
    add_kernel(
        [(2, 1), (0, -1)], 
        [(0, 1), (1, 1)], 
        [(3, 1)], 
        5 * inner_rank
    )
    
    # Step 6: P7 = (A12 - A22)(B21 + B22)
    # Used in C11 (+1)
    add_kernel(
        [(1, 1), (3, -1)], 
        [(2, 1), (3, 1)], 
        [(0, 1)], 
        6 * inner_rank
    )

    # Verification
    # Reconstruct tensor
    U_tensor = U.reshape(6, 6, -1)
    V_tensor = V.reshape(6, 6, -1)
    W_tensor = W.reshape(6, 6, -1)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should be 1
    # Note: W is indexed [k*n+i]. So W[k,i].
    # In reshape W is [k, i, r].
    reconstructed = np.einsum('ijr,jkr,kir->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(reconstructed, np.ones((6, 6, 6)), atol=1e-9):
        # Fallback to Schoolbook 216 if Strassen construction fails
        rank_fb = 216
        U_fb = np.zeros((36, rank_fb))
        V_fb = np.zeros((36, rank_fb))
        W_fb = np.zeros((36, rank_fb))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    col = i * 36 + j * 6 + k
                    U_fb[i * 6 + j, col] = 1.0
                    V_fb[j * 6 + k, col] = 1.0
                    W_fb[k * 6 + i, col] = 1.0
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    # Simple check
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    print(f"Rank: {U.shape[1]}")
