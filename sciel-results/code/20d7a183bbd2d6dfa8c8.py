"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def _build_schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wval = W_np[k * n + i]
                for j in range(m):
                    Uval = U_np[i * m + j]
                    Vval = V_np[j * p + k]
                    s += np.dot(np.multiply(Uval, Vval), Wval)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _get_tensor_for_dims(n, m, p):
    """
    Returns the matrix multiplication tensor T of size (n*m*p)
    such that T[i*m+j, j*p+k, k*n+i] = 1 (flattened).
    Actually, we represent it as a vector of length n*m*p.
    Index mapping: flat_idx = (i*m + j)*p + k ? No.
    Let's use the convention:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    
    The tensor element T[a, b, c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    We flatten T into a vector of length n*m*p? No, the dimensions are n*m, m*p, p*n.
    The total number of elements in the tensor is n*m*p.
    We can map (i, j, k) to a single index 0..n*m*p-1.
    idx = (i * m + j) * p + k
    """
    size = n * m * p
    T = np.zeros(size)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                T[idx] = 1
    return T

def _rank1_tensor_vector(i, j, k, n, m, p):
    """
    Returns the vector representation of a rank-1 tensor with 1s at (i,j,k).
    This is just a vector with a 1 at the corresponding index.
    """
    size = n * m * p
    vec = np.zeros(size)
    idx = (i * m + j) * p + k
    vec[idx] = 1
    return vec

def _find_decomposition_for_block(n, m, p, target_rank, num_candidates=2000, seed=12345, timeout=10):
    """
    Attempts to find a decomposition of M_{n,m,p} with rank <= target_rank.
    Uses a greedy approach with random sparse rank-1 terms.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    T = _get_tensor_for_dims(n, m, p)
    
    # Generate a pool of candidate rank-1 tensors
    # We use random indices
    rng = np.random.RandomState(seed)
    candidates = []
    candidate_indices = []
    
    for _ in range(num_candidates):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        vec = _rank1_tensor_vector(i, j, k, n, m, p)
        candidates.append(vec)
        candidate_indices.append((i, j, k))
        
    candidates = np.array(candidates) # Shape (num_candidates, n*m*p)
    
    # We want to find a subset of size r such that T is in the span.
    # Since the rank-1 tensors are basis vectors (if we pick unique indices), 
    # the span is just the set of indices covered.
    # Wait, the rank-1 tensors in the decomposition are NOT necessarily basis vectors 
    # with 1s at single positions. They can have multiple 1s?
    # No, a rank-1 tensor in the matrix multiplication tensor context is defined by 
    # U, V, W vectors. 
    # U is a vector of length n*m. V is m*p. W is p*n.
    # The tensor is U \otimes V \otimes W.
    # The entries are U[a] * V[b] * W[c].
    # For this to be 1 at (i,j,k) and 0 elsewhere, U, V, W must be sparse.
    # Specifically, if U has support S_U, V has support S_V, W has support S_W.
    # The tensor has support S_U x S_V x S_W.
    # For the sum of such tensors to equal the multiplication tensor, 
    # the supports must cover the required indices.
    
    # The "random rank-1 tensor" approach above assumes the tensor is a basis vector.
    # But a general rank-1 term is U \otimes V \otimes W.
    # We can parameterize U, V, W as sparse vectors with random supports and random coefficients.
    
    # Let's restart the candidate generation with this model.
    candidates_vecs = []
    candidates_params = []
    
    for _ in range(num_candidates):
        # Random supports
        # Size of support: random between 1 and min(n*m, 4)
        sup_u_size = rng.randint(1, min(n*m, 4) + 1)
        sup_v_size = rng.randint(1, min(m*p, 4) + 1)
        sup_w_size = rng.randint(1, min(p*n, 4) + 1)
        
        u_idx = rng.choice(n*m, sup_u_size, replace=False)
        v_idx = rng.choice(m*p, sup_v_size, replace=False)
        w_idx = rng.choice(p*n, sup_w_size, replace=False)
        
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        u[u_idx] = 1.0
        v[v_idx] = 1.0
        w[w_idx] = 1.0
        
        # Tensor vector
        # T_vec[x] where x maps to (a,b,c)
        # a in 0..n*m-1, b in 0..m*p-1, c in 0..p*n-1
        # We need to map (a,b,c) to a linear index.
        # But the tensor is only defined on the "multiplication manifold".
        # The full tensor space is (n*m) x (m*p) x (p*n).
        # We only care about the components where a=i*m+j, b=j*p+k, c=k*n+i.
        # There are n*m*p such components.
        
        # Let's project the rank-1 tensor onto the multiplication manifold.
        proj = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = u[a] * v[b] * w[c]
                    if val != 0:
                        idx = (i*m + j)*p + k
                        proj[idx] += val
                        
        if np.any(proj):
            candidates_vecs.append(proj)
            candidates_params.append((u, v, w))
            
    if not candidates_vecs:
        return None
        
    candidates_arr = np.array(candidates_vecs)
    
    # Greedy selection
    # We want to find a linear combination of a subset of candidates that equals T.
    # This is a linear system: X * c = T, where X is (n*m*p) x (num_selected)
    # We select columns greedily to reduce the residual.
    
    residual = T.copy()
    selected = []
    
    for _ in range(target_rank):
        if time.time() - start_time > timeout:
            return None
            
        # Find candidate that maximizes dot product with residual
        # Or simply pick the one with largest absolute dot product
        dots = np.abs(candidates_arr @ residual)
        best_idx = np.argmax(dots)
        
        if dots[best_idx] < 1e-9:
            break
            
        selected.append(best_idx)
        
        # Update residual by projecting out the selected candidate?
        # No, we need to solve for coefficients later.
        # For greedy, we can just subtract the best fit.
        # coeff = dot(residual, candidate) / dot(candidate, candidate)
        c = candidates_arr[best_idx]
        coeff = np.dot(residual, c) / np.dot(c, c)
        residual -= coeff * c
        
    if np.linalg.norm(residual) < 1e-9:
        # We have a set of candidates that spans T.
        # Now solve for the exact coefficients.
        # Build matrix A of selected candidates
        A = np.array([candidates_arr[i] for i in selected])
        # Solve A c = T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, T, rcond=None)
            # Verify
            recon = A @ coeffs
            if np.allclose(recon, T, atol=1e-9):
                # Construct U, V, W
                rank = len(selected)
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                for idx, sel_idx in enumerate(selected):
                    u, v, w = candidates_params[sel_idx]
                    coeff = coeffs[idx]
                    U[:, idx] = u * coeff
                    V[:, idx] = v * coeff
                    W[:, idx] = w * coeff
                    
                return U, V, W
        except:
            return None
            
    return None

def _build_strassen_tiled_4x4_4x6():
    """
    Fallback: The current best known constructive decomposition.
    """
    # Re-implementing the logic from the current program to ensure we have a baseline
    # If the search fails, we return this.
    
    def _strassen_2x2_factors():
        factors = []
        factors.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
        factors.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}})
        factors.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,1): 1, (1,1): 1}})
        factors.append({'U': {(1,1): 1}, 'V': {(1,1): 1, (0,1): -1}, 'W': {(0,0): 1, (1,0): 1}})
        factors.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): -1}})
        factors.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
        factors.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
        return factors

    factors = _strassen_2x2_factors()
    
    # Build 4x4x4 (rank 49)
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            f1 = factors[l1]
            f2 = factors[l2]
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in f1['U']: continue
                    c1 = f1['U'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in f2['U']: continue
                            c2 = f2['U'][(si, sj)]
                            ai = bi*2 + si
                            aj = bj*2 + sj
                            U_444[ai*4 + aj][col] += c1*c2
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in f1['V']: continue
                    c1 = f1['V'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in f2['V']: continue
                            c2 = f2['V'][(si, sj)]
                            bri = bi*2 + si
                            bcj = bj*2 + sj
                            V_444[bri*4 + bcj][col] += c1*c2
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in f1['W']: continue
                    c1 = f1['W'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in f2['W']: continue
                            c2 = f2['W'][(si, sj)]
                            cri = bi*2 + si
                            ccj = bj*2 + sj
                            W_444[ccj*4 + cri][col] += c1*c2
            col += 1

    # Build 4x4x2 (rank 28)
    U_442 = [[0] * 28 for _ in range(16)]
    V_442 = [[0] * 28 for _ in range(8)]
    W_442 = [[0] * 28 for _ in range(8)]
    col = 0
    for block_a_i in range(2):
        for block_a_j in range(2):
            b_row_offset = block_a_i * 2
            out_row_offset = block_a_i * 2
            for f in range(7):
                factor = factors[f]
                for (ri, ci), val in factor['U'].items():
                    a_row = block_a_i * 2 + ri
                    a_col = block_a_j * 2 + ci
                    U_442[a_row * 4 + a_col][col] += val
                for (ri, ci), val in factor['V'].items():
                    b_row = b_row_offset + ri
                    b_col = ci
                    V_442[b_row * 2 + b_col][col] += val
                for (ri, ci), val in factor['W'].items():
                    out_row = out_row_offset + ri
                    out_col = ci
                    W_442[out_col * 4 + out_row][col] += val
                col += 1

    # Combine
    rank_total = 49 + 28
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(24)] # 4*6
    W = [[0] * rank_total for _ in range(24)] # 6*4

    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
    for i in range(16):
        for c in range(49):
        # V_444 is 16x49 (4x4). Map to 4x6 (cols 0-3)
        # V_444 row j*4+k. Full V row j*6+k.
            j = i // 4
            k = i % 4
            full_v_idx = j * 6 + k
            V[full_v_idx][c] = V_444[i][c]
    for i in range(16):
        for c in range(49):
        # W_444 is 16x49 (4x4, k*n+i). Map to 6x4 (cols 0-3)
        # W_444 row k*4+i. Full W row (k+0)*4+i.
            k = i // 4
            out_i = i % 4
            full_w_idx = k * 4 + out_i
            W[full_w_idx][c] = W_444[i][c]

    # Add 4x4x2 part
    col_offset = 49
    for i in range(16):
        for c in range(28):
            U[i][c + col_offset] = U_442[i][c]
    for i in range(8):
        for c in range(28):
            j = i // 2
            k = i % 2
            full_v_idx = j * 6 + (k + 4)
            V[full_v_idx][c + col_offset] = V_442[i][c]
    for i in range(8):
        for c in range(28):
            k = i // 4
            out_i = i % 4
            full_w_idx = (k + 4) * 4 + out_i
            W[full_w_idx][c + col_offset] = W_442[i][c]

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # 1. Try to find a better decomposition using block search
    # Strategy: Decompose 4x4x6 into four 2x4x3 blocks.
    # If we find a rank-19 decomposition for 2x4x3, total rank is 76.
    
    best = _build_strassen_tiled_4x4_4x6()
    best_rank = 77
    
    # Attempt search for 2x4x3 rank 19
    # This is a structural change: using smaller, non-standard blocks and search.
    if time.time() < deadline - 20:
        try:
            block_U, block_V, block_W = _find_decomposition_for_block(2, 4, 3, 19, num_candidates=1000, timeout=15)
            if block_U is not None:
                # Construct full 4x4x6 from 4 blocks
                # A: 4x4. Blocks: Top-Left (2x4), Top-Right? No.
                # A is 4x4. Split into 4 blocks of 2x2? No, 2x4x3 block implies A is 2x4.
                # So we split A into Top (2x4) and Bottom (2x4).
                # B is 4x6. Split into Left (4x3) and Right (4x3).
                # Output C is 4x6.
                # C_TL = A_Top * B_Left (2x3)
                # C_TR = A_Top * B_Right (2x3)
                # C_BL = A_Bottom * B_Left (2x3)
                # C_BR = A_Bottom * B_Right (2x3)
                
                rank_block = 19
                total_rank = 4 * rank_block
                U_full = [[0] * total_rank for _ in range(16)]
                V_full = [[0] * total_rank for _ in range(24)]
                W_full = [[0] * total_rank for _ in range(24)]
                
                # Block 1: A_Top (rows 0-1, cols 0-3), B_Left (rows 0-3, cols 0-2) -> C_TL (rows 0-1, cols 0-2)
                # Block 2: A_Top, B_Right (rows 0-3, cols 3-5) -> C_TR (rows 0-1, cols 3-5)
                # Block 3: A_Bottom (rows 2-3, cols 0-3), B_Left -> C_BL (rows 2-3, cols 0-2)
                # Block 4: A_Bottom, B_Right -> C_BR (rows 2-3, cols 3-5)
                
                blocks = [
                    {'a_offset': 0, 'b_offset': 0, 'c_offset': 0, 'a_rows': [0,1], 'b_cols': [0,1,2], 'c_rows': [0,1], 'c_cols': [0,1,2]},
                    {'a_offset': 0, 'b_offset': 3, 'c_offset': 3, 'a_rows': [0,1], 'b_cols': [3,4,5], 'c_rows': [0,1], 'c_cols': [3,4,5]},
                    {'a_offset': 8, 'b_offset': 0, 'c_offset': 8, 'a_rows': [2,3], 'b_cols': [0,1,2], 'c_rows': [2,3], 'c_cols': [0,1,2]}, # A offset 8? A is 4x4=16. Row 2,3 start at 2*4=8.
                    {'a_offset': 8, 'b_offset': 3, 'c_offset': 11, 'a_rows': [2,3], 'b_cols': [3,4,5], 'c_rows': [2,3], 'c_cols': [3,4,5]},
                ]
                
                # Note: The block decomposition U, V, W is for M_{2,4,3}.
                # U shape: 2*4=8. V shape: 4*3=12. W shape: 3*2=6.
                
                for b_idx, block in enumerate(blocks):
                    col_start = b_idx * rank_block
                    
                    # Map U
                    # Block U index: i_block * 4 + j_block (i_block 0..1, j_block 0..3)
                    # Full U index: i_full * 4 + j_full
                    # i_full = block['a_rows'][i_block]
                    # j_full = j_block (since A cols 0-3 are used for both left/right? No.)
                    # Wait, A is 4x4. 
                    # Block 1 uses A_Top (rows 0-1, all cols 0-3).
                    # So j_full = j_block.
                    # Block 3 uses A_Bottom (rows 2-3, all cols 0-3).
                    
                    for i_block in range(2):
                        for j_block in range(4):
                            u_idx_block = i_block * 4 + j_block
                            i_full = block['a_rows'][i_block]
                            j_full = j_block
                            u_idx_full = i_full * 4 + j_full
                            for c in range(rank_block):
                                U_full[u_idx_full][col_start + c] = block_U[u_idx_block][c]
                                
                    # Map V
                    # Block V index: j_block * 3 + k_block (j_block 0..3, k_block 0..2)
                    # Full V index: j_full * 6 + k_full
                    # j_full = j_block
                    # k_full = block['b_cols'][k_block]
                    
                    for j_block in range(4):
                        for k_block in range(3):
                            v_idx_block = j_block * 3 + k_block
                            j_full = j_block
                            k_full = block['b_cols'][k_block]
                            v_idx_full = j_full * 6 + k_full
                            for c in range(rank_block):
                                V_full[v_idx_full][col_start + c] = block_V[v_idx_block][c]
                                
                    # Map W
                    # Block W index: k_block * 2 + i_block (k_block 0..2, i_block 0..1)
                    # Full W index: k_full * 4 + i_full
                    # k_full = block['c_cols'][k_block]
                    # i_full = block['c_rows'][i_block]
                    
                    for k_block in range(3):
                        for i_block in range(2):
                            w_idx_block = k_block * 2 + i_block
                            k_full = block['c_cols'][k_block]
                            i_full = block['c_rows'][i_block]
                            w_idx_full = k_full * 4 + i_full
                            for c in range(rank_block):
                                W_full[w_idx_full][col_start + c] = block_W[w_idx_block][c]
                                
                # Verify
                if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                    best = {"U": U_full, "V": V_full, "W": W_full}
                    best_rank = total_rank
                    break
                    
        except Exception as e:
            pass
            
    return best
# EVOLVE-BLOCK-END
