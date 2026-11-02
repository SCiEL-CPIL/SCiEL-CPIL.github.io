"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_decomposition(n, m, p):
    """Build mixed Strassen/Schoolbook decomposition for n×m × m×p.
    Uses Strassen for the first 4 columns (rank 49) and Schoolbook for the remaining 2 columns (rank 32).
    Returns (U, V, W) as numpy arrays."""
    rank_terms = []
    
    # Part 1: First 4 columns using Strassen (Rank 49)
    # We split the 4x4x4 product into 7 products of 2x2x2.
    # Each 2x2x2 product has rank 7. Total 49.
    
    # Define the 7 Strassen terms for a 2x2x2 block.
    # Inputs: A (2x2), B (2x2). Output: C (2x2).
    # Indices in U: (i, j) where i in 0..1, j in 0..1 (local to 2x2 block)
    # Indices in V: (j, k) where j in 0..1, k in 0..1 (local to 2x2 block)
    # Indices in W: (k, i) where k in 0..1, i in 0..1 (local to 2x2 block)
    
    strassen_terms_2x2 = [
        # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2: (A10+A11)B00 -> C10(+1), C11(-1)
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        # M3: A00(B01-B11) -> C01(+1), C11(+1)
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        # M4: A11(B10-B00) -> C00(+1), C10(+1)
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        # M5: (A00+A01)B11 -> C00(-1), C01(+1)
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        # M6: (A10-A00)(B00+B01) -> C11(+1)
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        # M7: (A01-A11)(B10+B11) -> C00(+1)
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    # Apply Strassen to the 4x4x4 part.
    # We have 4 blocks of A (2x2), 4 blocks of B (2x2), 4 blocks of C (2x2).
    # A blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B blocks: B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
    # C blocks: C00 (0:2, 0:2), C01 (0:2, 2:4), C10 (2:4, 0:2), C11 (2:4, 2:4)
    
    # Strassen formulas for 4x4x4:
    # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
    # M3 = A00(B01-B11) -> C01(+1), C11(+1)
    # M4 = A11(B10-B00) -> C00(+1), C10(+1)
    # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
    # M6 = (A10-A00)(B00+B01) -> C11(+1)
    # M7 = (A01-A11)(B10+B11) -> C00(+1)
    
    # Each M is a 2x2x2 product. We expand each M into 7 rank-1 terms.
    
    # Helper to add a 2x2x2 Strassen term with offsets
    def add_strassen_term(a_off, b_off, w_offs, u_dict, v_dict, w_dict):
        # a_off: (row_start, col_start) for A
        # b_off: (row_start, col_start) for B
        # w_offs: list of (row_start, col_start) for C blocks with coefficients
        # u_dict, v_dict, w_dict: local indices for the 2x2x2 term
        
        global_u = {}
        global_v = {}
        global_w = {}
        
        for (li, lj), coeff_u in u_dict.items():
            gi = a_off[0] + li
            gj = a_off[1] + lj
            key = (gi, gj)
            global_u[key] = global_u.get(key, 0) + coeff_u
            
        for (lj, lk), coeff_v in v_dict.items():
            gj = b_off[0] + lj
            gk = b_off[1] + lk
            key = (gj, gk)
            global_v[key] = global_v.get(key, 0) + coeff_v
            
        for (lk, li), coeff_w in w_dict.items():
            for (w_row_start, w_col_start, coeff_c) in w_offs:
                gi = w_row_start + li
                gk = w_col_start + lk
                key = (gk, gi) # W index is (k, i)
                global_w[key] = global_w.get(key, 0) + coeff_w * coeff_c
                
        rank_terms.append((global_u, global_v, global_w))

    # M1
    # A00+A11: A(0:2,0:2) + A(2:4,2:4)
    # B00+B11: B(0:2,0:2) + B(2:4,2:4)
    # This is a product of (A00+A11) and (B00+B11).
    # We cannot directly use add_strassen_term because the inputs are sums.
    # We must expand (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This requires 4 products.
    # So we add 4 sets of Strassen terms.
    
    def add_product(a_off, b_off, w_offs):
        for u_dict, v_dict, w_dict in strassen_terms_2x2:
            add_strassen_term(a_off, b_off, w_offs, u_dict, v_dict, w_dict)
            
    # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    add_product((0,0), (0,0), [(0,0,1), (2,2,1)])
    add_product((0,0), (2,2), [(0,0,1), (2,2,1)])
    add_product((2,2), (0,0), [(0,0,1), (2,2,1)])
    add_product((2,2), (2,2), [(0,0,1), (2,2,1)])
    
    # M2: (A10+A11)B00 -> C10(+1), C11(-1)
    add_product((2,0), (0,0), [(2,0,1), (2,2,-1)])
    add_product((2,2), (0,0), [(2,0,1), (2,2,-1)])
    
    # M3: A00(B01-B11) -> C01(+1), C11(+1)
    add_product((0,0), (0,2), [(0,2,1), (2,2,1)])
    add_product((0,0), (2,2), [(0,2,-1), (2,2,-1)])
    
    # M4: A11(B10-B00) -> C00(+1), C10(+1)
    add_product((2,2), (2,0), [(0,0,1), (2,0,1)])
    add_product((2,2), (0,0), [(0,0,-1), (2,0,-1)])
    
    # M5: (A00+A01)B11 -> C00(-1), C01(+1)
    add_product((0,0), (2,2), [(0,0,-1), (0,2,1)])
    add_product((0,2), (2,2), [(0,0,-1), (0,2,1)])
    
    # M6: (A10-A00)(B00+B01) -> C11(+1)
    add_product((2,0), (0,0), [(2,2,1)])
    add_product((2,0), (0,2), [(2,2,1)])
    add_product((0,0), (0,0), [(2,2,-1)])
    add_product((0,0), (0,2), [(2,2,-1)])
    
    # M7: (A01-A11)(B10+B11) -> C00(+1)
    add_product((0,2), (2,0), [(0,0,1)])
    add_product((0,2), (2,2), [(0,0,1)])
    add_product((2,2), (2,0), [(0,0,-1)])
    add_product((2,2), (2,2), [(0,0,-1)])
    
    # Part 2: Last 2 columns using Schoolbook (Rank 32)
    # C[:, 4:6] = A[:, 0:4] * B[:, 4:6]
    # A is 4x4, B is 4x2.
    # C is 4x2.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_dict = {(i, j): 1}
                v_dict = {(j, k+4): 1}
                w_dict = {(k+4, i): 1}
                rank_terms.append((u_dict, v_dict, w_dict))
                
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, r] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify the decomposition is correct."""
    np.random.seed(seed)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_computed += np.einsum('ij,jk,ki->ik', u_vec, v_vec, w_vec)
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _remove_redundant_terms(U, V, W, n, m, p, tol=1e-9):
    """Remove linearly dependent rank-1 terms from the decomposition.
    Two terms r1 and r2 are redundant if the tensor T_r1 is a scalar multiple of T_r2,
    i.e., U[:,r1] ⊗ V[:,r1] ⊗ W[:,r1] = c * (U[:,r2] ⊗ V[:,r2] ⊗ W[:,r2]).
    We check this by seeing if the outer product matrices are proportional."""
    rank = U.shape[1]
    keep = list(range(rank))
    
    for i in range(len(keep)):
        if i >= len(keep):
            break
        ri = keep[i]
        for j in range(i + 1, len(keep)):
            rj = keep[j]
            # Check if term rj is a scalar multiple of term ri
            # Compare U vectors
            u_i = U[:, ri]
            u_j = U[:, rj]
            v_i = V[:, ri]
            v_j = V[:, rj]
            w_i = W[:, ri]
            w_j = W[:, rj]
            
            # Find scaling factor from U vectors
            mask_u = np.abs(u_i) > tol
            if not np.any(mask_u):
                # u_i is zero, check if u_j is also zero
                if np.allclose(u_j, 0, atol=tol):
                    # Both U vectors are zero, check V and W
                    mask_v = np.abs(v_i) > tol
                    if not np.any(mask_v):
                        if np.allclose(v_j, 0, atol=tol) and np.allclose(w_j, 0, atol=tol):
                            keep.remove(rj)
                            j -= 1
                        continue
                    # v_i is zero but v_j is not -> not redundant
                    continue
                else:
                    continue
            
            c_u = u_j[mask_u] / u_i[mask_u]
            # Check consistency
            if not np.allclose(c_u, c_u[0], rtol=1e-7, atol=tol):
                continue
            c = c_u[0]
            
            # Check V vectors
            if not np.allclose(v_j, c * v_i, rtol=1e-7, atol=tol):
                continue
            # Check W vectors
            if not np.allclose(w_j, c * w_i, rtol=1e-7, atol=tol):
                continue
            
            # Term rj is redundant
            keep.remove(rj)
            j -= 1
    
    if len(keep) < rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, True
    return U, V, W, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build the block-based Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to remove redundant terms
    U, V, W, reduced = _remove_redundant_terms(U, V, W, n, m, p)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # If reduction broke it, use the original
        U, V, W = _build_strassen_decomposition(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but safety check
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
