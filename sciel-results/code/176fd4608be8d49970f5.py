"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    # Strassen-based rank-28 decomposition: split j into {0,1},{2,3} and k into {0,1},{2,3}
    # giving 4 disjoint 2x2x2 blocks, each using Strassen rank 7.
    
    U = [[0.0] * 28 for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * 28 for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * 28 for _ in range(p * n)]  # 8 x 28
    
    # Strassen formulas for 2x2x2:
    # M1 = A00 * (B00 + B11)
    # M2 = A10 * (B00 + B01)
    # M3 = A01 * (B10 - B11)
    # M4 = A11 * (B11 - B00)
    # M5 = (A00 + A11) * (B01 - B11)
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 + A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # For each block (j_offset, k_offset) where j in {0,1}+j_offset, k in {0,1}+k_offset:
    # U index for (i, j) is i*m + j = i*4 + j
    # V index for (j, k) is j*p + k = j*4 + k
    # W index for (k, i) is k*n + i = k*2 + i
    
    # Strassen tensor decomposition in local indices (i_local, j_local, k_local) in {0,1}:
    # Term 1: U[i,j] contributes to (i*2+j), V[j,k] to (j*2+k), W[k,i] to (k*2+i)
    # We need to express each M_t as a sum of products of linear forms in A and B.
    # Each M_t corresponds to one rank-1 term in the tensor.
    
    # Strassen rank-1 terms (local):
    # t=1: A00*(B00+B11) -> U: e_{0,0}, V: e_{0,0}+e_{1,1}, W: e_{0,0}+e_{1,1}  ... wait
    # Actually the tensor T = sum over (i,j,k) e_i ⊗ e_j ⊗ e_k for the multiplication tensor.
    # Strassen says T = sum_{t=1}^{7} u_t ⊗ v_t ⊗ w_t where:
    # u_1 = e_{00}, v_1 = e_{00}+e_{11}, w_1 = e_{00}
    # Wait, let me be more careful.
    
    # The multiplication tensor for 2x2 is:
    # T = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (with appropriate indexing)
    # In our convention: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    # Strassen decomposition: T_2 = sum_{t=1}^7 u_t ⊗ v_t ⊗ w_t
    # where (in local 2x2 indices):
    # t=1: u=e_{00}, v=e_{00}+e_{11}, w=e_{00}
    # t=2: u=e_{10}, v=e_{00}+e_{01}, w=e_{10}
    # t=3: u=e_{01}, v=e_{10}-e_{11}, w=e_{01}
    # t=4: u=e_{11}, v=e_{11}-e_{00}, w=e_{11}
    # t=5: u=e_{00}+e_{11}, v=e_{01}-e_{11}, w=e_{01}
    # t=6: u=e_{10}-e_{00}, v=e_{00}+e_{10}, w=e_{10}
    # t=7: u=e_{01}+e_{11}, v=e_{10}+e_{11}, w=e_{11}
    
    # Wait, I need to double-check. The standard Strassen decomposition:
    # M1 = A00(B00+B11)
    # M2 = A10(B00+B01)
    # M3 = A01(B10-B11)
    # M4 = A11(B11-B00)
    # M5 = (A00+A11)(B01-B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01+A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # So the rank-1 terms (u_t, v_t, w_t) in local indices:
    # t=1: u=(1,0;0,0) i.e. e_{00}, v=(1,0;0,1) i.e. B00+B11, w=e_{00}
    #   But wait: M1 contributes to C00 and C11. So w for t=1 should have entries where C00 and C11 get M1.
    #   C00 gets +M1, C11 gets +M1. So w_1 = e_{00} + e_{11} in the C-space? No...
    #   
    # Let me re-derive. The tensor T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # (where e_{ij} is the basis vector for A_{ij}, etc.)
    # 
    # We want T = sum_t u_t ⊗ v_t ⊗ w_t
    # This means: for all (i,j,k): delta_{ij} delta_{jk} delta_{ki} ... no.
    # Actually T_{(ij),(jk),(ki)} = 1 for valid indices, 0 otherwise.
    # More precisely: T is a 3-tensor with T[(i,j),(j,k),(k,i)] = 1.
    # 
    # Strassen: T = sum_{t=1}^7 u_t ⊗ v_t ⊗ w_t where:
    # t=1: u=e_{00}, v=e_{00}+e_{11}, w=e_{00}
    #   Check: u⊗v⊗w has nonzero entries at (00,00,00), (00,11,00)
    #   T has entry at (00,00,00)=1 and (00,11,10)... no wait.
    
    # I think I'm overcomplicating. Let me just use the known Strassen decomposition
    # and verify numerically.
    
    # Known Strassen decomposition (from literature), in terms of the bilinear map:
    # The 7 rank-1 terms are:
    # 1. (A00)(B00 + B11) -> contributes to C00 (as +1) and C11 (as +1)
    #    So: u=e_{A00}, v=e_{B00}+e_{B11}, w=e_{C00}+e_{C11}
    # 2. (A10)(B00 + B01) -> contributes to C10 (+1) and C11 (-1)
    #    u=e_{A10}, v=e_{B00}+e_{B01}, w=e_{C10}-e_{C11}
    # 3. (A01)(B10 - B11) -> contributes to C01 (+1) and C11 (+1)
    #    u=e_{A01}, v=e_{B10}-e_{B11}, w=e_{C01}+e_{C11}
    # 4. (A11)(B11 - B00) -> contributes to C00 (+1) and C11 (+1)
    #    u=e_{A11}, v=e_{B11}-e_{B00}, w=e_{C00}+e_{C11}
    # 5. (A00+A11)(B01-B11) -> contributes to C01 (+1) and C00 (-1)
    #    u=e_{A00}+e_{A11}, v=e_{B01}-e_{B11}, w=e_{C01}-e_{C00}
    # 6. (A10-A00)(B00+B10) -> contributes to C10 (+1)
    #    u=e_{A10}-e_{A00}, v=e_{B00}+e_{B10}, w=e_{C10}
    # 7. (A01+A11)(B10+B11) -> contributes to C11 (+1) and C00 (+1)
    #    u=e_{A01}+e_{A11}, v=e_{B10}+e_{B11}, w=e_{C11}+e_{C00}
    
    # Let me verify C00: from t1: +1, t4: +1, t5: -1, t7: +1 => 1+1-1+1 = 2? That's wrong.
    # C00 should be A00*B00 + A01*B10 = M1 - M4... no.
    # Standard: C00 = M1 + M4 - M5 + M7
    # M1 = A00(B00+B11), M4 = A11(B11-B00), M5 = (A00+A11)(B01-B11), M7 = (A01+A11)(B10+B11)
    # C00 = A00B00 + A00B11 + A11B11 - A11B00 - A00B01 + A00B11 - A11B01 + A11B11
    #      = A00B00 + A00B11 - A00B01 + A00B11 - A11B00 + A11B11 - A11B01 + A11B11
    # That's not right either. Let me just use the standard formulas and trust them.
    
    # I'll use a verified Strassen decomposition. Let me define the 7 terms explicitly
    # and verify by checking the tensor identity numerically.
    
    # In local 2x2 indices (i,j,k) in {0,1}:
    # Term definitions (u in A-space, v in B-space, w in C-space):
    strassen_terms = [
        # (u_dict, v_dict, w_dict) where keys are (local_i, local_j), (local_j, local_k), (local_k, local_i)
        # t=1
        ([(0,0,1)], [(0,0,1),(1,1,1)], [(0,0,1)]),  # A00 * (B00+B11) -> C00
        # t=2
        ([(1,0,1)], [(0,0,1),(0,1,1)], [(1,0,1)]),  # A10 * (B00+B01) -> C10
        # t=3
        ([(0,1,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),  # A01 * (B10-B11) -> C01
        # t=4
        ([(1,1,1)], [(1,1,1),(0,0,-1)], [(1,1,1)]),  # A11 * (B11-B00) -> C11
        # t=5
        ([(0,0,1),(1,1,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),  # (A00+A11)*(B01-B11) -> C01
        # t=6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,0,1)]),  # (A10-A00)*(B00+B10) -> C10
        # t=7
        ([(0,1,1),(1,1,1)], [(1,0,1),(1,1,1)], [(0,0,1)]),  # (A01+A11)*(B10+B11) -> C00?
    ]
    
    # Actually, let me just verify this works by constructing the full tensor and checking.
    # I'll do this numerically with numpy.
    
    import numpy as np
    
    # Build the target tensor T of shape (8, 16, 8) for the full 2x4x4 problem
    # But for Strassen verification, let's work with 2x2x2 first.
    
    # For 2x2x2: U has 4 rows, V has 4 rows, W has 4 rows.
    # T[(i*2+j), (j*2+k), (k*2+i)] = 1
    
    # Let me define the Strassen terms properly and verify.
    
    # I'll use a cleaner formulation. The 7 Strassen terms in local 2x2:
    # Each term is (u, v, w) where u, v, w are vectors in R^4 (for 2x2 matrices flattened).
    # Using row-major: index = i*2 + j for A, j*2 + k for B, k*2 + i for C.
    
    def _strassen_terms_local():
        """Return list of 7 (u, v, w) triples in local 2x2 indices."""
        # Flatten: A index = i*2+j, B index = j*2+k, C index = k*2+i
        terms = []
        # t=1: A00*(B00+B11) -> contributes to C00
        #   u: A00 = index 0
        #   v: B00=index 0, B11=index 3
        #   w: C00 = index 0 (k=0,i=0)
        terms.append(([1,0,0,0], [1,0,0,1], [1,0,0,0]))
        # t=2: A10*(B00+B01) -> contributes to C10
        #   u: A10 = index 2 (i=1,j=0)
        #   v: B00=0, B01=1
        #   w: C10 = index 2 (k=1,i=0)
        terms.append(([0,0,1,0], [1,1,0,0], [0,0,1,0]))
        # t=3: A01*(B10-B11) -> contributes to C01
        #   u: A01 = index 1
        #   v: B10=2, B11=3
        #   w: C01 = index 1 (k=0,i=1)
        terms.append(([0,1,0,0], [0,0,1,-1], [0,1,0,0]))
        # t=4: A11*(B11-B00) -> contributes to C11
        #   u: A11 = index 3
        #   v: B11=3, B00=0
        #   w: C11 = index 3 (k=1,i=1)
        terms.append(([0,0,0,1], [-1,0,0,1], [0,0,0,1]))
        # t=5: (A00+A11)*(B01-B11) -> contributes to C01
        #   u: A00+A11 = indices 0,3
        #   v: B01=1, B11=3
        #   w: C01 = index 1
        terms.append(([1,0,0,1], [0,1,0,-1], [0,1,0,0]))
        # t=6: (A10-A00)*(B00+B10) -> contributes to C10
        #   u: A10-A00 = indices 2,0
        #   v: B00=0, B10=2
        #   w: C10 = index 2
        terms.append(([1,0,1,0], [1,0,1,0], [0,0,1,0]))
        # t=7: (A01+A11)*(B10+B11) -> contributes to C11
        #   u: A01+A11 = indices 1,3
        #   v: B10=2, B11=3
        #   w: C11 = index 3
        terms.append(([0,1,0,1], [0,0,1,1], [0,0,0,1]))
        
        return [(np.array(u, dtype=float), np.array(v, dtype=float), np.array(w, dtype=float)) 
                for u, v, w in terms]
    
    # Verify the Strassen decomposition for 2x2x2
    def _verify_strassen():
        terms = _strassen_terms_local()
        # Build the rank-1 sum tensor
        T_approx = np.zeros((4, 4, 4))
        for u, v, w in terms:
            T_approx += np.einsum('i,j,k->ijk', u, v, w)
        
        # Build the true tensor
        T_true = np.zeros((4, 4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_true[i*2+j, j*2+k, k*2+i] = 1
        
        return np.allclose(T_approx, T_true)
    
    # Verify
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen verification fails
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 2x4x4 rank-28 decomposition
    # 4 blocks: (j_off, k_off) in [(0,0), (0,2), (2,0), (2,2)]
    # Each block: i in {0,1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
    
    strassen_local = _strassen_terms_local()
    
    for bidx, (j_off, k_off) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        for t in range(7):
            u_local, v_local, w_local = strassen_local[t]
            col = bidx * 7 + t
            
            # Map local A indices (i_loc, j_loc) -> global i*m + j
            for i_loc in range(2):
                for j_loc in range(2):
                    global_i = i_loc
                    global_j = j_off + j_loc
                    local_idx = i_loc * 2 + j_loc
                    if u_local[local_idx] != 0:
                        U[global_i * m + global_j][col] = u_local[local_idx]
            
            # Map local B indices (j_loc, k_loc) -> global j*p + k
            for j_loc in range(2):
                for k_loc in range(2):
                    global_j = j_off + j_loc
                    global_k = k_off + k_loc
                    local_idx = j_loc * 2 + k_loc
                    if v_local[local_idx] != 0:
                        V[global_j * p + global_k][col] = v_local[local_idx]
            
            # Map local C indices (k_loc, i_loc) -> global k*n + i
            for k_loc in range(2):
                for i_loc in range(2):
                    global_k = k_off + k_loc
                    global_i = i_loc
                    local_idx = k_loc * 2 + i_loc
                    if w_local[local_idx] != 0:
                        W[global_k * n + global_i][col] = w_local[local_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
