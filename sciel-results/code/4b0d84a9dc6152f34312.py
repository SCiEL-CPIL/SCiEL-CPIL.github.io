"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # 4, 5, 5
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # Strassen 2x2x2 coefficients
    # m_r = (sum u_ij A_ij) * (sum v_jk B_jk)
    # C_ik += w_ik * m_r
    
    # Indices for A: 0,1 (local)
    # Indices for B: 0,1 (local)
    # Indices for C: 0,1 (local)
    
    strassen_coeffs = [
        # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        # m1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # m2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # m3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # m4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # m5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # m6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # m7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]
    
    # 4x4x4 part using Strassen
    # 7 blocks, each 2x2x2
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for u_idx, u_vals, v_idx, v_vals, w_idx, w_vals in strassen_coeffs:
                    # We need to add terms for each non-zero entry in u, v, w
                    # But u, v, w are sparse.
                    # Actually, the term is a product of sums.
                    # (sum u A) (sum v B)
                    # This expands to sum_{a in u_idx} sum_{b in v_idx} u_a v_b A_a B_b
                    # And contributes w_c to C_c.
                    # So for each pair (a, b) in u_idx x v_idx:
                    # add_term(global_i(a), global_j(a), global_j(b), global_k(b), u_val * v_val, w_val)
                    
                    # Wait, the tensor term is U * V * W.
                    # If we expand the product, we get multiple rank-1 terms?
                    # No, Strassen's rank 7 means 7 multiplications.
                    # Each multiplication is a scalar product of linear forms.
                    # This is a single rank-1 tensor in the space of (LinearForm(A), LinearForm(B), LinearForm(C))?
                    # No, the tensor decomposition is in the space of A entries, B entries, C entries.
                    # A term $m_r = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$ is NOT a rank-1 tensor in $A \otimes B \otimes C$.
                    # It is a rank-1 tensor in $(\text{span}(A)) \otimes (\text{span}(B)) \otimes (\text{span}(C))$?
                    # No.
                    # The rank of the tensor $T$ is the minimum number of rank-1 tensors $u \otimes v \otimes w$ summing to $T$.
                    # A rank-1 tensor $u \otimes v \otimes w$ has entries $T_{ijk} = u_i v_j w_k$.
                    # Strassen's algorithm says $M_{2,2,2}$ has rank 7.
                    # This means there exist 7 vectors $u_r, v_r, w_r$ such that
                    # $T_{ijk} = \sum_{r=1}^7 u_{r,i} v_{r,j} w_{r,k}$.
                    
                    # My previous logic was slightly confused.
                    # Let's define the 7 rank-1 terms for $M_{2,2,2}$.
                    
                    # Term r:
                    # u_r is a 2x2 matrix (or vector of 4 elements).
                    # v_r is a 2x2 matrix.
                    # w_r is a 2x2 matrix.
                    
                    # From Strassen:
                    # m1 = (a00+a11)(b00+b11)
                    # This is NOT a single rank-1 term in the $A \otimes B \otimes C$ space.
                    # It is a product of two linear forms.
                    # However, the rank of the multiplication tensor is 7.
                    # This means we can write the tensor as sum of 7 rank-1 tensors.
                    
                    # Let's find the 7 rank-1 tensors for $M_{2,2,2}$.
                    # Reference: "Strassen's algorithm" decomposes the tensor into 7 rank-1 tensors.
                    
                    # The 7 rank-1 tensors are:
                    # T1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] ? No.
                    
                    # Let's derive.
                    # T_{ijk} = \delta_{i,k} \delta_{j,j}? No.
                    # T_{ijk} A_{ij} B_{jk} = C_{ik}.
                    
                    # The rank-1 tensors $R_r$ satisfy $\sum R_r = T$.
                    
                    # Standard decomposition:
                    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
                    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
                    # ...
                    
                    # Let's use the known 7 terms.
                    # See e.g. https://en.wikipedia.org/wiki/Strassen_algorithm
                    # The 7 multiplications are scalar products.
                    # But the tensor rank is 7.
                    
                    # Actually, the 7 multiplications $m_r$ are not the rank-1 tensors.
                    # The rank-1 tensors are the basis for the decomposition.
                    
                    # Let's construct the 7 rank-1 tensors for $M_{2,2,2}$.
                    
                    # Term 1:
                    # u1 = [1, 0, 0, 1] (a00, a01, a10, a11)
                    # v1 = [1, 0, 0, 1] (b00, b01, b10, b11)
                    # w1 = [1, 0, 0, 1] (c00, c01, c10, c11)
                    # Check: u1 A v1 B = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
                    # w1 C = c00 + c11.
                    # Does T1 A B = T1 C?
                    # T1_{ijk} = u_{ij} v_{jk} w_{ik}.
                    # T1_{000} = 1*1*1 = 1. A00 B00 -> C00.
                    # T1_{011} = 1*1*0 = 0.
                    # T1_{110} = 0*1*0 = 0.
                    # T1_{111} = 1*1*1 = 1. A11 B11 -> C11.
                    # Sum: A00B00 + A11B11.
                    # But m1 has cross terms.
                    
                    # So T1 is not m1.
                    
                    # The 7 rank-1 tensors are:
                    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
                    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
                    # 3. u=[1,0,0,0], v=[0,1,-1,0], w=[0,1,0,1]
                    # 4. u=[0,0,0,1], v=[-1,0,1,0], w=[1,0,1,0]
                    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
                    # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
                    # 7. u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
                    
                    # Let's verify Term 1:
                    # A00B00 + A11B11.
                    # C00 += 1, C11 += 1.
                    
                    # Term 2:
                    # u=[0,0,1,1] (a10, a11)
                    # v=[1,0,0,0] (b00)
                    # w=[0,0,1,-1] (c10, c11)
                    # A10B00 + A11B00.
                    # C10 += 1, C11 -= 1.
                    
                    # Sum 1+2:
                    # A00B00 + A10B00 + A11B00 + A11B11.
                    # C00=1, C10=1, C11=0.
                    
                    # This seems to work.
                    
                    # We will use these 7 terms.
                    
                    break # Break out of inner loop to restructure
                    
    # RESTRUCTURE: 
    # Define the 7 rank-1 tensors for 2x2x2.
    # u, v, w are lists of 4 elements corresponding to (0,0), (0,1), (1,0), (1,1).
    
    strassen_terms = [
        # u, v, w
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term_global(i_g, j_g, k_g, val_u, val_v, val_w):
        nonlocal col
        if col >= rank:
            return
        U[i_g * m + j_g][col] = val_u
        V[j_g * p + k_g][col] = val_v
        W[k_g * n + i_g][col] = val_w
        col += 1

    # 4x4x4 part
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for u, v, w in strassen_terms:
                    # u, v, w are length 4.
                    # Map local (li, lj) to global (gi, gj)
                    # li=0,1 -> gi = 2I + li
                    # lj=0,1 -> gj = 2J + lj
                    # lk=0,1 -> gk = 2K + lk
                    
                    # u[li*2 + lj]
                    # v[lj*2 + lk]
                    # w[li*2 + lk]
                    
                    # We need to add terms for all non-zero combinations?
                    # No, the rank-1 tensor is defined by u, v, w.
                    # The term in the sum is $\sum_{li, lj, lk} u_{li,lj} v_{lj,lk} w_{li,lk} A_{gi,gj} B_{gj,gk} C_{gi,gk}$?
                    # No.
                    # The rank-1 tensor $R$ has entries $R_{i,j,k} = u_{i,j} v_{j,k} w_{k,i}$?
                    # No.
                    # The tensor $T$ has entries $T_{i,j,k}$.
                    # A rank-1 tensor is $u_i v_j w_k$.
                    # Here indices are (i, j) for A, (j, k) for B, (k, i) for C.
                    # So a rank-1 term is $U_{i,j} V_{j,k} W_{k,i}$.
                    
                    # For the 2x2x2 block:
                    # We have 7 terms.
                    # For each term r, we have vectors $u_r, v_r, w_r$ of length 4.
                    # $u_r[li*2+lj]$, $v_r[lj*2+lk]$, $w_r[li*2+lk]$.
                    # We add a term for each non-zero triplet?
                    # No, the rank-1 tensor is a single object.
                    # But in our sparse representation, we add terms to U, V, W.
                    # If $u_r$ has multiple non-zeros, does it become multiple rank-1 terms?
                    # No.
                    # $U_{i,j}^{(r)} = u_r[li*2+lj]$.
                    # This is a single entry in the matrix U for column r.
                    # So for each r, we iterate over all li, lj, lk.
                    # If $u_r[li*2+lj] \neq 0$ AND $v_r[lj*2+lk] \neq 0$ AND $w_r[li*2+lk] \neq 0$:
                    # Then we have a non-zero contribution to $U_{gi,gj}^{(r)}, V_{gj,gk}^{(r)}, W_{gi,gk}^{(r)}$.
                    # But wait, U, V, W are matrices of size (n*m) x rank.
                    # Each column r is a vector.
                    # So for a fixed r, $U[:, r]$ is a vector.
                    # $U[gi*m+gj, r] = u_r[li*2+lj]$.
                    # This is consistent.
                    
                    # So for each r, we just set the values.
                    # But we are adding terms sequentially?
                    # No, each r is ONE column in U, V, W.
                    # So we should not "add_term" for each non-zero.
                    # We should assign to column r.
                    
                    # But my add_term increments col.
                    # So I should call add_term ONCE per rank-1 tensor?
                    # No, a rank-1 tensor in the full space might have non-zeros in multiple (i,j) pairs.
                    # But in U, it's just one column.
                    # So for each r, I fill the column.
                    
                    # However, my add_term is designed to add a scalar term.
                    # If I use add_term for each non-zero, I am treating the rank-1 tensor as a sum of basis tensors.
                    # Is $u \otimes v \otimes w$ (where u is a vector) a rank-1 tensor?
                    # Yes.
                    # But if u has multiple non-zeros, it is still ONE rank-1 tensor.
                    # So I should NOT call add_term for each non-zero.
                    # I should use ONE column for the entire rank-1 tensor.
                    
                    # But U, V, W are sparse.
                    # If I use one column, I set $U[gi*m+gj, col] = u_r[...]$.
                    # This is correct.
                    
                    # So I need a function to add a full rank-1 tensor defined by u, v, w.
                    
                    break # Break to implement correctly
    
    # Correct Implementation:
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_rank1_tensor(u_vec, v_vec, w_vec, offset_i, offset_j, offset_k, scale=1):
        """
        u_vec, v_vec, w_vec are length 4 vectors for 2x2 blocks.
        offset_i, offset_j, offset_k are global offsets.
        """
        nonlocal col
        if col >= rank:
            return
            
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    u_val = u_vec[li*2+lj]
                    v_val = v_vec[lj*2+lk]
                    w_val = w_vec[li*2+lk]
                    
                    if u_val == 0 or v_val == 0 or w_val == 0:
                        continue
                    
                    gi = offset_i + li
                    gj = offset_j + lj
                    gk = offset_k + lk
                    
                    U[gi*m + gj][col] += u_val * scale
                    V[gj*p + gk][col] += v_val * scale
                    W[gk*n + gi][col] += w_val * scale
        col += 1

    # Add 4x4x4 part
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for u, v, w in strassen_terms:
                    add_rank1_tensor(u, v, w, 2*I, 2*J, 2*K)
    
    # Add 4x4x1 part (16 terms)
    for i in range(4):
        for j in range(4):
            add_rank1_tensor([1 if (li==0 and lj==0) else 0 for li in range(2) for lj in range(2)], 
                             [1 if (lj==0 and lk==0) else 0 for lj in range(2) for lk in range(2)], 
                             [1 if (li==0 and lk==0) else 0 for li in range(2) for lk in range(2)], 
                             2*(i//2), 2*(j//2), 4)
            # Wait, this logic is messy.
            # 4x4x1 is just schoolbook.
            # i in 0..3, j in 0..3, k=4.
            # U[i*5+j] = 1, V[j*5+4] = 1, W[4*4+i] = 1.
            # This is a rank-1 tensor with single non-zeros.
            # I can just use add_term for these.
            
    # Let's use add_term for the simple parts.
    # Reset col and U,V,W? No, col is 49.
    
    def add_simple_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[i*m + j][col] = u_val
        V[j*p + k][col] = v_val
        W[k*n + i][col] = w_val
        col += 1

    # 4x4x1
    for i in range(4):
        for j in range(4):
            add_simple_term(i, j, 4, 1, 1, 1)
            
    # 4x1x4
    for i in range(4):
        for k in range(4):
            add_simple_term(i, 4, k, 1, 1, 1)
            
    # 4x1x1
    for i in range(4):
        add_simple_term(i, 4, 4, 1, 1, 1)
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
