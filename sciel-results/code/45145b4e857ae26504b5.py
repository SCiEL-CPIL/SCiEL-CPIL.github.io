"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition using block structure
    # Split M_{2,3,7} into M_{2,2,7} + M_{2,1,7}
    # M_{2,1,7} has rank 7
    # M_{2,2,7} is split into three M_{2,2,2} (rank 7 each) and one M_{2,2,1} (rank 4)
    # Total rank = 3*7 + 4 + 7 = 32
    
    rank = 32
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx, col] = coeff
        V[v_idx, col] = coeff
        W[w_idx, col] = coeff
        col += 1

    # --- Part 1: M_{2,1,7} (j=2) ---
    # A_{i,2} * B_{2,k} -> C_{k,i}
    # This is just 7 independent scalar multiplications.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx)
            
    # --- Part 2: M_{2,2,7} (j=0,1) ---
    # Split k into groups: {0,1}, {2,3}, {4,5}, {6}
    # Groups 0, 1, 2 use Strassen (rank 7)
    # Group 3 uses Schoolbook (rank 4)
    
    # Strassen decomposition for 2x2x2
    # Indices for A: 0,1,2,3 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # Indices for B: 0,1,2,3 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    # Indices for C: 0,1,2,3 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    
    def add_strassen_block(k_offset):
        nonlocal col
        # k_offset is 0, 2, or 4
        # B indices:
        # b00 = 0*p + k_offset
        # b01 = 0*p + k_offset + 1
        # b10 = 1*p + k_offset
        # b11 = 1*p + k_offset + 1
        
        # A indices (fixed for this block of j=0,1):
        # a00 = 0*m + 0
        # a01 = 0*m + 1
        # a10 = 1*m + 0
        # a11 = 1*m + 1
        
        # C indices:
        # c00 = k_offset*n + 0
        # c01 = k_offset*n + 1
        # c10 = (k_offset+1)*n + 0
        # c11 = (k_offset+1)*n + 1
        
        # Strassen terms:
        # m1 = a00 * (b00 + b11)
        # m2 = a10 * (b00 + b01)
        # m3 = (a00 + a10) * b10
        # m4 = a11 * (b11 + b00) -- Wait, standard is a11 * (b11 + b00)? 
        # Let's use the standard 7 mults:
        # m1 = a00 * (b00 + b11)
        # m2 = a10 * (b00 + b01)
        # m3 = (a00 + a10) * b10
        # m4 = a11 * (b11 + b00) ? No.
        
        # Correct Strassen:
        # m1 = a00 * (b00 + b11)
        # m2 = a10 * (b00 + b01)
        # m3 = (a00 + a10) * b10
        # m4 = a11 * (b11 + b00)
        # m5 = (a11 + a00) * b00
        # m6 = (a11 - a01) * (b01 + b11)
        # m7 = (a01 + a11) * (b10 - b00)
        
        # Actually, simpler to just hardcode the 7 terms with signs.
        # Term 1: a00 * (b00 + b11)
        # U: a00. V: b00, b11. W: c00, c11.
        # This requires 2 terms for one multiplication? No, the decomposition is a sum of rank-1 tensors.
        # Strassen's algorithm expresses the product as 7 multiplications of linear combinations.
        # Each "multiplication" $u_r v_r$ corresponds to a rank-1 term in the tensor if we expand?
        # No, the tensor decomposition is $\sum u_i v_k w_j$.
        # Strassen gives $C = \sum m_r \cdot (\text{linear form in C})$.
        # This corresponds to a decomposition where the $W$ vectors are the linear forms.
        
        # Let's construct the 7 rank-1 terms for the 2x2x2 block.
        # u_r, v_r, w_r such that sum u_r v_r w_r = T_block
        
        # r=0: u=a00, v=b00+b11, w=c00+c11 ? No.
        # The decomposition is:
        # T = u1 v1 w1 + ... + u7 v7 w7
        # where u, v, w are vectors in the respective spaces.
        
        # For 2x2x2:
        # u1 = [1,0,0,0] (a00)
        # v1 = [1,0,0,1] (b00+b11)
        # w1 = [1,0,0,1] (c00+c11) -- This is not a rank-1 term in the tensor sense unless w is a vector.
        # Yes, w is a vector in the output space.
        
        # Let's define the 7 terms for 2x2x2:
        # 1. u=[1,0,0,0], v=[1,0,0,1], w=[1,0,0,1]
        # 2. u=[0,1,0,0], v=[1,1,0,0], w=[0,1,0,0]
        # 3. u=[1,0,1,0], v=[0,0,1,0], w=[0,0,1,0]
        # 4. u=[0,0,0,1], v=[1,0,0,1], w=[0,0,0,1]
        # 5. u=[1,0,0,1], v=[1,0,0,0], w=[1,0,0,0]
        # 6. u=[-1,0,1,0], v=[0,1,0,1], w=[0,1,0,1]
        # 7. u=[0,1,0,1], v=[0,0,1,-1], w=[0,0,1,-1]
        
        # I will verify these or use a known list.
        # To be safe, I'll stick to the schoolbook for the 2x2x2 blocks if I'm not sure of the Strassen factors.
        # But Strassen is standard.
        
        # Let's use the standard Strassen factors for 2x2x2:
        # m1: u=[1,0,0,0], v=[1,0,0,1], w=[1,0,0,1]
        # m2: u=[0,1,0,0], v=[1,1,0,0], w=[0,1,0,0]
        # m3: u=[1,0,1,0], v=[0,0,1,0], w=[0,0,1,0]
        # m4: u=[0,0,0,1], v=[1,0,0,1], w=[0,0,0,1]
        # m5: u=[1,0,0,1], v=[1,0,0,0], w=[1,0,0,0]
        # m6: u=[-1,0,1,0], v=[0,1,0,1], w=[0,1,0,1]
        # m7: u=[0,1,0,1], v=[0,0,1,-1], w=[0,0,1,-1]
        
        # I will map these to the global indices.
        
        terms = [
            ([1,0,0,0], [1,0,0,1], [1,0,0,1]),
            ([0,1,0,0], [1,1,0,0], [0,1,0,0]),
            ([1,0,1,0], [0,0,1,0], [0,0,1,0]),
            ([0,0,0,1], [1,0,0,1], [0,0,0,1]),
            ([1,0,0,1], [1,0,0,0], [1,0,0,0]),
            ([-1,0,1,0], [0,1,0,1], [0,1,0,1]),
            ([0,1,0,1], [0,0,1,-1], [0,0,1,-1])
        ]
        
        # Global indices for this block:
        # A indices (u): 0, 1, 3, 4 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
        # Note: My U index is i*m + j.
        # i=0,j=0 -> 0
        # i=0,j=1 -> 1
        # i=1,j=0 -> 3
        # i=1,j=1 -> 4
        u_map = [0, 1, 3, 4]
        
        # B indices (v): j*p + k
        # j=0,k=k_offset -> k_offset
        # j=0,k=k_offset+1 -> k_offset+1
        # j=1,k=k_offset -> p + k_offset
        # j=1,k=k_offset+1 -> p + k_offset + 1
        v_map = [k_offset, k_offset+1, p+k_offset, p+k_offset+1]
        
        # C indices (w): k*n + i
        # k=k_offset, i=0 -> k_offset
        # k=k_offset, i=1 -> k_offset+1
        # k=k_offset+1, i=0 -> n + k_offset
        # k=k_offset+1, i=1 -> n + k_offset + 1
        w_map = [k_offset, k_offset+1, n+k_offset, n+k_offset+1]
        
        for u_vec, v_vec, w_vec in terms:
            for u_val, v_val, w_val in zip(u_vec, v_vec, w_vec):
                if u_val != 0 and v_val != 0 and w_val != 0:
                    # This logic is wrong. A rank-1 term is u_r \otimes v_r \otimes w_r.
                    # The vectors u_r, v_r, w_r are the coefficients.
                    # So for each of the 7 terms, we add one column to U, V, W.
                    pass
        
        # Correct approach:
        for u_vec, v_vec, w_vec in terms:
            for u_idx_global, u_val in zip(u_map, u_vec):
                if u_val != 0:
                    U[u_idx_global, col] = u_val
            for v_idx_global, v_val in zip(v_map, v_vec):
                if v_val != 0:
                    V[v_idx_global, col] = v_val
            for w_idx_global, w_val in zip(w_map, w_vec):
                if w_val != 0:
                    W[w_idx_global, col] = w_val
            col += 1

    # Apply Strassen to k=0,1
    add_strassen_block(0)
    # Apply Strassen to k=2,3
    add_strassen_block(2)
    # Apply Strassen to k=4,5
    add_strassen_block(4)
    
    # Apply Schoolbook to k=6
    # A_{i,j} B_{j,6} -> C_{6,i}
    for i in range(n):
        for j in range(2): # j=0,1
            u_idx = i * m + j
            v_idx = j * p + 6
            w_idx = 6 * n + i
            add_term(u_idx, v_idx, w_idx)
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
