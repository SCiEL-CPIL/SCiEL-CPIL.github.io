"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Part 1: Strassen 4x4x4 (rank 56)
    # Part 2: Schoolbook 4x4x1 (rank 16)
    # Total rank 72
    
    rank_part1 = 56
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    # A (4x4) * b (4x1)
    # U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
    # Here k=4, p=5, n=4
    # V index: j*5 + 4
    # W index: 4*4 + i = 16 + i
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4
    
    # --- Part 1: Strassen 4x4x4 ---
    # Blocks of A and B are 2x2
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Strassen products M1..M7
    # Each Mk is 2x2 matrix product, rank 8 (schoolbook)
    
    # Define the linear combinations for A and B for each Mk
    # Mk = Lk(A) * Rk(B)
    
    # We will iterate over the 7 products
    # For each product, we have 8 terms (schoolbook 2x2x2)
    
    # Helper to get global index for A block element
    # Block bi (0..3) is 2x2.
    # bi=0 -> rows 0-1, cols 0-1
    # bi=1 -> rows 0-1, cols 2-3
    # bi=2 -> rows 2-3, cols 0-1
    # bi=3 -> rows 2-3, cols 2-3
    
    def a_block_idx(bi, u, v):
        # bi: 0,1,2,3
        # u, v: 0,1
        row = (bi // 2) * 2 + u
        col = (bi % 2) * 2 + v
        return row * m + col
    
    def b_block_idx(bi, u, v):
        # bi: 0,1,2,3
        # u, v: 0,1
        row = (bi // 2) * 2 + u
        col = (bi % 2) * 2 + v
        return row * p + col  # Wait, B is 4x5. But we are using 4x4 part.
        # For the 4x4 part, B is 4x4.
        # The columns of B are 0..3.
        # So col index is just col.
        # But V index is j*p + k.
        # Here j is row of B (0..3), k is col of B (0..3).
        # So V index is j*5 + k.
        
    def c_block_idx(bi, u, v):
        # bi: 0,1,2,3
        # u, v: 0,1
        # Output C is 4x4.
        # W index is k*n + i.
        # Here k is col of C (0..3), i is row of C (0..3).
        # C block bi corresponds to rows (bi//2)*2 + u, cols (bi%2)*2 + v.
        # So i = (bi//2)*2 + u
        # k = (bi%2)*2 + v
        return k * n + i

    # Strassen definitions
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output combination
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map the local 2x2 indices of the Mk products to the global indices.
    
    # For each Mk, we have a schoolbook multiplication.
    # Local indices:
    # U_local: 4 entries (2x2)
    # V_local: 4 entries (2x2)
    # W_local: 4 entries (2x2)
    
    # Schoolbook 2x2x2:
    # Term t (0..7) corresponds to (u,v) in U, (u,v) in V?
    # No, schoolbook:
    # C_uv = sum_w A_uw B_wv
    # Terms: (u,w,v)
    # U index: u*2 + w
    # V index: w*2 + v
    # W index: v*2 + u (Note: W convention is k*n+i. For 2x2 block, local i=u, local k=v?
    # Wait, the global W index is k*n+i.
    # For the 4x4 output, the block C_bi has rows i, cols k.
    # Inside the block, local row u, local col v.
    # Global i = block_row * 2 + u
    # Global k = block_col * 2 + v
    # So W index = k*4 + i.
    
    # Let's build the tensors for each Mk.
    
    for mk in range(7):
        # Determine the linear combinations for A and B
        # Lk(A): dict of (local_u, local_v) -> list of (global_a_idx, coeff)
        # Rk(B): dict of (local_u, local_v) -> list of (global_b_idx, coeff)
        
        L_A = {}
        R_B = {}
        
        if mk == 0: # M1
            # A' = A00 + A11
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(0, u, v), 1.0), (a_block_idx(3, u, v), 1.0)]
            # B' = B00 + B11
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(0, u, v), 1.0), (b_block_idx(3, u, v), 1.0)]
        elif mk == 1: # M2
            # A' = A10 + A11
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(2, u, v), 1.0), (a_block_idx(3, u, v), 1.0)]
            # B' = B00
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(0, u, v), 1.0)]
        elif mk == 2: # M3
            # A' = A00
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(0, u, v), 1.0)]
            # B' = B01 - B11
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(1, u, v), 1.0), (b_block_idx(3, u, v), -1.0)]
        elif mk == 3: # M4
            # A' = A11
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(3, u, v), 1.0)]
            # B' = B10 - B00
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(2, u, v), 1.0), (b_block_idx(0, u, v), -1.0)]
        elif mk == 4: # M5
            # A' = A00 + A01
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(0, u, v), 1.0), (a_block_idx(1, u, v), 1.0)]
            # B' = B11
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(3, u, v), 1.0)]
        elif mk == 5: # M6
            # A' = A10 - A00
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(2, u, v), 1.0), (a_block_idx(0, u, v), -1.0)]
            # B' = B00 + B01
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(0, u, v), 1.0), (b_block_idx(1, u, v), 1.0)]
        elif mk == 6: # M7
            # A' = A01 - A11
            for u in range(2):
                for v in range(2):
                    L_A[(u,v)] = [(a_block_idx(1, u, v), 1.0), (a_block_idx(3, u, v), -1.0)]
            # B' = B10 + B11
            for u in range(2):
                for v in range(2):
                    R_B[(u,v)] = [(b_block_idx(2, u, v), 1.0), (b_block_idx(3, u, v), 1.0)]
        
        # Now, for each of the 8 terms in the schoolbook multiplication of A' and B'
        # Term t corresponds to (u, w, v) in A' and B'
        # U_local index: u*2 + w
        # V_local index: w*2 + v
        # W_local index: v*2 + u (local output C'_{uv})
        
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # This term contributes to C'_{uv}
                    # The local output index is v*2 + u?
                    # No, schoolbook: C_{uv} += A_{uw} B_{wv}
                    # So for a fixed (u,w,v), it adds to C_{uv}.
                    # The local W index for C_{uv} is v*2 + u?
                    # Wait, standard matrix multiplication C = A B.
                    # C_{i,j} = sum_k A_{i,k} B_{k,j}.
                    # Here local i=u, local j=v, local k=w.
                    # So C_{u,v} gets contribution from A_{u,w} B_{w,v}.
                    # The local W index corresponding to C_{u,v} is v*2 + u?
                    # No, the W matrix in the tensor decomposition maps to the output.
                    # In the schoolbook tensor, for each term (i,k,j),
                    # U has 1 at i*2+k, V has 1 at k*2+j, W has 1 at j*2+i.
                    # So for term (u,w,v),
                    # U_local = u*2 + w
                    # V_local = w*2 + v
                    # W_local = v*2 + u
                    
                    col = mk * 8 + (u * 4 + w * 2 + v)
                    
                    # Fill U
                    # U_local = u*2 + w
                    # This corresponds to A'_{u,w}
                    # A'_{u,w} is a linear combination of global A elements.
                    for (ga_idx, coeff) in L_A[(u, w)]:
                        U[ga_idx, col] += coeff
                    
                    # Fill V
                    # V_local = w*2 + v
                    # This corresponds to B'_{w,v}
                    for (gb_idx, coeff) in R_B[(w, v)]:
                        V[gb_idx, col] += coeff
                    
                    # Fill W
                    # W_local = v*2 + u
                    # This corresponds to C'_{u,v}
                    # We need to map C'_{u,v} to the global C.
                    # C' is the result of Mk.
                    # Mk contributes to the global C blocks.
                    
                    # We need to know which global C block C'_{u,v} belongs to and with what coefficient.
                    # This depends on the output combination formula.
                    
                    # Let's store the contributions to W in a temporary structure or handle them later.
                    # Actually, we can just accumulate them in a list of (mk, u, v, col)
                    # and then apply the output combination.
                    
                    # But W is a matrix. We can just add to it directly if we know the mapping.
                    # The mapping from Mk to global C is:
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # So for a given mk, u, v, the contribution to global C is:
                    # sum over global blocks bi of coeff * C_{bi, u, v}
                    
                    # Let's define the output coefficients for each mk
                    # out_coeff[mk][bi]
                    
                    if mk == 0: # M1
                        coeffs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}
                    elif mk == 1: # M2
                        coeffs = {0: 0.0, 1: 0.0, 2: 1.0, 3: -1.0}
                    elif mk == 2: # M3
                        coeffs = {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0}
                    elif mk == 3: # M4
                        coeffs = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0}
                    elif mk == 4: # M5
                        coeffs = {0: -1.0, 1: 1.0, 2: 0.0, 3: 0.0}
                    elif mk == 5: # M6
                        coeffs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}
                    elif mk == 6: # M7
                        coeffs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
                    
                    for bi, coeff in coeffs.items():
                        if coeff != 0:
                            # Global C block bi, local u, v
                            # Global i = (bi//2)*2 + u
                            # Global k = (bi%2)*2 + v
                            gi = (bi // 2) * 2 + u
                            gk = (bi % 2) * 2 + v
                            w_idx = gk * n + gi
                            W[w_idx, col] += coeff
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
