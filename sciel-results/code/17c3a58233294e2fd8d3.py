DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Target rank: 98 using 4+1 split with Strassen(4x4) and Schoolbook(cross terms)
    # Strassen(4x4) rank 49
    # Cross terms: 16 + 16 + 16 = 48
    # Center: 1
    # Total: 98
    
    total_rank = 98
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add a schoolbook decomposition for a sub-block
    def add_schoolblock(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

    # Helper to add a Strassen decomposition for a 2x2 block
    # Indices are relative to the block start
    def add_strassen2x2(i_start, j_start, k_start):
        nonlocal col
        # Strassen 2x2 coefficients
        # U, V, W are for the 2x2 sub-problem
        # We need to map local indices to global
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the terms as (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict of (i,j) -> coeff
        # v_coeffs: dict of (j,k) -> coeff
        # w_coeffs: dict of (i,k) -> coeff
        
        terms = []
        
        # M1
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(0,0): 1, (1,1): 1} # Wait, W is for C. C00 gets +1, C11 gets +1? No.
            # C = W * M.
            # M1 contributes to C00 (+1), C11 (+1).
            # So W coeffs: {(0,0): 1, (1,1): 1}
        ))
        # M2
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1} # C10 +1, C11 -1
        ))
        # M3
        terms.append((
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1} # C01 +1, C11 +1
        ))
        # M4
        terms.append((
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1} # C00 +1, C10 +1
        ))
        # M5
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1} # C00 -1, C01 +1
        ))
        # M6
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1} # C11 +1
        ))
        # M7
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): -1} # C00 +1, C01 -1? 
            # C00 = ... + M7. C01 = ... No.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # So M7 contributes +1 to C00.
            # My W coeff for M7 should be {(0,0): 1}
        ))
        
        # Correcting M7 W coeffs
        terms[6] = (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
        
        # Now, each term is a 2x2 multiplication? No.
        # The term is a product of two linear forms.
        # But in the global decomposition, we need to decompose this into rank-1 tensors.
        # Wait, the "term" here is one scalar multiplication in the Strassen algorithm.
        # The "linear forms" are sums of A and B entries.
        # This IS a rank-1 tensor in the context of the 2x2 sub-problem?
        # No, a rank-1 tensor for 2x2x2 is u⊗v⊗w.
        # The Strassen term is exactly such a tensor.
        # So we can just add these 7 terms directly to the global U, V, W.
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            for (li, lj), uc in u_coeffs.items():
                gi = i_start + li
                gj = j_start + lj
                U[gi * m + gj, col] += uc
            for (lj, lk), vc in v_coeffs.items():
                gj = j_start + lj
                gk = k_start + lk
                V[gj * p + gk, col] += vc
            for (li, lk), wc in w_coeffs.items():
                gi = i_start + li
                gk = k_start + lk
                W[gk * n + gi, col] += wc
            col += 1

    # 1. A44 B44 (4x4 block at 0,0)
    # Decompose 4x4 using Strassen structure with 2x2 Strassen leaves.
    # This means we call add_strassen2x2 for the 4 blocks of the 4x4 matrix?
    # No, Strassen 4x4 combines 4x4 blocks into 7 products of 2x2 matrices.
    # We need to implement Strassen 4x4.
    
    # Let's use a simpler approach: Just use Schoolbook for 4x4 (Rank 64) 
    # to ensure correctness and simplicity, then optimize if time permits.
    # Rank 113 is a valid improvement.
    
    # Actually, let's try to implement Strassen 4x4 properly.
    # It's complex. Let's stick to Schoolbook 4x4 for now.
    
    # Block 1: 4x4 at (0,0)
    add_schoolblock(0, 4, 0, 4, 0, 4) # Rank 64
    
    # Block 2: 4x1x4 (A41 B14)
    # i in 0..3, j=4, k in 0..3
    add_schoolblock(0, 4, 4, 5, 0, 4) # Rank 16
    
    # Block 3: 4x4x1 (A44 B41)
    # i in 0..3, j in 0..3, k=4
    add_schoolblock(0, 4, 0, 4, 4, 5) # Rank 16
    
    # Block 4: 1x4x4 (A14 B44)
    # i=4, j in 0..3, k in 0..3
    add_schoolblock(4, 5, 0, 4, 0, 4) # Rank 16
    
    # Block 5: 1x1x1 (A11 B11)
    # i=4, j=4, k=4
    add_schoolblock(4, 5, 4, 5, 4, 5) # Rank 1
    
    # Total Rank used: 64 + 16 + 16 + 16 + 1 = 113
    # We allocated 98, so we need to resize or adjust.
    # Let's just return the matrices with the actual rank used.
    
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
