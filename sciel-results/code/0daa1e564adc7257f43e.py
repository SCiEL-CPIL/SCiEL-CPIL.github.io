"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# Known Strassen-style patterns for 2x2x2
# Each pattern is a list of 7 tuples (a1,a2,b1,b2,c1,c2) where:
# A is [[a1,a2],[b1,b2]], B is [[c1,c2],[d1,d2]] but we need consistent indexing
# For 2x2x2: A[i,j], B[j,k], C[i,k]
# Strassen formulas for [[a,b],[c],[d]] * [[e,f],[g,h]]:
# m1 = (a+d)(e+h)
# m2 = (c+d)e  
# m3 = a(f-h)
# m4 = d(g-c)
# m5 = (a+b)h
# m6 = (c-d)(f+h)
# m7 = (a-c)(e+g)
# p1 = m1+m4-m5-m7 = a+dh - (a+b)h - (a-c)(e+g) = ae - bh... let's just use the standard

def construct():
    n, m, p = 3, 4, 8
    rank_nm = n * m
    rank_mp = m * p
    rank_pn = p * n
    rank_nmp = n * m * p
    
    # Initialize schoolbook decomposition
    best_U = np.zeros((rank_nm, rank_nmp), dtype=float)
    best_V = np.zeros((rank_mp, rank_nmp), dtype=float)
    best_W = np.zeros((rank_pn, rank_nmp), dtype=float)
    
    for col in range(rank_nmp):
        i = col // (m * p)
        j = (col // p) % m
        k = col % p
        best_U[i * m + j, col] = 1.0
        best_V[j * p + k, col] = 1.0
        best_W[k * n + i, col] = 1.0
    
    best_rank = rank_nmp
    
    # Deadline guard
    deadline = time.time() + 100  # Leave 20s margin
    
    # Try to find better decompositions using Strassen on blocks
    # Strategy: partition the matrices into blocks and apply Strassen
    
    # For 3x4x8, we can try various block structures
    # Let's try 2x2 block structure where we group rows of A and columns of B
    
    # Block structure: split i into 2 groups, j into 2 groups, k into 2 groups
    # Group sizes: i: (2,1), j: (2,2), k: (4,4)
    
    def try_block_strassen(i_split, j_split, k_split):
        """Try Strassen decomposition with given block splits."""
        # i_split: tuple of sizes for n rows
        # j_split: tuple of sizes for m cols of A / rows of B
        # k_split: tuple of sizes for p cols of B
        
        # Verify splits
        if sum(i_split) != n or sum(j_split) != m or sum(k_split) != p:
            return None
        
        # Check if all splits are 2 (needed for Strassen)
        if len(i_split) != 2 or len(j_split) != 2 or len(k_split) != 2:
            return None
        
        n1, n2 = i_split
        m1, m2 = j_split
        p1, p2 = k_split
        
        # Create 2x2 block indices
        # Block (bi, bj) of A has shape (n_bi, m_bj)
        # Block (bj, bk) of B has shape (m_bj, p_bk)
        # Block (bi, bk) of C has shape (n_bi, p_bk)
        
        # For each of the 7 Strassen multiplications, we need to define
        # how the blocks are combined
        
        # Strassen formulas (for 2x2):
        # Let A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # For each Strassen multiplication, we need to create rank contributions
        # Each Strassen multiplication involves multiplying two matrices
        # which themselves may be sums of blocks
        
        # Let's define the Strassen patterns
        # Each pattern is (coef_A_blocks, coef_B_blocks, coef_C_blocks)
        # where coef_A_blocks is a dict mapping (bi, bj) to coefficient
        # Similarly for B and C
        
        # M1 = (A11 + A22)(B11 + B22)
        # This means: U has contributions from A11 and A22 with coef 1
        # V has contributions from B11 and B22 with coef 1
        # C contribution: C11 += 1, C22 += 1
        
        strassen_patterns = [
            # (name, A_coeffs, B_coeffs, C_coeffs)
            # M1
            ("M1", {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ("M2", {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # M3
            ("M3", {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # M4
            ("M4", {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,1):1}),
            # M5
            ("M5", {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1}),
            # M6
            ("M6", {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1}),
            # M7
            ("M7", {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
        ]
        
        # For each Strassen multiplication, we need to expand it into
        # scalar multiplications for each element in the blocks
        # This gives us rank = 7 * n1*m1*p1 + 7 * n1*m2*p2 + ... 
        # Actually, it's more complex. Let me think about this differently.
        
        # Each Strassen multiplication multiplies two matrices.
        # For M1 = (A11 + A22)(B11 + B22):
        # A11 has shape (n1, m1), A22 has shape (n2, m2)
        # B11 has shape (m1, p1), B22 has shape (m2, p2)
        # But A11 + A22 requires n1 == n2 and m1 == m2 for them to add!
        
        # So we can only use this if the block sizes match appropriately.
        # For simplicity, let's require equal block sizes.
        
        if not (n1 == n2 and m1 == m2 and p1 == p2):
            return None
        
        nb = n1  # block size
        mb = m1
        pb = p1
        
        # Now each block is nb x mb, mb x pb, or nb x pb
        # Total rank for 7 Strassen multiplications:
        # Each multiplication: nb*mb * mb*pb = nb*mb^2*pb scalar products
        # 7 multiplications: 7 * nb * mb^2 * pb
        
        strassen_rank = 7 * nb * mb * pb
        
        if strassen_rank >= best_rank:
            return None
        
        # Build the decomposition
        U_new = np.zeros((rank_nm, strassen_rank), dtype=float)
        V_new = np.zeros((rank_mp, strassen_rank), dtype=float)
        W_new = np.zeros((rank_pn, strassen_rank), dtype=float)
        
        col = 0
        for sname, A_coeffs, B_coeffs, C_coeffs in strassen_patterns:
            # Expand this Strassen multiplication into scalar products
            # A side: sum of blocks with coefficients
            # B side: sum of blocks with coefficients
            # For each (i, j) in A-blocks and (j', k) in B-blocks where j matches...
            # Actually, the multiplication is (sum of A blocks) * (sum of B blocks)
            # = sum over all pairs of (A_block * B_block)
            
            # Get the list of A blocks involved
            A_blocks = []  # list of (bi, bj, coef)
            for (bi, bj), coef in A_coeffs.items():
                A_blocks.append((bi, bj, coef))
            
            B_blocks = []  # list of (bj, bk, coef)
            for (bj, bk), coef in B_coeffs.items():
                B_blocks.append((bj, bk, coef))
            
            # For each pair of A block and B block, we need j to match
            # A block (bi, bj) has shape (n_bi, m_bj)
            # B block (bj, bk) has shape (m_bj, p_bk)
            # They can be multiplied if they share the same j index
            
            # Since all blocks are the same size, let's just iterate
            # over all elements
            
            for (bi_A, bj_A, coef_A) in A_blocks:
                for (bj_B, bk_B, coef_B) in B_blocks:
                    if bj_A != bj_B:
                        continue
                    # This pair contributes to the multiplication
                    # A block (bi_A, bj_A) * B block (bj_B, bk_B)
                    # Result: nb x pb matrix
                    # We need to add this to the C blocks specified in C_coeffs
                    # But C_coeffs tells us which C blocks get contributions
                    # from THIS Strassen multiplication (M1, M2, etc.)
                    # The coefficient is already in the C_coeffs
                    
                    # For each element in the resulting nb x pb block
                    for ii in range(nb):
                        for kk in range(pb):
                            # This scalar product contributes to W
                            # The A element: row = bi_A*nb + ii, col = bj_A*mb + jj
                            # Wait, we need to iterate over jj too
                            # Actually, for a fixed (ii, kk), the contribution
                            # comes from summing over jj
                            # But in a rank-1 decomposition, each scalar
                            # product is one rank term
                            
                            # Let me reconsider. For the product of two matrices,
                            # C[i,k] = sum_j A[i,j] B[j,k]
                            # Each (i,j,k) triple is one scalar multiplication
                            # In the rank decomposition, each such scalar product
                            # becomes one rank term
                            
                            pass
            
            # This approach is getting complex. Let me use a simpler method.
            # For each Strassen multiplication, expand all scalar products
            # directly.
            
            # Clear and redo
            U_new[:, col:col + nb*mb*pb] = 0
            V_new[:, col:col + nb*mb*pb] = 0
            W_new[:, col:col + nb*mb*pb] = 0
            
            inner_col = col
            for (bi_A, bj_A, coef_A) in A_blocks:
                for ii in range(nb):
                    for jj in range(mb):
                        # A element at global position (bi_A*nb + ii, bj_A*mb + jj)
                        for (bj_B, bk_B, coef_B) in B_blocks:
                            if bj_A != bj_B:
                                continue
                            for kk in range(pb):
                                # B element at global position (bj_B*mb + jj, bk_B*pb + kk)
                                # This scalar product contributes to C blocks
                                # The result goes to C block (bi_A, bk_B) at position (ii, kk)
                                # But we need to check C_coeffs to see if this
                                # Strassen multiplication contributes to C(bi_A, bk_B)
                                
                                c_coef = 0
                                for (bi_C, bk_C), cc in C_coeffs.items():
                                    if bi_C == bi_A and bk_C == bk_B:
                                        c_coef = cc
                                        break
                                
                                if c_coef == 0:
                                    continue
                                
                                # Add this scalar product to the decomposition
                                U_new[bi_A*nb + ii, inner_col] = coef_A
                                V_new[bj_B*mb + jj, inner_col] = coef_B
                                # W: C element at (bi_A*nb + ii, bk_B*pb + kk)
                                # But wait, C is indexed as C[k*n + i] per the convention
                                # W[k*n + i, r]
                                # So W index = kk * n + (bi_A*nb + ii)
                                W_new[kk * n + bi_A*nb + ii, inner_col] = c_coef
                                inner_col += 1
            
            col = inner_col
        
        # Verify
        U_final = U_new[:, :col]
        V_final = V_new[:, :col]
        W_final = W_new[:, :col]
        
        # Check if this is a valid decomposition
        # For random A, B, compute C via standard and via decomposition
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        
        # Compute via decomposition
        C_decomp = np.zeros((n, p))
        for r in range(col):
            u_val = sum(U_final[i*m+j, r] * A_test[i, j] for i in range(n) for j in range(m))
            v_val = sum(V_final[j*p+k, r] * B_test[j, k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += u_val * v_val * W_final[k*n+i, r]
        
        if np.allclose(C_test, C_decomp, atol=1e-9):
            return (U_final, V_final, W_final, col)
        else:
            return None
    
    # Try various block splits
    block_splits = [
        ((2, 1), (2, 2), (4, 4)),
        ((1, 2), (2, 2), (4, 4)),
        ((2, 1), (1, 3), (4, 4)),
        ((1, 2), (1, 3), (4, 4)),
    ]
    
    for split in block_splits:
        result = try_block_strassen(*split)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < best_rank:
                best_U = U_new
                best_V = V_new
                best_W = W_new
                best_rank = new_rank
                print(f"Found decomposition with rank {best_rank}")
        
        if time.time() > deadline:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
