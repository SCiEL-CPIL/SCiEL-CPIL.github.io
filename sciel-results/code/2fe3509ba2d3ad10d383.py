import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 53 decomposition.
    # M_{4,4,4} part: rank 49
    # M_{4,4,1} part: rank 4
    # Total rank 53
    
    # We will build the tensors U, V, W
    # U shape: (16, R)
    # V shape: (20, R)
    # W shape: (20, R)
    
    R = 53
    U = np.zeros((16, R), dtype=float)
    V = np.zeros((20, R), dtype=float)
    W = np.zeros((20, R), dtype=float)
    
    # Helper to add a term
    # u_idx: (i, j) -> i*m + j
    # v_idx: (j, k) -> j*p + k
    # w_idx: (k, i) -> k*n + i
    # val: value for all three
    
    def add_term(i, j, k, val, col):
        U[i*m + j, col] += val
        V[j*p + k, col] += val
        W[k*n + i, col] += val
        
    # This simple add_term is for schoolbook.
    # For Strassen, we need to add linear combinations.
    # U_{idx, col} = sum of coeffs * A_{i,j}
    # V_{idx, col} = sum of coeffs * B_{j,k}
    # W_{idx, col} = sum of coeffs * C_{k,i}
    
    # Let's build the Strassen decomposition for M_{4,4,4}
    # We need to map the 7 Strassen products M1..M7 to the base 2x2 multiplications.
    # Each M_l is a 2x2 matrix product.
    # M_l = A_l * B_l
    # A_l and B_l are linear combinations of A blocks.
    
    # This is getting complex to hardcode manually without errors.
    # I will use a known rank 50 decomposition if I can find it, or stick to 53.
    # Given the risk of error in 53 construction, I will try to search for a rank 50 or 53 decomposition using a simple optimizer if possible, but 120s is tight.
    
    # Let's try to return the schoolbook rank 80 if I can't be sure about 53.
    # But 53 is a big jump.
    # I will implement the 53 construction carefully.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we apply this to 2x2 blocks.
    # Let A blocks be A00, A01, A10, A11 (each 2x2)
    # B blocks be B00, B01, B10, B11 (each 2x2)
    # C blocks be C00, C01, C10, C11 (each 2x2)
    
    # We need to decompose each M_l (2x2 product) into 7 scalar products (Strassen) or 4 (schoolbook).
    # Using Strassen for M_l gives rank 7 for each M_l.
    # Total 7 * 7 = 49.
    
    # I will implement a recursive function to build the decomposition.
    
    def build_strassen(n):
        # Returns U, V, W for M_{n, n, n}
        if n == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            return U, V, W
            
        # n is even
        half = n // 2
        # Split indices
        # A is n x n. Blocks are half x half.
        # We need to construct the 7 M_l terms.
        # Each M_l is a product of two n/2 x n/2 matrices.
        # We can recursively build the decomposition for M_{n/2, n/2, n/2}.
        
        U_sub, V_sub, W_sub = build_strassen(half)
        rank_sub = U_sub.shape[1]
        
        # We need to define the linear combinations for A and B for each of the 7 terms.
        # M1: A1 = A00+A11, B1 = B00+B11
        # M2: A2 = A10+A11, B2 = B00
        # M3: A3 = A00, B3 = B01-B11
        # M4: A4 = A11, B4 = B10-B00
        # M5: A5 = A00+A01, B5 = B11
        # M6: A6 = A10-A00, B6 = B00+B01
        # M7: A7 = A01-A11, B7 = B10+B11
        
        # And C combinations:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will create U, V, W of size (n*n, 7*rank_sub)
        U_full = np.zeros((n*n, 7*rank_sub))
        V_full = np.zeros((n*n, 7*rank_sub))
        W_full = np.zeros((n*n, 7*rank_sub))
        
        # Indices for blocks
        # Block (bi, bj) corresponds to rows bi*half to (bi+1)*half, cols bj*half to (bj+1)*half
        # Index in U for A_{i, j} is i*n + j? No, U is (n*m, R). Here m=n.
        # So U index is i*n + j.
        
        def get_block_indices(bi, bj):
            rows = range(bi*half, (bi+1)*half)
            cols = range(bj*half, (bj+1)*half)
            indices = []
            for r in rows:
                for c in cols:
                    indices.append(r*n + c)
            return indices
            
        # Define the 7 terms
        # Each term has A_block_coeffs and B_block_coeffs
        # A_block_coeffs is a list of (block_id, coeff)
        # block_id is (bi, bj)
        
        terms = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # C_coeffs is a list of (block_id, coeff) for C
            # M1
            ( [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)] ), # C00 gets M1, C11 gets M1
            # M2
            ( [((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)] ), # C10 gets M2, C11 gets -M2
            # M3
            ( [((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)] ), # C01 gets M3, C11 gets M3
            # M4
            ( [((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)] ), # C00 gets M4, C10 gets M4
            # M5
            ( [((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)] ), # C00 gets -M5, C01 gets M5
            # M6
            ( [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)] ), # C11 gets M6
            # M7
            ( [((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)] ) # C00 gets M7
        ]
        
        # Wait, the C_coeffs in the tuple above are for the contribution of M_l to the C blocks.
        # I need to distribute the rank_sub terms of the sub-decomposition to the full U, V, W.
        
        for l in range(7):
            A_coeffs = terms[l][0]
            B_coeffs = terms[l][1]
            C_coeffs = terms[l][2]
            
            # For each sub-term r in rank_sub
            for r in range(rank_sub):
                col = l * rank_sub + r
                
                # Fill U
                for (bi, bj), coeff in A_coeffs:
                    indices = get_block_indices(bi, bj)
                    for idx in indices:
                        U_full[idx, col] += coeff * U_sub[idx % (half*half), r]
                        
                # Fill V
                for (bi, bj), coeff in B_coeffs:
                    indices = get_block_indices(bi, bj)
                    for idx in indices:
                        V_full[idx, col] += coeff * V_sub[idx % (half*half), r]
                        
                # Fill W
                for (bi, bj), coeff in C_coeffs:
                    indices = get_block_indices(bi, bj)
                    for idx in indices:
                        W_full[idx, col] += coeff * W_sub[idx % (half*half), r]
                        
        return U_full, V_full, W_full
        
    # Build M_{4,4,4}
    U44, V44, W44 = build_strassen(4)
    
    # Now add the M_{4,4,1} part
    # This is A (4x4) * B_col (4x1) -> C_col (4x1)
    # We can use schoolbook for this: 4 terms.
    # A_{i,j} * B_{j,4} -> C_{i,4}
    # Note: B index j*p + k. Here k=4.
    # C index k*n + i. Here k=4.
    
    U_full = np.zeros((16, 53))
    V_full = np.zeros((20, 53))
    W_full = np.zeros((20, 53))
    
    # Copy M_{4,4,4} part
    # U44 is (16, 49), V44 is (16, 49), W44 is (16, 49)
    # But V and W for M_{4,4,4} only use k=0..3.
    # V44 index j*4 + k.
    # We need to map V44 (16x49) to V_full (20x53).
    # V_full index j*5 + k.
    # For k=0..3, V_full[j*5 + k, col] = V44[j*4 + k, col].
    
    for j in range(4):
        for k in range(4):
            V_full[j*5 + k, 0:49] = V44[j*4 + k, 0:49]
            
    # W44 index k*4 + i.
    # W_full index k*4 + i. (n=4)
    # So W_full[0:16, 0:49] = W44.
    W_full[0:16, 0:49] = W44
    
    U_full[0:16, 0:49] = U44
    
    # Now add the 4 terms for the last column
    for j in range(4):
        col = 49 + j
        # U: A_{i,j} -> i*4 + j
        for i in range(4):
            U_full[i*4 + j, col] = 1
        # V: B_{j,4} -> j*5 + 4
        V_full[j*5 + 4, col] = 1
        # W: C_{i,4} -> 4*4 + i = 16 + i
        for i in range(4):
            W_full[16 + i, col] = 1
            
    return {"U": U_full, "V": V_full, "W": W_full}

