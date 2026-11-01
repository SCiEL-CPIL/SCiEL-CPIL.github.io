"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # We want to decompose the matrix multiplication tensor of order (n, m, p) = (2, 2, 6)
    # The tensor is T[i, j, k] = delta_{i,j,k} where multiplication is C = A * B
    # A is n x m (2x2), B is m x p (2x6), C is n x p (2x6)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor entries are T[i, j, k] = 1 if we consider the standard multiplication
    # Actually, the tensor for matrix multiplication has entries that are 0 or 1.
    # For the (n,m,p) matrix multiplication tensor, the entries are:
    # T[i, j, k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # Wait, let me think about this more carefully.
    #
    # The matrix multiplication tensor M_{n,m,p} is defined such that:
    # (A * B)[i,k] = sum_j A[i,j] B[j,k] = sum_{j,r} U[i,j,r] V[j,k,r] W[i,k,r]
    # Wait, the convention is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta condition
    # 
    # Actually, looking at the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So for each (i,j,k), we have one rank-1 term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # And C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    # Wait, that's not right either. Let me re-read.
    #
    # The decomposition says: for the multiplication tensor, we need
    # sum_r U[a, r] V[b, r] W[c, r] = T[a,b,c]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # And T[a,b,c] = 1 if this combination corresponds to a valid multiplication term, 0 otherwise.
    #
    # In the schoolbook, for each (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i, T=1
    # All other combinations have T=0.
    #
    # Now, for 2x2 * 2x6: n=2, m=2, p=6
    # U has 4 rows (i*m+j for i in {0,1}, j in {0,1})
    # V has 12 rows (j*p+k for j in {0,1}, k in {0..5})
    # W has 12 rows (k*n+i for k in {0..5}, i in {0,1})
    #
    # Strassen's idea: for 2x2 * 2x2, we can do it in rank 7 instead of 8.
    # Here we have 2x2 * 2x6, which can be seen as three 2x2 * 2x2 blocks? No, that's not right.
    # 2x2 * 2x6: the output is 2x6. We can think of this as multiplying a 2x2 matrix by 
    # three 2x2 matrices (the columns of B grouped by pairs), but that gives 3 independent 2x2 mults.
    # Each 2x2 * 2x2 can be done in rank 7, so total rank = 3*7 = 21.
    # 
    # Alternatively, we can use a more clever approach. Let me think about the structure.
    #
    # Actually, let's use the known result: the rank of (2,2,6) matrix multiplication.
    # This is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.
    # We can view the 2x6 matrix as having 3 column-pairs, each of size 2x2.
    # But actually, we can do better by sharing computations.
    #
    # Let me try a rank-16 approach using Strassen-like ideas.
    #
    # For 2x2 * 2x2, Strassen uses 7 multiplications.
    # For 2x2 * 2x6, if we treat it as 3 separate 2x2 * 2x2 problems, we need 21.
    # But we can share some computations.
    #
    # Let me just implement the 3-block Strassen approach with rank 21, which is known to work.
    #
    # For each block s = 0, 1, 2 (corresponding to columns 2s, 2s+1 of B):
    # A is 2x2, B_s is 2x2 (columns 2s, 2s+1 of B)
    # Strassen for 2x2*2x2:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # But in our tensor decomposition, we need to express this in terms of U, V, W matrices.
    # This is getting complex. Let me just use the rank-21 schoolbook-with-Strassen-blocks.
    
    # Actually, let me just return the schoolbook rank-24 for now, but that gives score 1.0.
    # The task says score is the rank, and seed is 24. We need to do better.
    
    # Let me try to construct a rank-21 decomposition using Strassen on 3 independent 2x2 blocks.
    
    rank = 21
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each block s (0, 1, 2), we use 7 Strassen multiplications
    # Block s corresponds to B columns (2s, 2s+1)
    # In the V matrix, V[j*p + k, r] where k = 2s or 2s+1
    # In the W matrix, W[k*n + i, r] where k = 2s or 2s+1
    
    for s in range(3):
        k0 = 2 * s
        k1 = 2 * s + 1
        base = 7 * s  # base index in the rank dimension for this block
        
        # Strassen 7 multiplications for 2x2 * 2x2
        # A indices: i in {0,1}, j in {0,1} -> U[i*m+j, r]
        # B_s indices: j in {0,1}, k in {k0, k1} -> V[j*p+k, r]
        # C indices: i in {0,1}, k in {k0, k1} -> W[k*n+i, r]
        
        # M0 = (A00 + A11)(B00 + B11)
        # U[0, base+0] += 1, U[3, base+0] += 1
        # V[0, base+0] += 1, V[p+1, base+0] += 1  # V[0*p+0], V[1*p+1]
        # W[k0, base+0] += 1, W[k1+1, base+0] += 1  # W[k0*n+0], W[k1*n+1]
        # C00 += M0: W[k0*n+0, base+0] += 1
        # C11 += M0: W[k1*n+1, base+0] += 1
        
        # This is getting too complex to do manually. Let me use a different approach.
        # 
        # Let me just verify that rank 21 is achievable and construct it properly.
        # 
        # Actually, I realize I should just use a known good decomposition. Let me search my memory.
        #
        # For (2,2,6) matrix multiplication, the rank is known to be at most 21 (by 3x Strassen).
        # Whether it can be done lower is an open question or at least non-trivial.
        #
        # Let me just implement the 3-Strassen-block construction carefully.
        
        # Strassen formulas for 2x2 * 2x2:
        # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = (a10 + a11) * b00
        # M2 = a00 * (b01 - b11)
        # M3 = a11 * (b10 - b00)
        # M4 = (a00 + a01) * b11
        # M5 = (a10 - a00) * (b00 + b01)
        # M6 = (a01 - a11) * (b10 + b11)
        # M7 = (a00 + a11) * (b00 + b11)
        #
        # C00 = M7 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M7 - M1 + M2 + M5
        
        # For block s:
        # A is the full 2x2 matrix A (shared across blocks)
        # B_s = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        # C_s = [[C[0,k0], C[0,k1]], [C[1,k0], C[1,k1]]]
        
        # In our indexing:
        # A[i,j] -> U[i*m + j, r]
        # B[j,k] -> V[j*p + k, r]
        # C[i,k] -> W[k*n + i, r]
        
        # M1 = (A[1,0] + A[1,1]) * B[0,k0]
        # U[1*2+0, base+0] += 1  -> U[2, base+0] += 1
        # U[1*2+1, base+0] += 1  -> U[3, base+0] += 1
        # V[0*6+k0, base+0] += 1 -> V[k0, base+0] += 1
        # W contribution: C[1,k0] += M1 -> W[k0*2+1, base+0] += 1
        U[2][base+0] += 1
        U[3][base+0] += 1
        V[k0][base+0] += 1
        W[k0*2+1][base+0] += 1
        
        # M2 = A[0,0] * (B[0,k1] - B[1,k1])
        # U[0, base+1] += 1
        # V[k1, base+1] += 1  (B[0,k1])
        # V[6+k1, base+1] -= 1  (B[1,k1])
        # W[k1*2+0, base+1] += 1  (C[0,k1] += M2)
        U[0][base+1] += 1
        V[k1][base+1] += 1
        V[6+k1][base+1] -= 1
        W[k1*2+0][base+1] += 1
        
        # M3 = A[1,1] * (B[1,k0] - B[0,k0])
        # U[3, base+2] += 1
        # V[6+k0, base+2] += 1  (B[1,k0])
        # V[k0, base+2] -= 1  (B[0,k0])
        # W[k0*2+1, base+2] += 1  (C[1,k0] += M3)
        # W[k0*2+0, base+2] += 1  (C[0,k0] += M3)
        U[3][base+2] += 1
        V[6+k0][base+2] += 1
        V[k0][base+2] -= 1
        W[k0*2+1][base+2] += 1
        W[k0*2+0][base+2] += 1
        
        # M4 = (A[0,0] + A[0,1]) * B[1,k1]
        # U[0, base+3] += 1
        # U[1, base+3] += 1
        # V[6+k1, base+3] += 1
        # W[k1*2+0, base+3] += 1  (C[0,k1] += M4)
        # W[k0*2+0, base+3] += -1  (C[0,k0] += -M4)
        U[0][base+3] += 1
        U[1][base+3] += 1
        V[6+k1][base+3] += 1
        W[k1*2+0][base+3] += 1
        W[k0*2+0][base+3] -= 1
        
        # M5 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        # U[2, base+4] += 1
        # U[0, base+4] -= 1
        # V[k0, base+4] += 1
        # V[k1, base+4] += 1
        # W[k1*2+1, base+4] += 1  (C[1,k1] += M5)
        U[2][base+4] += 1
        U[0][base+4] -= 1
        V[k0][base+4] += 1
        V[k1][base+4] += 1
        W[k1*2+1][base+4] += 1
        
        # M6 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        # U[1, base+5] += 1
        # U[3, base+5] -= 1
        # V[6+k0, base+5] += 1
        # V[6+k1, base+5] += 1
        # W[k0*2+0, base+5] += 1  (C[0,k0] += M6)
        U[1][base+5] += 1
        U[3][base+5] -= 1
        V[6+k0][base+5] += 1
        V[6+k1][base+5] += 1
        W[k0*2+0][base+5] += 1
        
        # M7 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # U[0, base+6] += 1
        # U[3, base+6] += 1
        # V[k0, base+6] += 1
        # V[6+k1, base+6] += 1
        # W[k0*2+0, base+6] += 1  (C[0,k0] += M7)
        # W[k1*2+1, base+6] += 1  (C[1,k1] += M7)
        U[0][base+6] += 1
        U[3][base+6] += 1
        V[k0][base+6] += 1
        V[6+k1][base+6] += 1
        W[k0*2+0][base+6] += 1
        W[k1*2+1][base+6] += 1
    
    # Now let's verify the W matrix for C entries:
    # C[0,k0] = M7 + M3 - M4 + M6
    # W[k0*2+0] should have: +1 (M7), +1 (M3), -1 (M4), +1 (M6) -> net 2? That's wrong.
    # Wait, I think I made errors in the W assignments. Let me redo this more carefully.
    
    # Let me restart with a cleaner approach.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(3):
        k0 = 2 * s
        k1 = 2 * s + 1
        base = 7 * s
        
        # Define the 7 Strassen products for block s
        # Each product r in 0..6 contributes to U, V, W
        
        # We'll define each product as (U_indices_with_signs, V_indices_with_signs, W_contributions)
        # W_contributions is a list of (w_index, sign)
        
        products = []
        
        # M1 = (A10 + A11) * B0k0
        products.append((( (2, 1), (3, 1) ), ((k0, 1),), [(k0*2+1, 1)]))
        
        # M2 = A00 * (B0k1 - B1k1)
        products.append((( (0, 1), ), ((k1, 1), (6+k1, -1)), [(k1*2+0, 1)]))
        
        # M3 = A11 * (B1k0 - B0k0)
        products.append((( (3, 1), ), ((6+k0, 1), (k0, -1)), [(k0*2+1, 1), (k0*2+0, 1)]))
        
        # M4 = (A00 + A01) * B1k1
        products.append((( (0, 1), (1, 1) ), ((6+k1, 1),), [(k1*2+0, 1), (k0*2+0, -1)]))
        
        # M5 = (A10 - A00) * (B0k0 + B0k1)
        products.append((( (2, 1), (0, -1) ), ((k0, 1), (k1, 1)), [(k1*2+1, 1)]))
        
        # M6 = (A01 - A11) * (B1k0 + B1k1)
        products.append((( (1, 1), (3, -1) ), ((6+k0, 1), (6+k1, 1)), [(k0*2+0, 1)]))
        
        # M7 = (A00 + A11) * (B0k0 + B1k1)
        products.append((( (0, 1), (3, 1) ), ((k0, 1), (6+k1, 1)), [(k0*2+0, 1), (k1*2+1, 1)]))
        
        for r, (u_terms, v_terms, w_terms) in enumerate(products):
            for idx, sign in u_terms:
                U[idx][base + r] += sign
            for idx, sign in v_terms:
                V[idx][base + r] += sign
            for idx, sign in w_terms:
                W[idx][base + r] += sign
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
