"""Matrix multiplication 3x5x5 - low rank tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = 3, 5, 5
    # Start with schoolbook rank 75, then try to reduce
    # Use a systematic approach: find linear dependencies
    
    # The multiplication tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Use the fact that for fixed i, the 5x5 multiplication can be
    # decomposed more efficiently. Try rank 71 decomposition.
    
    # Build schoolbook first, then try to find a lower rank by
    # identifying and exploiting linear dependencies
    
    rank = 75
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # For each i, we have 25 terms (j,k) that all contribute to row i
    # The key is that W[k*n+i] only depends on k and i, not on j
    # So for fixed i, the terms are: sum_{j,k} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's try a different approach: use the known result that 
    # 3x5x5 can be done in rank 71
    # This saves 4 ranks from the schoolbook 75
    
    # One way: for each i, the 5x5 matrix multiply can potentially
    # share some structure. Let's try to find a rank-71 decomposition.
    
    # Actually, let me try a direct construction based on the idea that
    # we can combine some terms. For a 2x2x2 block, Strassen saves 1 rank.
    # We have floor(3/2)*floor(5/2)*floor(5/2) = 1*2*2 = 4 such 2x2x2 blocks
    # Wait, that's not quite right. Let me think more carefully.
    
    # For 3x5x5: we can partition into blocks.
    # i: 3 rows -> can do one 2x2 block in i (using rows 0,1) and row 2 separately
    # j: 5 cols -> two 2x2 blocks (cols 0,1 and 2,3) and col 4 separately  
    # k: 5 rows -> two 2x2 blocks (rows 0,1 and 2,3) and row 4 separately
    
    # So we have 1*2*2 = 4 "full" 2x2x2 blocks where Strassen applies
    # Each saves 1 rank: 75 - 4 = 71
    
    # Let me construct this explicitly.
    
    # Partition indices:
    # i: block0 = {0,1}, single = {2}
    # j: block0 = {0,1}, block1 = {2,3}, single = {4}
    # k: block0 = {0,1}, block1 = {2,3}, single = {4}
    
    # For each combination of (i_type, j_type, k_type), we handle the block.
    
    # Let me build this from scratch with the Strassen optimization.
    
    U_new = []
    V_new = []
    W_new = []
    
    def add_term(u_val, v_val, w_val):
        """Add a single rank-1 term."""
        U_new.append(u_val)
        V_new.append(v_val)
        W_new.append(w_val)
    
    # For a 2x2x2 Strassen block with indices (i0,i1), (j0,j1), (k0,k1):
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In our tensor decomposition context:
    # U[i*m+j, c] corresponds to A[i,j]
    # V[j*p+k, c] corresponds to B[j,k]  
    # W[k*n+i, c] corresponds to C[k,i]
    
    # For the 2x2x2 block, the 8 schoolbook terms are:
    # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
    # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
    
    # Strassen replaces these 8 with 7 terms.
    
    # Let me define helper for 2x2x2 Strassen block
    def strassen_222(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for a 2x2x2 block."""
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[i0*m+j0] = 1; u1[i1*m+j1] = 1
        v1[j0*p+k0] = 1; v1[j1*p+k1] = 1
        w1[k0*n+i0] = 1; w1[k1*n+i1] = 1
        # Wait, this isn't quite right. Let me reconsider.
        
        # Actually, in the tensor decomposition:
        # term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For a product (A[i,j]) * (B[j,k]), we need:
        # U[i*m+j, c] = coefficient for A[i,j]
        # V[j*p+k, c] = coefficient for B[j,k]
        # W[k*n+i, c] = coefficient for C[k,i]
        
        # For M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]):
        # This gives contributions to C[i0,j0,k0], C[i0,j0,k1], C[i1,j1,k0], C[i1,j1,k1]
        # But with specific signs based on Strassen formulas.
        
        # Let me be more careful. The Strassen algorithm computes:
        # C = A*B where A is 2x2, B is 2x2, C is 2x2
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # In our indexing: A[i,j] is the (i,j) element, B[j,k] is (j,k), C[k,i] is... 
        # Wait, the output is C[k*n+i] which is C[k,i] in (row,col) = (k,i)
        # But standard matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
        # Here we have: output[k*n+i] = sum_j A[i,j]*B[j,k]
        # So output is C[k,i] where C = B^T * A? No...
        
        # Let me re-examine: W[k*n+i, c] and the product is U[i*m+j]*V[j*p+k]*W[k*n+i]
        # = A[i,j] * B[j,k] * C_contribution[k,i]
        # So sum over j: sum_j A[i,j]*B[j,k] = (A*B)[i,k]
        # And this should equal output[k*n+i]
        # So output[k*n+i] = (A*B)[i,k], meaning the output matrix is (A*B)^T
        # i.e., output[r,c] = (A*B)[c,r] where r=k, c=i
        
        # So if I think of the output as D[k,i] = (A*B)[i,k] = (A*B)^T[k,i]
        # Then D = (A*B)^T
        
        # For Strassen on 2x2: if A' is 2x2 (rows i0,i1; cols j0,j1)
        # B' is 2x2 (rows j0,j1; cols k0,k1)
        # Then C' = A'*B' is 2x2 (rows i0,i1; cols k0,k1)
        # And D' = C'^T is 2x2 (rows k0,k1; cols i0,i1)
        # D'[k0-i0, i0-i0] = C'[i0-i0, k0-k0] = C'[0,0]
        # D'[k0-i0, i1-i0] = C'[i1-i0, k0-k0] = C'[1,0]
        # D'[k1-i0, i0-i0] = C'[i0-i0, k1-k0] = C'[0,1]
        # D'[k1-i0, i1-i0] = C'[i1-i0, k1-k0] = C'[1,1]
        
        # So in terms of D (which is our output):
        # D[k0,i0] = C'[0,0] = M1 + M4 - M5 + M7
        # D[k0,i1] = C'[1,0] = M2 + M4
        # D[k1,i0] = C'[0,1] = M3 + M5
        # D[k1,i1] = C'[1,1] = M1 - M2 + M3 + M6
        
        # Where:
        # M1 = (A'[0,0]+A'[1,1])*(B'[0,0]+B'[1,1]) = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # M2 = (A'[1,0]+A'[1,1])*B'[0,0] = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        # M3 = A'[0,0]*(B'[1,0]-B'[1,1]) = A[i0,j0]*(B[j1,k0]-B[j1,k1])
        # M4 = A'[1,1]*(B'[0,1]-B'[0,0]) = A[i1,j1]*(B[j0,k1]-B[j0,k0])
        # M5 = (A'[0,0]+A'[0,1])*B'[1,1] = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        # M6 = (A'[1,0]-A'[0,0])*(B'[0,0]+B'[1,0]) = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j1,k0])
        # M7 = (A'[0,1]-A'[1,1])*(B'[0,1]+B'[1,1]) = (A[i0,j1]-A[i1,j1])*(B[j0,k1]+B[j1,k1])
        
        # Now I can write out each M as a rank-1 term in our decomposition.
        
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # U: A[i0,j0]=1, A[i1,j1]=1
        # V: B[j0,k0]=1, B[j1,k1]=1
        # W: contributes to D[k0,i0], D[k0,i1], D[k1,i0], D[k1,i1] with coefficient 1
        # From above: D[k0,i0]+=M1, D[k0,i1]+=0, D[k1,i0]+=0, D[k1,i1]+=M1
        # Wait no, let me re-read:
        # D[k0,i0] = M1 + M4 - M5 + M7  -> M1 contributes +1 to W[k0*n+i0]
        # D[k0,i1] = M2 + M4             -> M1 contributes 0
        # D[k1,i0] = M3 + M5             -> M1 contributes 0
        # D[k1,i1] = M1 - M2 + M3 + M6  -> M1 contributes +1 to W[k1*n+i1]
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        # D[k0,i0]: 0, D[k0,i1]: +1, D[k1,i0]: 0, D[k1,i1]: -1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        add_term(u, v, w)
        
        # M3: A[i0,j0]*(B[j1,k0]-B[j1,k1])
        # D[k0,i0]: 0, D[k0,i1]: 0, D[k1,i0]: +1, D[k1,i1]: +1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0] = 1
        v[j1*p+k0] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M4: A[i1,j1]*(B[j0,k1]-B[j0,k0])
        # D[k0,i0]: +1, D[k0,i1]: +1, D[k1,i0]: 0, D[k1,i1]: 0
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i1*m+j1] = 1
        v[j0*p+k1] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        add_term(u, v, w)
        
        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        # D[k0,i0]: -1, D[k0,i1]: 0, D[k1,i0]: +1, D[k1,i1]: 0
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = -1; w[k1*n+i0] = 1
        add_term(u, v, w)
        
        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j1,k0])
        # D[k0,i0]: 0, D[k0,i1]: 0, D[k1,i0]: 0, D[k1,i1]: +1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j1*p+k0] = 1
        w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M7: (A[i0,j1]-A[i1,j1])*(B[j0,k1]+B[j1,k1])
        # D[k0,i0]: +1, D[k0,i1]: 0, D[k1,i0]: 0, D[k1,i1]: 0
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j0*p+k1] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        add_term(u, v, w)
    
    # Now build the full decomposition
    # i blocks: {0,1} and {2}
    # j blocks: {0,1}, {2,3}, and {4}
    # k blocks: {0,1}, {2,3}, and {4}
    
    i_blocks = [(0,1), (2,2)]  # (start, end) inclusive, but (2,2) is single
    j_blocks = [(0,1), (2,3), (4,4)]
    k_blocks = [(0,1), (2,3), (4,4)]
    
    # For each combination of i_block, j_block, k_block:
    # If all are size 2, use Strassen (7 terms)
    # Otherwise, use schoolbook for that sub-block
    
    for (i0, i1) in i_blocks:
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                if i0 < i1 and j0 < j1 and k0 < k1:
                    # 2x2x2 block, use Strassen
                    strassen_222(i0, i1, j0, j1, k0, k1)
                else:
                    # Schoolbook for this sub-block
                    for i in range(i0, i1+1):
                        for j in range(j0, j1+1):
                            for k in range(k0, k1+1):
                                u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                                u[i*m+j] = 1
                                v[j*p+k] = 1
                                w[k*n+i] = 1
                                add_term(u, v, w)
    
    U_final = np.array(U_new)
    V_final = np.array(V_new)
    W_final = np.array(W_new)
    
    # Verify
    # Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(len(U_new)):
                    val += U_final[i*m+j, c] * V_final[j*p+k, c] * W_final[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
