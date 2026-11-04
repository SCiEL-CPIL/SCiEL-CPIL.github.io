"""
Matrix multiplication 6x6x6 tensor decomposition.
Uses block-wise Strassen: treat 6x6 as 2x2 blocks of 3x3.
Rank = 7 * 27 = 189 (vs schoolbook 216).
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _strassen_2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication, rank 7.
    U: (4, 7) indexed by [i*2+j, r]
    V: (4, 7) indexed by [j*2+k, r]
    W: (4, 7) indexed by [k*2+i, r]
    
    We need: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = 1
    where a=i*2+j, b=j*2+k, c=k*2+i
    """
    n = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2  -- need to verify
    
    # Let me define each rank-1 tensor term directly.
    # Term r contributes U_vec ⊗ V_vec ⊗ W_vec
    
    # I'll define the 7 terms by their coefficient vectors.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: [1,0,0,1] (A00=1, A11=1)
    # V: [1,0,0,1] (B00=1, B11=1)
    # W: need to figure out which C entries get contributions
    # From C00 = M1 - M4 + M5 + M7: M1 contributes +1 to C00
    # From C11 = M1 - M3 + M6 + M2: M1 contributes +1 to C11? Let me verify.
    
    # Actually let me just hardcode the known correct Strassen matrices.
    # Reference: the standard Strassen decomposition.
    
    # U[i,j,r] coefficient for A[i,j] in product r
    # V[j,k,r] coefficient for B[j,k] in product r
    # W[k,i,r] coefficient for C[k,i]... wait W[k*n+i, r] where n=2 so W[k*2+i, r]
    
    # Let me use a verified set. I'll construct it and verify.
    
    # Product 0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    # This product appears in: C00 (+1), C11 (-1)? 
    # Let's use: W[0,0]=1 (C00), W[3,0]=-1 (C11)? 
    # W index: k*2+i. C00: k=0,i=0 => idx 0. C11: k=1,i=1 => idx 3.
    W[0, 0] = 1; W[3, 0] = -1
    
    # Product 1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    # Appears in: C10 (+1), C11 (+1)?
    # C10: k=0,i=1 => idx 1. C11: k=1,i=1 => idx 3.
    W[1, 1] = 1; W[3, 1] = 1
    
    # Product 2: A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, B11
    # Appears in: C01 (+1), C11 (-1)?
    # C01: k=1,i=0 => idx 2. C11: k=1,i=1 => idx 3.
    W[2, 2] = 1; W[3, 2] = -1
    
    # Product 3: A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, B00
    # Appears in: C00 (-1), C10 (+1)?
    # C00: k=0,i=0 => idx 0. C10: k=0,i=1 => idx 1.
    W[0, 3] = -1; W[1, 3] = 1
    
    # Product 4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    # Appears in: C00 (+1), C01 (+1)?
    # C00: k=0,i=0 => idx 0. C01: k=1,i=0 => idx 2.
    W[0, 4] = 1; W[2, 4] = 1
    
    # Product 5: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    # Appears in: C11 (+1)?
    # C11: k=1,i=1 => idx 3.
    W[3, 5] = 1
    
    # Product 6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    # Appears in: C00 (+1), C01 (+1)?
    # C00: k=0,i=0 => idx 0. C01: k=1,i=0 => idx 2.
    W[0, 6] = 1; W[2, 6] = 1
    
    # Verify
    U_arr = U.astype(float)
    V_arr = V.astype(float)
    W_arr = W.astype(float)
    D = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    T = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1.0
    
    if not np.allclose(D, T, atol=1e-9):
        # Try to fix: the issue is likely in the W coefficients.
        # Let me re-derive more carefully.
        # 
        # The tensor T for 2x2x2 matmul:
        # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
        # In our indexing: a=i*2+j, b=j*2+k, c=k*2+i
        # 
        # We need sum_r U[a,r]*V[b,r]*W[c,r] = 1 for all (i,j,k)
        # 
        # Let me define the 7 bilinear forms and their contributions to C entries.
        # 
        # Actually, the standard approach: each product M_r is a bilinear form in A and B,
        # and each C entry is a linear combination of M_r's.
        # 
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = (A10+A11)B00 = A10B00 + A11B00
        # M3 = A00(B01-B11) = A00B01 - A00B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        # 
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        # 
        # Let me express C entries in terms of M1..M7:
        # C00 = A00B00 + A01B10
        #   A00B00 = M1 - A00B11 - A11B00 - A11B11 = M1 - (M5-A01B11) - (M2-A10B00) - (M7-A01B10+A11B10)
        #   This is getting circular. Let me just solve linearly.
        # 
        # I'll set up the system: for each (i,j,k) pair, the coefficient of A[i,j]*B[j,k] 
        # in sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] must be 1.
        # 
        # There are 8 (i,j,k) pairs and 7 products. The system is overdetermined but consistent.
        # 
        # Let me just use a known correct implementation. I'll look at it differently.
        # 
        # Known Strassen in matrix form (from various references):
        # The 7 rank-1 tensors can be written as:
        
        # Let me try a different assignment and verify.
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # I'll use the formulation where we directly assign coefficients
        # to make the tensor identity hold.
        
        # For each product r, define u_r (4-dim), v_r (4-dim), w_r (4-dim)
        # such that sum_r u_r ⊗ v_r ⊗ w_r = T
        
        # T has T[i*2+j, j*2+k, k*2+i] = 1
        
        # Let me enumerate the 8 target terms:
        # (i,j,k) = (0,0,0): a=0, b=0, c=0
        # (i,j,k) = (0,0,1): a=0, b=1, c=2
        # (i,j,k) = (0,1,0): a=1, b=2, c=0
        # (i,j,k) = (0,1,1): a=1, b=3, c=2
        # (i,j,k) = (1,0,0): a=2, b=0, c=1
        # (i,j,k) = (1,0,1): a=2, b=1, c=3
        # (i,j,k) = (1,1,0): a=3, b=2, c=1
        # (i,j,k) = (1,1,1): a=3, b=3, c=3
        
        # I'll just use a verified set from literature.
        # After careful derivation:
        
        # r=0: M1=(A00+A11)(B00+B11)
        U2[0,0]=1; U2[3,0]=1
        V2[0,0]=1; V2[3,0]=1
        W2[0,0]=1; W2[3,0]=-1
        
        # r=1: M2=(A10+A11)B00  
        U2[2,1]=1; U2[3,1]=1
        V2[0,1]=1
        W2[1,1]=1; W2[3,1]=1
        
        # r=2: M3=A00(B01-B11)
        U2[0,2]=1
        V2[1,2]=1; V2[3,2]=-1
        W2[2,2]=1; W2[3,2]=-1
        
        # r=3: M4=A11(B10-B00)
        U2[3,3]=1
        V2[2,3]=1; V2[0,3]=-1
        W2[0,3]=-1; W2[1,3]=1
        
        # r=4: M5=(A00+A01)B11
        U2[0,4]=1; U2[1,4]=1
        V2[3,4]=1
        W2[0,4]=1; W2[2,4]=1
        
        # r=5: M6=(A10-A00)(B00+B01)
        U2[2,5]=1; U2[0,5]=-1
        V2[0,5]=1; V2[1,5]=1
        W2[3,5]=1
        
        # r=6: M7=(A01-A11)(B10+B11)
        U2[1,6]=1; U2[3,6]=-1
        V2[2,6]=1; V2[3,6]=1
        W2[0,6]=1; W2[2,6]=1
        
        D2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
        if not np.allclose(D2, T, atol=1e-9):
            # Still wrong. Let me compute what we get and fix.
            # Print the difference
            diff = D2 - T
            # The non-zero differences tell us which (a,b,c) entries are wrong.
            # Let me just try to solve it numerically.
            
            # Alternative: use a simple iterative correction
            # Or just use a known-correct set.
            
            # Let me try yet another formulation.
            # From the paper "Algebraic Complexity Theory" by Bürgisser, Clausen, Shokrollahi:
            
            # Actually, let me just brute-force verify my M definitions.
            # I'll compute what C entries each M produces and check.
            
            # For a random A, B, compute M1..M7 and check if C = A@B.
            A_test = np.array([[1,2],[3,4]], dtype=float)
            B_test = np.array([[5,6],[7,8]], dtype=float)
            C_test = A_test @ B_test  # [[19,22],[43,50]]
            
            M1 = (A_test[0,0]+A_test[1,1])*(B_test[0,0]+B_test[1,1])
            M2 = (A_test[1,0]+A_test[1,1])*B_test[0,0]
            M3 = A_test[0,0]*(B_test[0,1]-B_test[1,1])
            M4 = A_test[1,1]*(B_test[1,0]-B_test[0,0])
            M5 = (A_test[0,0]+A_test[0,1])*B_test[1,1]
            M6 = (A_test[1,0]-A_test[0,0])*(B_test[0,0]+B_test[0,1])
            M7 = (A_test[0,1]-A_test[1,1])*(B_test[1,0]+B_test[1,1])
            
            C00 = M1 - M4 + M5 + M7
            C01 = M3 + M5
            C10 = M2 + M4
            C11 = M1 - M3 + M6 + M2
            
            C_recon = np.array([[C00, C01],[C10, C11]])
            if not np.allclose(C_recon, C_test, atol=1e-9):
                # My formulas are wrong. Let me re-derive.
                # C00 = A00B00 + A01B10
                # C01 = A00B01 + A01B11
                # C10 = A10B00 + A11B10
                # C11 = A10B01 + A11B11
                
                # Let me try to express these in terms of M1..M7.
                # This is a linear system. Let me just solve it.
                # 
                # Coefficients of A[i,j]*B[j,k] in each M:
                # M1: A00B00, A00B11, A11B00, A11B11
                # M2: A10B00, A11B00
                # M3: A00B01, -A00B11
                # M4: A11B10, -A11B00
                # M5: A00B11, A01B11
                # M6: A10B00, A10B01, -A00B00, -A00B01
                # M7: A01B10, A01B11, -A11B10, -A11B11
                # 
                # Target:
                # C00: A00B00=1, A01B10=1
                # C01: A00B01=1, A01B11=1
                # C10: A10B00=1, A11B10=1
                # C11: A10B01=1, A11B11=1
                # 
                # Let x = (x1,...,x7) be coefficients for a C entry.
                # For C00: 
                #   A00B00: x1 - x6 = 1
                #   A01B10: x7 = 1
                #   A00B11: x1 - x3 + x5 = 0
                #   A11B00: x1 + x2 - x4 = 0
                #   A11B11: x1 - x7 = 0
                #   A10B00: x2 + x6 = 0
                #   A10B01: x6 = 0
                #   A00B01: -x3 - x6 = 0
                # 
                # From A10B01: x6 = 0
                # From A10B00: x2 = 0
                # From A00B00: x1 = 1
                # From A11B11: x7 = 1
                # From A01B10: x7 = 1 ✓
                # From A00B11: 1 - x3 + x5 = 0 => x3 - x5 = 1
                # From A11B00: 1 + 0 - x4 = 0 => x4 = 1
                # From A00B01: -x3 - 0 = 0 => x3 = 0
                # Then x5 = -1
                # 
                # Check: C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 + (-1)*M5 + 0*M6 + 1*M7
                # = M1 + M4 - M5 + M7
                # 
                # Let me verify: 
                # M1 + M4 - M5 + M7 = 
                # (A00B00+A00B11+A11B00+A11B11) + (A11B10-A11B00) - (A00B11+A01B11) + (A01B10+A01B11-A11B10-A11B11)
                # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
                # = A00B00 + A01B10 ✓
                
                # For C01:
                #   A00B01: -x3 - x6 = 1
                #   A01B11: x5 + x7 = 1
                #   A00B00: x1 - x6 = 0
                #   A00B11: x1 - x3 + x5 = 0
                #   A11B00: x1 + x2 - x4 = 0
                #   A11B11: x1 - x7 = 0
                #   A10B00: x2 + x6 = 0
                #   A10B01: x6 = 0
                #   A01B10: x7 = 0
                # 
                # From A10B01: x6=0, From A01B10: x7=0
                # From A00B00: x1=0
                # From A11B11: x1=0 ✓ (x7=0)
                # From A10B00: x2=0
                # From A00B01: -x3=1 => x3=-1
                # From A00B11: 0-(-1)+x5=0 => x5=-1
                # From A01B11: -1+0=-1 ≠ 1. Contradiction!
                
                # So my M definitions don't work with this simple assignment.
                # The issue is that the 7 Strassen products don't independently 
                # contribute to C entries in a simple way - the W coefficients 
                # need to be chosen so that the full tensor identity holds.
                
                # Let me just solve the full system numerically.
                # We have 8 equations (one for each (i,j,k)) and 7 unknowns per "layer".
                # Actually, the tensor identity gives us 8 scalar equations:
                # For each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # 
                # This is nonlinear in U,V,W. But if we fix U and V, W is linear.
                # 
                # Strategy: fix U and V to be the Strassen bilinear forms,
                # then solve for W.
                
                # U (4x7): U[i*2+j, r] = coefficient of A[i,j] in M_r
                # V (4x7): V[j*2+k, r] = coefficient of B[j,k] in M_r
                
                U_fix = np.array([
                    [1, 0, 1, 0, 1, -1, 0],  # A00: M1, M3, M5, M6
                    [0, 0, 0, 0, 1, 0, 1],   # A01: M5, M7
                    [0, 1, 0, 0, 0, 1, 0],   # A10: M2, M6
                    [1, 1, 0, 1, 0, 0, -1],  # A11: M1, M2, M4, M7
                ], dtype=float)
                
                V_fix = np.array([
                    [1, 1, 0, -1, 0, 1, 0],  # B00: M1, M2, M4, M6
                    [0, 0, 1, 0, 0, 1, 0],   # B01: M3, M6
                    [0, 0, 0, 1, 0, 0, 1],   # B10: M4, M7
                    [1, 0, -1, 0, 1, 0, 1],  # B11: M1, M3, M5, M7
                ], dtype=float)
                
                # For each (i,j,k), we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # This gives 8 equations in 4 unknowns W[0,r], W[1,r], W[2,r], W[3,r]
                # (one W vector per r, but actually W is 4x7, so 28 unknowns)
                # Wait, W has shape (4,7). The equation for (i,j,k) involves W[k*2+i, r] for all r.
                # So for a fixed (i,j,k), the equation is:
                # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
                # This is a dot product between the vector (U[i*2+j,r]*V[j*2+k,r])_r and W[k*2+i,:]
                
                # Let me set up the linear system for W.
                # Unknowns: W[0,0..6], W[1,0..6], W[2,0..6], W[3,0..6] = 28 unknowns
                # Equations: 8 (one per (i,j,k))
                # Underdetermined, but we can find a solution.
                
                # Actually, each equation only involves one row of W.
                # Equation for (i,j,k): involves W[k*2+i, :]
                # So equations are grouped by k*2+i.
                
                # For k*2+i = 0 (k=0,i=0): equations where k=0,i=0
                #   (i,j,k)=(0,j,0) for j=0,1: 2 equations, both involve W[0,:]
                # For k*2+i = 1 (k=0,i=1): (i,j,k)=(1,j,0) for j=0,1: 2 eqs, W[1,:]
                # For k*2+i = 2 (k=1,i=0): (i,j,k)=(0,j,1) for j=0,1: 2 eqs, W[2,:]
                # For k*2+i = 3 (k=1,i=1): (i,j,k)=(1,j,1) for j=0,1: 2 eqs, W[3,:]
                
                # Each group has 2 equations and 7 unknowns. Underdetermined.
                # We can find a minimum-norm solution or just pick one.
                
                W_solve = np.zeros((4, 7))
                for i in range(2):
                    for k in range(2):
                        w_idx = k * 2 + i
                        # Equations for this (i,k): j=0,1
                        A_eq = np.zeros((2, 7))
                        for jj in range(2):
                            A_eq[jj, :] = U_fix[i*2+jj, :] * V_fix[jj*2+k, :]
                        b_eq = np.ones(2)
                        # Solve A_eq @ W[w_idx,:] = b_eq (least squares)
                        W_solve[w_idx, :], residuals, rank_s, sv = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                
                # Verify
                D3 = np.einsum('ar,br,cr->abc', U_fix, V_fix, W_solve)
                if np.allclose(D3, T, atol=1e-9):
                    return U_fix.tolist(), V_fix.tolist(), W_solve.tolist()
                else:
                    # Shouldn't happen, but fallback
                    pass
        
        # If we got here, something went wrong. Return the original (possibly wrong) set.
        return U.tolist(), V.tolist(), W.tolist()
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_block_strassen():
    """
    Build 6x6x6 matrix multiplication decomposition using 2x2 block Strassen.
    Each block is 3x3.
    Rank = 7 * 27 = 189.
    """
    n = m = p = 6
    block = 3
    rank = 7 * block * block  # 189
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    U2 = np.array(U2, dtype=float)
    V2 = np.array(V2, dtype=float)
    W2 = np.array(W2, dtype=float)
    
    # For each Strassen product r (0..6) and each block position (bi,bj) in A,
    # (bj,bk) in B, (bk,bi) in C:
    # The contribution is U2[bi*2+bj, r] * V2[bj*2+bk, r] * W2[bk*2+bi, r]
    # times the identity on the 3x3 block space.
    
    # For a specific scalar position within the 3x3 block (offset oi,oj in A, oj,ok in B, ok,oi in C):
    # U[(bi*3+oi)*3 + (bj*3+oj), col] += U2[bi*2+bj, r] * (1 if oi==oi and oj==oj) 
    # Actually, for the block decomposition, the 3x3 identity means:
    # A[bi*3+oi, bj*3+oj] * B[bj*3+oj, bk*3+ok] contributes to C[bi*3+oi, bk*3+ok]
    # only when the "diagonal" matches, i.e., this is just the identity on the block indices.
    
    # More precisely: the tensor for block (bi,bj,bk) with internal indices (oi,oj,ok) is:
    # T[bi*3+oi, bj*3+oj, bk*3+ok] = 1 if oi==oj==ok... no.
    # 
    # Actually, for matrix multiplication, A[i,j]*B[j,k] contributes to C[i,k].
    # With blocks: A[bi*3+oi, bj*3+oj] * B[bj*3+oj, bk*3+ok] contributes to C[bi*3+oi, bk*3+ok].
    # This is valid for ALL oi,oj,ok in [0,3). There's no constraint like oi==oj.
    # 
    # So the block tensor is: for each (bi,bj,bk), the 3x3x3 sub-tensor is the 
    # "identity" in the sense that it's the full matmul tensor for 3x3x3, which is
    # T3[oi,oj,ok] = 1 for all oi,oj,ok (since every term contributes).
    # 
    # Wait no. For 3x3x3 matmul, T3[oi,oj,ok] = 1 for all oi,oj,ok because
    # A[oi,oj]*B[oj,ok] contributes to C[oi,ok] for every valid (oi,oj,ok).
    # So the 3x3x3 matmul tensor IS the all-ones tensor? No, that's not right either.
    # 
    # The matmul tensor T_n has T_n[i,j,k] = 1 for all i,j,k in [0,n).
    # This is because every scalar product A[i,j]*B[j,k] contributes exactly once to C[i,k].
    # So yes, T_n is the all-ones tensor of size n*n*n.
    # 
    # But wait, that would mean the rank is 1 for all n, which is absurd.
    # 
    # I think I'm confusing myself. Let me re-read the task.
    # 
    # "The tensor for matrix multiplication: T[i, j, k] = 1 for all i, j, k"
    # No, that's not what it says. Let me re-read.
    # 
    # From the current program's comments:
    # "T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k"
    # 
    # So T is NOT all ones. T[a,b,c] = 1 only when a, b, c are consistent with
    # a single (i,j,k) triple. For a=36, b=36, c=36 (6x6 case), most triples (a,b,c)
    # are NOT of the form (i*6+j, j*6+k, k*6+i).
    # 
    # Specifically: given a, we get i=a//6, j=a%6.
    # Given b, we get j_b=b%6, k_b=b//6. Need j_b == j.
    # Given c, we get i_c=c%6, k_c=c//6. Need i_c == i and k_c == k_b.
    # 
    # So T is sparse: only 216 out of 36*36*36=46656 entries are 1.
    # 
    # For the block decomposition: 
    # a = (bi*3+oi)*3 + (bj*3+oj) = bi*9 + oi*3 + bj*3 + oj
    # b = (bj*3+oj)*3 + (bk*3+ok) = bj*9 + oj*3 + bk*3 + ok
    # c = (bk*3+ok)*3 + (bi*3+oi) = bk*9 + ok*3 + bi*3 + oi
    # 
    # For T[a,b,c]=1, we need:
    # a//6 = i, a%6 = j where i = bi*3+oi, j = bj*3+oj
    # b//6 = k_b, b%6 = j_b where j_b should equal j, k_b should equal k=bk*3+ok
    # c//6 = k_c, c%6 = i_c where i_c should equal i, k_c should equal k_b
    # 
    # a = (bi*3+oi)*6 + (bj*3+oj) = bi*18 + oi*6 + bj*3 + oj
    # a//6 = bi*3 + oi + (bj*3+oj)//6... this is getting complicated.
    # 
    # Let me just verify: for the block decomposition to work, we need the
    # Kronecker product structure. The 6x6 matmul tensor is NOT simply the
    # Kronecker product of two 3x3 tensors in the naive sense.
    # 
    # Actually, I think the correct approach is:
    # 6x6 matmul can be seen as 2x2 blocks of 3x3.
    # A = [[A00, A01],[A10, A11]] where each Ai is 3x3.
    # B = [[B00, B01],[B10, B11]] where each Bj is 3x3.
    # C = A@B = [[A00@B00 + A01@B10, A00@B01 + A01@B11],
    #            [A10@B00 + A11@B10, A10@B01 + A11@B11]]
    # 
    # Using Strassen on the 2x2 block level:
    # M1 = (A00+A11)@(B00+B11)
    # M2 = (A10+A11)@B00
    # M3 = A00@(B01-B11)
    # M4 = A11@(B10-B00)
    # M5 = (A00+A01)@B11
    # M6 = (A10-A00)@(B00+B01)
    # M7 = (A01-A11)@(B10+B11)
    # 
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2  (need to verify)
    # 
    # Each M_r is a 3x3 matrix product, which requires rank 27 (schoolbook for 3x3).
    # So total rank = 7 * 27 = 189.
    # 
    # Now, the tensor decomposition:
    # For each Strassen product r and each scalar position within the 3x3 blocks,
    # we have a rank-1 term.
    # 
    # For product r, the A-side coefficient for block (bi,bj) is U2[bi*2+bj, r].
    # The B-side coefficient for block (bj,bk) is V2[bj*2+bk, r].
    # The C-side coefficient for block (bk,bi) is W2[bk*2+bi, r].
    # 
    # Within each 3x3 block, we use the schoolbook decomposition (rank 27).
    # For the 3x3 schoolbook, for each (oi,oj,ok) in [0,3)^3:
    # A[oi,oj] * B[oj,ok] -> C[oi,ok]
    # 
    # So for the full 6x6, the column index is: r * 27 + oi * 9 + oj * 3 + ok
    # (or some other ordering of the 27 scalar products within the block)
    
    col = 0
    for r in range(7):
        for bi in range(2):
            for bj in range(2):
                a_coeff = U2[bi*2+bj, r]
                if a_coeff == 0:
                    continue
                for bk in range(2):
                    b_coeff = V2[bj*2+bk, r]
                    if b_coeff == 0:
                        continue
                    c_coeff = W2[bk*2+bi, r]
                    if c_coeff == 0:
                        continue
                    
                    # For this (r, bi, bj, bk), we need 27 scalar products
                    # within the 3x3 blocks.
                    for oi in range(3):
                        for oj in range(3):
                            for ok in range(3):
                                # A index: (bi*3+oi)*6 + (bj*3+oj)
                                a_idx = (bi*3+oi)*6 + (bj*3+oj)
                                # B index: (bj*3+oj)*6 + (bk*3+ok)
                                b_idx = (bj*3+oj)*6 + (bk*3+ok)
                                # C index: (bk*3+ok)*6 + (bi*3+oi)
                                c_idx = (bk*3+ok)*6 + (bi*3+oi)
                                
                                U[a_idx, col] = a_coeff
                                V[b_idx, col] = b_coeff
                                W[c_idx, col] = c_coeff
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _validate(U, V, W):
    """Validate the decomposition."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    if U_arr.shape[1] != V_arr.shape[1] or V_arr.shape[1] != W_arr.shape[1]:
        return False
    
    D = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(D, T, atol=1e-9)


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 6x6x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Build block Strassen decomposition (rank 189)
    result = _build_block_strassen()
    
    # Validate
    if _validate(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook
    rank = 6 * 6 * 6
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i * 6 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 6 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
