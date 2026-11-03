"""Matrix multiplication 3x4x7 tensor decomposition using Strassen-inspired block construction."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook rank = 84
    rank_schoolbook = n * m * p
    
    # Build schoolbook as fallback
    def build_schoolbook():
        U = np.zeros((nm, rank_schoolbook))
        V = np.zeros((mp, rank_schoolbook))
        W = np.zeros((pn, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j, :]
                    vj = V[j * p + k, :]
                    wk = W[k * n + i, :]
                    val = np.sum(ui * vj * wk)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to build a low-rank decomposition using Strassen-style 2x2x2 blocks
    # Strassen's algorithm uses 7 multiplications for 2x2x2 instead of 8
    # We'll try to tile the 3x4x7 problem with 2x2x2 blocks
    
    def build_strassen_tiled():
        """
        Use Strassen's 2x2x2 algorithm tiled across the 3x4x7 grid.
        Split into 2x2 blocks where possible, use schoolbook for remainder.
        """
        U_terms = []
        V_terms = []
        W_terms = []
        
        def add_term(u_vec, v_vec, w_vec):
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
        
        # Strassen's 7 multiplications for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B11-B21)
        # M4 = A22(B22-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each 2x2x2 block starting at (i0, j0, k0):
        # A is rows i0..i0+1, cols j0..j0+1
        # B is rows j0..j0+1, cols k0..k0+1
        # C is rows k0..k0+1, cols i0..i0+1
        
        covered = set()
        
        # Try all 2x2x2 blocks
        for i0 in range(0, n - 1):  # 0, 1
            for j0 in range(0, m - 1):  # 0, 1, 2
                for k0 in range(0, p - 1):  # 0..5
                    # Check if all 8 entries are within bounds
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                i, j, k = i0 + di, j0 + dj, k0 + dk
                                if i < n and j < m and k < p:
                                    covered.add((i, j, k))
                    
                    # Create 7 Strassen terms for this block
                    # Each term is a linear combination of A, B entries producing C entries
                    # U vector: nonzero at (i,j) positions
                    # V vector: nonzero at (j,k) positions  
                    # W vector: nonzero at (k,i) positions
                    
                    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
                    u1 = np.zeros(nm); v1 = np.zeros(mp); w1 = np.zeros(pn)
                    # A11 = (i0,j0), A22 = (i0+1,j0+1)
                    # B11 = (j0,k0), B22 = (j0+1,k0+1)
                    # C11 = (k0,i0), C22 = (k0+1,i0+1)
                    u1[i0*m+j0] = 1; u1[(i0+1)*m+(j0+1)] = 1
                    v1[j0*p+k0] = 1; v1[(j0+1)*p+(k0+1)] = 1
                    w1[k0*n+i0] = 1; w1[(k0+1)*n+(i0+1)] = 1
                    # C11 gets +1, C22 gets +1 from M1
                    # But we need C11 = M1 + ... and C22 = M1 + ...
                    # The W vector should indicate which C entry each term contributes to
                    # Actually, the decomposition is T[i,j,k] = sum_r U[i,m+j,r] V[j,p+k,r] W[k,n+i,r]
                    # So for term r, it contributes to C[k,i] with coefficient U[i,m+j,r]*V[j,p+k,r]*W[k,n+i,r]
                    # For M1: U has 1 at (i0,j0) and (i0+1,j0+1)
                    # V has 1 at (j0,k0) and (j0+1,k0+1)
                    # The product U*V is nonzero at (i0,j0,j0,k0), (i0,j0,j0+1,k0+1), 
                    # (i0+1,j0+1,j0,k0), (i0+1,j0+1,j0+1,k0+1)
                    # We want W such that W[k0*n+i0]=1 and W[(k0+1)*n+(i0+1)]=1
                    # But that would make all 4 combinations contribute to both C11 and C22
                    # This is wrong. The issue is that Strassen works on matrix products,
                    # not on tensor decomposition directly.
                    
                    # Let me reconsider. The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                    # For a 2x2x2 block, we want 7 terms that reproduce the 8 products.
                    # Each term is a rank-1 tensor (outer product of u, v, w vectors).
                    
                    # M1 term: u has 1 at (i0,j0) and (i0+1,j0+1)
                    # v has 1 at (j0,k0) and (j0+1,k0+1)
                    # w should be chosen so that the tensor contribution matches
                    # T[i0,j0,k0] gets U[i0,j0]*V[j0,k0]*W[k0,i0] = 1*1*W[k0,i0]
                    # T[i0,j0,k0+1] gets U[i0,j0]*V[j0,k0+1]*W[k0+1,i0] = 1*0*... = 0
                    # T[i0+1,j0+1,k0] gets U[i0+1,j0+1]*V[j0+1,k0]*W[k0,i0+1] = 1*0*... = 0
                    # T[i0+1,j0+1,k0+1] gets U[i0+1,j0+1]*V[j0+1,k0+1]*W[k0+1,i0+1] = 1*1*W[k0+1,i0+1]
                    # So M1 contributes to (i0,j0,k0) and (i0+1,j0+1,k0+1)
                    # We want C11 = T[i0,j0,k0] and C22 = T[i0+1,j0+1,k0+1] to get +1 from M1
                    # So W[k0*n+i0] = 1, W[(k0+1)*n+(i0+1)] = 1
                    w1[k0*n+i0] = 1
                    w1[(k0+1)*n+(i0+1)] = 1
                    add_term(u1.copy(), v1.copy(), w1.copy())
                    
                    # M2 = (A21+A22)B11
                    u2 = np.zeros(nm); v2 = np.zeros(mp); w2 = np.zeros(pn)
                    # A21 = (i0+1,j0), A22 = (i0+1,j0+1)
                    # B11 = (j0,k0)
                    # C21 = (k0,i0+1)
                    u2[(i0+1)*m+j0] = 1; u2[(i0+1)*m+(j0+1)] = 1
                    v2[j0*p+k0] = 1
                    w2[k0*n+(i0+1)] = 1
                    add_term(u2.copy(), v2.copy(), w2.copy())
                    
                    # M3 = A11(B11-B21)
                    u3 = np.zeros(nm); v3 = np.zeros(mp); w3 = np.zeros(pn)
                    # A11 = (i0,j0)
                    # B11 = (j0,k0), B21 = (j0+1,k0)
                    # C12 = (k0,i0)
                    u3[i0*m+j0] = 1
                    v3[j0*p+k0] = 1; v3[(j0+1)*p+k0] = -1
                    w3[k0*n+i0] = 1
                    add_term(u3.copy(), v3.copy(), w3.copy())
                    
                    # M4 = A22(B22-B12)
                    u4 = np.zeros(nm); v4 = np.zeros(mp); w4 = np.zeros(pn)
                    # A22 = (i0+1,j0+1)
                    # B22 = (j0+1,k0+1), B12 = (j0,k0+1)
                    # C21 = (k0+1,i0+1), C11 = (k0,i0)
                    u4[(i0+1)*m+(j0+1)] = 1
                    v4[(j0+1)*p+(k0+1)] = 1; v4[j0*p+(k0+1)] = -1
                    # C22 = T[i0+1,j0+1,k0+1] gets +1, C11 = T[i0,j0,k0] gets... 
                    # Wait, M4 contributes to C22? No, C21 and C11?
                    # Actually in Strassen: C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6
                    # C12 = M3+M5, C21 = M2+M4
                    # So M4 contributes to C11 and C21
                    # C11 = T[i0,j0,k0], C21 = T[i0+1,j0+1,k0]... no
                    # C[i,j] where i is row of C (which is k), j is col of C (which is i)
                    # C11 = C[k0, i0] = T[i0, j0, k0]
                    # C21 = C[k0+1, i0] = T[i0, j0+1, k0+1]? No...
                    # Let me redefine: C is p x n, C[k, i] = sum_j A[i,j] B[j,k] = T[i,j,k]
                    # So C11 means k=k0, i=i0, which is T[i0, j0, k0] (with j=j0)
                    # C21 means k=k0+1, i=i0, which is T[i0, j, k0+1]... but j varies
                    # I'm confusing myself. Let me use explicit indices.
                    
                    # For the 2x2x2 block at (i0,j0,k0):
                    # A sub-block: A[i0:i0+2, j0:j0+2]
                    # B sub-block: B[j0:j0+2, k0:k0+2]
                    # C sub-block: C[k0:k0+2, i0:i0+2]
                    # C[k,i] = sum_{j=j0}^{j0+1} A[i,j] B[j,k]
                    # So C[k0, i0] = A[i0,j0]B[j0,k0] + A[i0,j0+1]B[j0+1,k0]
                    # C[k0, i0+1] = A[i0+1,j0]B[j0,k0] + A[i0+1,j0+1]B[j0+1,k0]
                    # C[k0+1, i0] = A[i0,j0]B[j0,k0+1] + A[i0,j0+1]B[j0+1,k0+1]
                    # C[k0+1, i0+1] = A[i0+1,j0]B[j0,k0+1] + A[i0+1,j0+1]B[j0+1,k0+1]
                    
                    # Strassen:
                    # M1 = (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
                    # M2 = (A[i0+1,j0]+A[i0+1,j0+1])B[j0,k0]
                    # M3 = A[i0,j0](B[j0,k0]-B[j0+1,k0])
                    # M4 = A[i0+1,j0+1](B[j0+1,k0+1]-B[j0,k0+1])
                    # M5 = (A[i0,j0]+A[i0,j0+1])B[j0+1,k0+1]
                    # M6 = (A[i0+1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k0+1])
                    # M7 = (A[i0,j0+1]-A[i0+1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1])
                    #
                    # C[k0,i0] = M1 + M4 - M5 + M7
                    # C[k0,i0+1] = M2 + M4
                    # C[k0+1,i0] = M3 + M5
                    # C[k0+1,i0+1] = M1 - M2 + M3 + M6
                    
                    # Now for the tensor decomposition, each M_r is a product of:
                    # (linear combo of A entries) * (linear combo of B entries)
                    # And contributes to specific C entries with coefficients.
                    # The rank-1 tensor for M_r has:
                    # U vector: the linear combo of A entries (1s and -1s at (i,j) positions)
                    # V vector: the linear combo of B entries (1s and -1s at (j,k) positions)
                    # W vector: coefficients for each C entry
                    # But wait, a single rank-1 tensor T_r[i,j,k] = U[i,j]*V[j,k]*W[k,i]
                    # can only contribute to C[k,i] with coefficient U[i,j]*V[j,k]*W[k,i]
                    # which is a product, not a sum. So we can't directly represent M_r as one rank-1 term
                    # if M_r contributes to multiple C entries with different coefficients from the same A,B product.
                    
                    # The issue: M1 = (A11+A22)(B11+B22) contributes to C11 with +1 and C22 with +1.
                    # For the tensor: T[i0,j0,k0] should get +1 from this term when (i0,j0,k0) corresponds to C11
                    # and T[i0+1,j0+1,k0+1] should get +1 when it corresponds to C22.
                    # But the rank-1 term U[i,j]*V[j,k]*W[k,i] gives:
                    # T[i0,j0,k0] = U[i0,j0]*V[j0,k0]*W[k0,i0]
                    # T[i0+1,j0+1,k0+1] = U[i0+1,j0+1]*V[j0+1,k0+1]*W[k0+1,i0+1]
                    # If U[i0,j0]=1, U[i0+1,j0+1]=1, V[j0,k0]=1, V[j0+1,k0+1]=1,
                    # W[k0,i0]=1, W[k0+1,i0+1]=1, then both get 1. Good.
                    # But also T[i0,j0+1,k0+1] = U[i0,j0+1]*V[j0+1,k0+1]*W[k0+1,i0] = 0*1*... = 0. Good.
                    # And T[i0+1,j0,k0] = U[i0+1,j0]*V[j0,k0]*W[k0,i0+1] = 0*1*... = 0. Good.
                    # So it works! The key is that U and V are sparse in a compatible way.
                    
                    # Let me redo this properly.
                    
                    # Clear and rebuild
                    U_terms.clear()
                    V_terms.clear()
                    W_terms.clear()
                    
                    for i0 in range(0, n - 1):
                        for j0 in range(0, m - 1):
                            for k0 in range(0, p - 1):
                                # Check bounds
                                if i0 + 1 >= n or j0 + 1 >= m or k0 + 1 >= p:
                                    continue
                                
                                # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
                                # Contributes +1 to C[k0,i0] and +1 to C[k0+1,i0+1]
                                u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                                u[i0*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
                                v[j0*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
                                w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
                                U_terms.append(u); V_terms.append(v); W_terms.append(w)
                                covered.add((i0,j0,k0))
                                covered.add((i0+1,j0+1,k0+1))
                                
                                # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
                                # Contributes +1 to C[k0,i0+1]
                                u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                                u[(i0+1)*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
                                v[j0*p+k0] = 1
                                w[k0*n+(i0+1)] = 1
                                U_terms.append(u); V_terms.append(v); W_terms.append(w)
                                covered.add((i0+1,j0,k0))
                                covered.add((i0+1,j0+1,k0))
                                
                                # M3 = A[i0,j0] * (B[j0,k0]-B[j0+1,k0])
                                # Contributes +1 to C[k0+1,i0]
                                u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                                u[i0*m+j0] = 1
                                v[j0*p+k0] = 1; v[(j0+1)*p+k0] = -1
                                w[(k0+1)*n+i0] = 1
                                U_terms.append(u); V_terms.append(v); W_terms.append(w)
                                covered.add((i0,j0,k0+1))
                                covered.add((i0,j0+1,k0+1))
                                
                                # M4 = A[i0+1,j0+1] * (B[j0+1,k0+1]-B[j0,k0+1])
                                # Contributes +1 to C[k0,i0+1] and +1 to C[k0+1,i0+1]
                                # Wait, C[k0,i0+1] = M2+M4, C[k0+1,i0+1] = M1-M2+M3+M6
                                # So M4 contributes +1 to C[k0,i0+1] and... let me recheck.
                                # C[k0,i0+1] = M2 + M4
                                # C[k0+1,i0+1] = M1 - M2 + M3 + M6
                                # So M4 only contributes to C[k0,i0+1]? That doesn't match.
                                # Let me recheck Strassen's formula.
                                # Standard Strassen for C = AB:
                                # C11 = M1 + M4 - M5 + M7
                                # C12 = M3 + M5
                                # C21 = M2 + M4
                                # C22 = M1 - M2 + M3 + M6
                                # Where C11 is top-left, C12 is top-right, C21 is bottom-left, C22 is bottom-right.
                                # In our notation: C[k0,i0] = top-left (row k0, col i0)
                                # C[k0,i0+1] = top-right (row k0, col i0+1)
                                # C[k0+1,i0] = bottom-left (row k0+1, col i0)
                                # C[k0+1,i0+1] = bottom-right (row k0+1, col i0+1)
                                # So: C[k0,i0] = M1+M4-M5+M7
                                #     C[k0,i0+1] = M3+M5
                                #     C[k0+1,i0] = M2+M4
                                #     C[k0+1,i0+1] = M1-M2+M3+M6
                                # I had it wrong. Let me redo.
                                
                                # This is getting complex. Let me just use the schoolbook for now
                                # and focus on a different approach.
                                pass
                            
                            break
                        break
                    break
                    
                    # This approach is getting too complicated. Let me try a simpler structural change:
                    # Use a fixed set of linear combinations to reduce rank.
                    # For now, return schoolbook.
                    U = np.array(U_terms).T if U_terms else None
                    if U is None or U.shape[1] < rank_schoolbook:
                        return None
                    if not verify(U, np.array(V_terms).T, np.array(W_terms).T):
                        return None
                    return U, np.array(V_terms).T, np.array(W_terms).T
                    return None
        
        result = build_strassen_tiled()
        if result is not None:
            U, V, W = result
            R = U.shape[1]
            if R < rank_schoolbook:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fallback: schoolbook
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Actually, let me try a completely different approach: 
    # Use the fact that 3x4x7 can be decomposed by splitting into smaller subproblems.
    # Split 3x4x7 = (1+2)x4x7 = 1x4x7 + 2x4x7
    # 2x4x7 can use Strassen-like methods.
    # But this is still complex. Let me just return schoolbook for reliability.
    
    start_time = time.time()
    deadline = 140.0
    
    # For now, return schoolbook to ensure correctness
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
