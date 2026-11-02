import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # For 2x4x7 matrix multiplication:
    # A is 2x4, B is 4x7, C is 2x7
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (28, rank)  
    # W has shape (p*n, rank) = (14, rank)
    # 
    # The decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal sum_j A[i,j]*B[j,k] for all A, B.
    
    # Strategy: Use Strassen (rank 7) for 2x2x2 blocks and schoolbook (rank 4) for 2x2x1 blocks.
    # Split j dimension into {0,1} and {2,3}
    # Split k dimension into {0,1}, {2,3}, {4,5}, {6}
    # 
    # For j in {0,1} (2x2 A block):
    #   k in {0,1}: 2x2x2 Strassen (7 ops)
    #   k in {2,3}: 2x2x2 Strassen (7 ops)
    #   k in {4,5}: 2x2x2 Strassen (7 ops)
    #   k in {6}:   2x2x1 schoolbook (4 ops)
    # For j in {2,3} (2x2 A block):
    #   k in {0,1}: 2x2x2 Strassen (7 ops)
    #   k in {2,3}: 2x2x2 Strassen (7 ops)
    #   k in {4,5}: 2x2x2 Strassen (7 ops)
    #   k in {6}:   2x2x1 schoolbook (4 ops)
    # Total: 6*7 + 2*4 = 42 + 8 = 50
    
    # Try to find a better decomposition using a different approach.
    # The theoretical lower bound is max(2*4, 4*7, 2*7) = 28.
    # But achieving rank 28 is unlikely for this specific case.
    
    # Let's try a randomized search for a lower rank decomposition.
    # We'll start with rank 48 and try to verify if it works, then try lower.
    
    # Actually, let's first try a more clever block decomposition.
    # What if we use a 2x3x3 block? The rank of 2x3x3 matrix multiplication
    # is known to be 9 (schoolbook) or possibly lower.
    
    # Let me try: split j into {0,1,2} and {3}
    # A[:, 0:3] is 2x3, B[0:3, :] is 3x7
    # A[:, 3] is 2x1, B[3, :] is 1x7
    
    # For 2x3 * 3x7: split k into {0,1,2,3,4,5,6}
    # We can use 2x3x2 blocks (rank 6 schoolbook or maybe 5 with Strassen-like)
    # and 2x3x1 blocks (rank 3 schoolbook)
    
    # Actually, the rank of 2x3x2 matrix multiplication is 6 (schoolbook) 
    # and I believe it's tight.
    # 2x3x1 is rank 3.
    
    # So: 2x3 * 3x7 can be done as:
    # 3 blocks of 2x3x2 (rank 6 each) + 1 block of 2x3x1 (rank 3) = 21
    # Plus 2x1 * 1x7: rank 2*7 = 14... that's too much.
    
    # Wait, 2x1 * 1x7 is just element-wise: C[i,k] = A[i,3]*B[3,k]
    # This requires 2*7 = 14 rank-1 terms. That's bad.
    
    # Better to stick with the 2x2 splitting of j.
    
    # Let me try yet another approach: use the fact that we can share
    # some of the Strassen operations between adjacent blocks.
    
    # For now, let's implement the rank-50 solution and try to reduce it
    # by finding linear dependencies.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        V[h, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1
        W[c11, col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a, col] += 1
        V[f, col] += 1
        V[h, col] -= 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d, col] += 1
        V[g, col] += 1
        V[e, col] -= 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a, col] += 1
        U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += 1
        U[c, col] += 1
        U[a, col] -= 1
        V[e, col] += 1
        V[f, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += 1
        U[b, col] += 1
        U[d, col] -= 1
        V[g, col] += 1
        V[h, col] += 1
        W[c00, col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a, col] += 1
        V[e, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P2 = b*g
        U[b, col] += 1
        V[g, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P3 = c*e
        U[c, col] += 1
        V[e, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P4 = d*g
        U[d, col] += 1
        V[g, col] += 1
        W[c10, col] += 1
        col += 1
        
        return True

    # Process blocks
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                break
            if not add_strassen_222(0, j_start, k_start):
                break
        if time.time() > deadline:
            break
        if not add_schoolbook_221(0, j_start, 6):
            break

    # Now try to reduce the rank by finding linear dependencies.
    # The tensor T is defined by:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta(i,i') * delta(j,j') * delta(k,k')
    # Wait, that's not quite right. Let me define it properly.
    
    # The multiplication tensor for C[i,k] = sum_j A[i,j]B[j,k] is:
    # T_{i,j,k} (where i in [0,n), j in [0,m), k in [0,p))
    # such that C[i,k] = sum_{i',j',k'} T_{i,j,k} A[i',j'] B[j',k']
    # This requires T_{i,j,k} = delta(i,i') * delta(j,j') * delta(k,k')... 
    # No, that's not right either.
    
    # Actually, the tensor is:
    # T_{i,j,k} where the indices correspond to (row of A, col of A = row of B, col of B)
    # And the output is C[i,k] = sum_j A[i,j] B[j,k]
    # So T_{i,j,k} = 1 if we're computing C[i,k] using A[i,j] and B[j,k]
    # But we need to be careful about the factorization.
    
    # The factorization is:
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. Let me re-derive.
    
    # We want: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # to equal sum_j A[i,j] B[j,k]
    
    # This means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,j} (i.e., 1)
    # when the indices match appropriately.
    
    # Actually, I think the correct condition is:
    # For all i, k: sum_j [sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]] = 1
    # No, that's not right either.
    
    # Let me think about this differently. The tensor T has entries:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the factorization is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, that can't be right because T[i,j,k] = 1 for all i,j,k,
    # but the left side is a scalar and the right side is a sum of products.
    
    # I think I'm confusing myself. Let me look at this from the perspective
    # of the bilinear complexity.
    
    # The matrix multiplication C = A*B where A is n x m and B is m x p
    # can be written as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The rank of this bilinear map is the minimum number of scalar multiplications
    # needed. Each scalar multiplication is of the form:
    # (linear combo of A entries) * (linear combo of B entries)
    # and the result contributes to C entries via linear combinations.
    
    # So we have:
    # C[i,k] = sum_r (sum_{i',j'} U[i*m+j', r] * A[i',j']) * (sum_{j'',k'} V[j''*p+k', r] * B[j'',k']) * W[k*n+i, r]
    
    # Wait, that's not quite right. Let me be more careful.
    
    # The standard form is:
    # C[i,k] = sum_r alpha_r(i,k) * (sum_{i',j'} U[i',j',r] * A[i',j']) * (sum_{j'',k'} V[j'',k'',r] * B[j'',k''])
    
    # But in our case, the factors are:
    # U has shape (n*m, rank), so U[i*m+j, r] is the coefficient for A[i,j] in the r-th term.
    # V has shape (m*p, rank), so V[j*p+k, r] is the coefficient for B[j,k] in the r-th term.
    # W has shape (p*n, rank), so W[k*n+i, r] is the coefficient for C[i,k] in the r-th term.
    
    # So the decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    
    # Wait, that doesn't look right either because j is being summed in both U and V.
    
    # Let me try a different approach. I'll just verify the current decomposition
    # by checking if it produces the correct result for random matrices.
    
    def verify(U, V, W, rank):
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # The r-th term contributes:
            # (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            # But this is getting complicated. Let me just use the tensor form.
            
            # Actually, I think the correct interpretation is:
            # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # where j is the "middle" index that gets summed over.
            
            # So: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            for j in range(m):
                C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Let me verify the current decomposition
    # Actually, I realize my verification function is wrong. Let me fix it.
    
    def verify_decomposition(U, V, W, rank):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
        # If verification fails, return a simple schoolbook decomposition
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce the rank by finding linear dependencies.
    # We can do this by checking if any of the rank-1 terms can be expressed
    # as a linear combination of the others.
    
    # Actually, a simpler approach: try to find a decomposition with rank < 50
    # by using a different tiling strategy.
    
    # Let me try splitting the 7 columns into 3+4 instead of 2+2+2+1.
    # For j in {0,1}: 2x2 A block
    #   k in {0,1,2}: 2x2x3 block
    #   k in {3,4,5,6}: 2x2x4 block
    # For j in {2,3}: 2x2 A block
    #   k in {0,1,2}: 2x2x3 block
    #   k in {3,4,5,6}: 2x2x4 block
    
    # The rank of 2x2x3 is 6 (schoolbook) or possibly 5.
    # The rank of 2x2x4 is 8 (schoolbook) or possibly 7.
    
    # If we can get 2x2x3 to rank 5 and 2x2x4 to rank 7:
    # Total: 2*5 + 2*7 = 10 + 14 = 24. That would be great!
    
    # But I'm not sure if those ranks are achievable. Let me just return
    # the current rank-50 decomposition for now.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
