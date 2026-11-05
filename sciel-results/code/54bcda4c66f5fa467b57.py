import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    
    # Helper to add a Strassen block for 2x2 * 2x2
    def strassen_222(i0, j0, k0):
        nonlocal r
        # U indices: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
        # V indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
        # W indices: k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i0*m+j0, r] += 1
        U[(i0+1)*m+(j0+1), r] += 1
        V[j0*p+k0, r] += 1
        V[(j0+1)*p+(k0+1), r] += 1
        W[k0*n+i0, r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+1)*m+j0, r] += 1
        U[(i0+1)*m+(j0+1), r] += 1
        V[j0*p+k0, r] += 1
        W[k0*n+(i0+1), r] += 1
        W[(k0+1)*n+(i0+1), r] -= 1
        r += 1
        
        # M3 = A00(B01 - B11)
        U[i0*m+j0, r] += 1
        V[j0*p+(k0+1), r] += 1
        V[(j0+1)*p+(k0+1), r] -= 1
        W[(k0+1)*n+i0, r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1
        
        # M4 = A11(B10 - B00)
        U[(i0+1)*m+(j0+1), r] += 1
        V[(j0+1)*p+k0, r] += 1
        V[j0*p+k0, r] -= 1
        W[k0*n+i0, r] += 1
        W[k0*n+(i0+1), r] += 1
        r += 1
        
        # M5 = (A00 + A01)B11
        U[i0*m+j0, r] += 1
        U[i0*m+(j0+1), r] += 1
        V[(j0+1)*p+(k0+1), r] += 1
        W[k0*n+i0, r] -= 1
        W[(k0+1)*n+i0, r] += 1
        r += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i0+1)*m+j0, r] += 1
        U[i0*m+j0, r] -= 1
        V[j0*p+k0, r] += 1
        V[j0*p+(k0+1), r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i0*m+(j0+1), r] += 1
        U[(i0+1)*m+(j0+1), r] -= 1
        V[(j0+1)*p+k0, r] += 1
        V[(j0+1)*p+(k0+1), r] += 1
        W[k0*n+i0, r] += 1
        r += 1

    # Helper to add a naive 2x1 * 1x2 block
    def naive_212(i0, j0, k0):
        nonlocal r
        # A is 2x1 at col j0
        # B is 1x2 at row j0, cols k0, k0+1
        # C is 2x2 at rows i0, i0+1, cols k0, k0+1
        
        # 1. A[i0, j0] * B[j0, k0] -> C[i0, k0]
        U[i0*m+j0, r] += 1
        V[j0*p+k0, r] += 1
        W[k0*n+i0, r] += 1
        r += 1
        
        # 2. A[i0, j0] * B[j0, k0+1] -> C[i0, k0+1]
        U[i0*m+j0, r] += 1
        V[j0*p+(k0+1), r] += 1
        W[(k0+1)*n+i0, r] += 1
        r += 1
        
        # 3. A[i0+1, j0] * B[j0, k0] -> C[i0+1, k0]
        U[(i0+1)*m+j0, r] += 1
        V[j0*p+k0, r] += 1
        W[k0*n+(i0+1), r] += 1
        r += 1
        
        # 4. A[i0+1, j0] * B[j0, k0+1] -> C[i0+1, k0+1]
        U[(i0+1)*m+j0, r] += 1
        V[j0*p+(k0+1), r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1

    # Apply Strassen for j-pairs (0,1) and (2,3) with k-pairs (0,1), (2,3), (4,5)
    strassen_222(0, 0, 0)
    strassen_222(0, 0, 2)
    strassen_222(0, 0, 4)
    
    strassen_222(0, 2, 0)
    strassen_222(0, 2, 2)
    strassen_222(0, 2, 4)
    
    # Apply Naive for j=4 with k-pairs (0,1), (2,3), (4,5)
    naive_212(0, 4, 0)
    naive_212(0, 4, 2)
    naive_212(0, 4, 4)
    
    # Now try to reduce rank by finding linear dependencies among the columns
    # We'll check if any column is a linear combination of others and eliminate it
    # This is a simple greedy approach: for each column, check if it's in the span of previous ones
    
    U_arr = U[:, :r].astype(float)
    V_arr = V[:, :r].astype(float)
    W_arr = W[:, :r].astype(float)
    
    # Build the set of "active" columns
    active = list(range(r))
    
    # For each column, try to see if it can be expressed as a combination of others
    # We'll use a simple approach: for each column i, check if removing it still allows
    # us to represent the tensor. This is expensive, so we'll use a heuristic.
    
    # Actually, let's try a different approach: use the fact that the tensor rank is at most
    # the rank of any mode flattening. Let's compute the rank of the mode-2 flattening.
    
    # Mode-2 flattening: reshape the tensor T[i,j,k] into a matrix of size (m * n) x (p * n)
    # Wait, the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # Mode-2 flattening: fix j, so we get a matrix of size (n * p) x (n * p) for each j?
    # No, mode-2 flattening means we group indices (i,k) and treat j as the "free" index.
    # So we get a matrix of size (n * p) x (m * n * p)? No, that's not right.
    
    # Let me think again. The tensor T has indices i, j, k with sizes n, m, p.
    # Mode-2 flattening: we want a matrix where one dimension is j and the other is (i,k).
    # So the matrix has size m x (n * p). But that's not useful for rank.
    
    # Actually, the tensor rank is the minimum r such that T[i,j,k] = sum_l U[i,j,l] * V[j,k,l] * W[k,i,l]
    # (with appropriate index conventions)
    
    # A better approach: try to find if any of the 54 terms is redundant.
    # We can do this by checking if the tensor can be represented with 53 terms.
    # This is a hard problem, so we'll use a heuristic.
    
    # Heuristic: for each column l, check if it can be removed by solving for the other coefficients.
    # This is expensive, so we'll only try a few.
    
    # Actually, let's just return the rank-54 solution for now, since the problem is to find
    # a rank-minimal decomposition, and we don't have a good way to reduce it quickly.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
