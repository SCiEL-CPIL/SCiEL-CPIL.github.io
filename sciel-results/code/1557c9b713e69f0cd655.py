import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0
    
    # We will construct a set of terms and then reduce the rank.
    # Start with the naive decomposition of 3x3 * 3x8.
    # This is 3*3*8 = 72 terms.
    # We can potentially reduce this.
    
    U_t = []
    V_t = []
    W_t = []
    
    # Direct terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t.append([((i, j), 1)])
                V_t.append([((j, k), 1)])
                W_t.append([((i, k), 1)])
                
    rank = len(W_t)
    
    # Build the tensor matrix to find rank
    # The tensor T is of shape (n*m, m*p, p*n)
    # We have 'rank' terms. Each term is a vector of size n*m * m*p * p*n? No.
    # Each term is a rank-1 tensor.
    # We can form a matrix where each column is a flattened rank-1 tensor.
    # Then the rank of this matrix is the rank of the tensor decomposition.
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    M = np.zeros((size_U * size_V, rank))
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            M[i * m + j, idx] = coeff
        # Note: The tensor is sum U_i V_j W_k.
        # The matrix M should represent the coefficients of the basis tensors.
        # A rank-1 tensor is outer product of u, v, w.
        # We can flatten u, v, w into a single vector? No, that's not how rank works.
        # The rank of a tensor is the minimum number of rank-1 tensors that sum to it.
        # If we have a decomposition, the terms are linearly independent if the rank is minimal.
        # We can check if the terms are linearly dependent by forming a matrix of the terms.
        # But the terms are in a high-dimensional space.
        # We can map each term to a vector in R^(n*m * m*p * p*n)? No.
        # The space of tensors is R^(n*m * m*p * p*n).
        # Each term is a vector in this space.
        # We form a matrix with these vectors as columns.
        # The rank of this matrix is the rank of the set of terms.
        
    # This is too large: 9 * 24 * 24 = 5184 dimensions.
    # 72 columns.
    # We can compute the SVD of this matrix.
    
    # Let's build the matrix properly.
    # Term idx: u, v, w.
    # The tensor is T = sum u_i v_j w_k.
    # We can reshape T to a matrix of size (n*m) x (m*p * p*n)?
    # Or (n*m * m*p) x (p*n)?
    # Let's use the matricization: (n*m) x (m*p * p*n) is not standard.
    # Standard: (n*m) x (m*p) for each k? No.
    # The tensor is T_{i,j,k}.
    # We can view it as a matrix of size (n*m) x (m*p * p*n) by fixing i,j? No.
    # Let's flatten the last two indices: (n*m) x (m*p * p*n).
    # Or flatten the first two: (n*m * m*p) x (p*n).
    
    # Let's use (n*m * m*p) x (p*n).
    # Row index: (i, j, k) -> i*m*p + j*p + k? No.
    # The term is A_{i,j} B_{j,k} C_{i,k}.
    # The tensor is T_{i,j,k}.
    # We can flatten i,j into one index (size n*m) and k into another?
    # No, k is in B and C.
    # B is m*p. C is p*n.
    # The term is A_{i,j} * B_{j,k} * C_{i,k}.
    # This is a 3-index tensor T_{i, j, k}.
    # We can flatten (i, j) to size n*m.
    # Then for each (i, j), we have a vector of size p (for k).
    # So T is a matrix of size (n*m) x (p)? No, B has p columns.
    # T_{i,j,k} = A_{i,j} B_{j,k} C_{i,k}.
    # For fixed i, j, T_{i,j,k} is a vector in k.
    # So T is a tensor of size n x m x p.
    # We can flatten n, m to get a matrix of size (n*m) x p.
    # But C depends on i and k.
    # So T_{i,j,k} is not separable in that way.
    
    # Let's just use the full vector space.
    # Dimension: n*m*p = 3*3*8 = 72.
    # Each term is a vector of size 72.
    # We have 72 terms.
    # We form a 72x72 matrix.
    
    M = np.zeros((n * m * p, rank))
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            for (j2, k), coeff2 in V_t[idx]:
                if j == j2:
                    for (i2, k2), coeff3 in W_t[idx]:
                        if i == i2 and k == k2:
                            M[i * m * p + j * p + k, idx] = coeff * coeff2 * coeff3
                            
    # Now M is 72x72.
    # We want to find the rank of M.
    # If the rank is r, we can reduce the number of terms to r.
    # We need to find a basis of the column space of M.
    # Then we can express the original terms as linear combinations of the basis.
    # But we need the coefficients for U, V, W.
    # This is tricky because the linear combination is in the tensor space,
    # not necessarily preserving the rank-1 structure of U, V, W.
    
    # However, if we just want to reduce the rank, we can find the minimal rank decomposition.
    # This is a hard problem.
    # But for this specific structure, the terms might be linearly independent.
    # Let's check the rank of M.
    
    # If rank is 72, we are stuck with 72.
    # If rank is less, we can try to find a better decomposition.
    
    # For now, let's just return the 72-term decomposition.
    # We can try to improve it by using Strassen.
    
    # Let's try the Strassen-based 68-term decomposition from the seed.
    # It is better than 72.
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
        
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
        
    U_t = []
    V_t = []
    W_t = []
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, c:c+2] for c in 0, 2, 4, 6
    for c in range(0, 8, 2):
        strassen_decomp(0, 0, 0, c, U_t, V_t, W_t)
        
    # Direct terms for remaining A indices
    # A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
            
    # A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
            
    # A[2, 2] * B[2, 0:8]
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
        
    rank = len(W_t)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
            
    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res
        
    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
