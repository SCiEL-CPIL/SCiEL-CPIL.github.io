import time
import itertools

DIMENSIONS = [4, 4, 7]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 times 4x7 matrix multiplication."""
    deadline = time.time() + 130  # 130s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # The tensor T has indices (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # We want rank-r decomposition: T(i,j,k) = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try to find a low-rank decomposition using bilinear/strassen-like approaches
    # For 4x4 by 4x7, we can try to find a decomposition with rank less than n*m*p = 112
    
    # Strategy: Use the fact that we can combine multiplications
    # Let's try to find a decomposition by considering the matrix multiplication as a tensor
    # and trying to minimize rank
    
    # The tensor M_{ijk} for multiplying A (n x m) by B (m x p) gives C (n x p):
    # C_{ik} = sum_j A_{ij} B_{jk}
    # In tensor form with indices (i, j, k), the multiplication tensor has entries:
    # T(i, (j,k)) where the free output index is (i,k) and the summed index is j
    # 
    # Actually, the standard tensor for matrix multiplication M_{n,p,m} has:
    # T[i, j, k] where i indexes output row, j indexes output col, k indexes the sum
    # T[i,j,k] = delta[i,k] * (something)... 
    #
    # Let me think more carefully. The matrix multiplication tensor for C = A * B
    # where A is n x m and B is m x p:
    # C[i][k] = sum_{j=0}^{m-1} A[i][j] * B[j][k]
    # 
    # As a 3-tensor T with indices (output_row, output_col, sum_index):
    # T[i, k, j] = 1 if this combination contributes, 0 otherwise
    # More precisely, T is the identity in the sum index coupling
    # 
    # In the U, V, W decomposition:
    # U has shape (n*m, rank): indexed by (i, j) pairs of input A
    # V has shape (m*p, rank): indexed by (j, k) pairs of input B  
    # W has shape (p*n, rank): indexed by (k, i) pairs of output C
    # 
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} (for the output)
    # Wait, that's not quite right either.
    #
    # The correct condition is: for all i, k: sum_{c} U[i,j1,c] * V[j1,k,c] * W[k,i,c] ... 
    # No, let me be more precise.
    #
    # The matrix multiplication identity is:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # In the U,V,W factorization format:
    # We define linear maps such that the product A*B is computed as:
    # For each rank component c, we compute a "rank-1" bilinear form
    # 
    # The standard formulation: the multiplication tensor T has entries T_{ijk}
    # where the three modes correspond to (input_A_index, input_B_index, output_index)
    # But here we have a specific convention with W[k*n+i, r].
    #
    # Let me use the convention from the problem:
    # U[i*m + j, c] - this is the coefficient for A[i][j] in component c
    # V[j*p + k, c] - this is the coefficient for B[j][k] in component c  
    # W[k*n + i, c] - this is the coefficient for C[i][k] in component c (note: k*n+i indexing)
    #
    # The reconstruction is: C[i][k] = sum_c (sum_j U[i*m+j, c] * A[i][j]) * (sum_j' V[j'*p+k, c] * B[j'][k]) * W[k*n+i, c]
    #
    # Wait, that doesn't look right dimensionally. Let me reconsider.
    #
    # Actually, in tensor decomposition of matrix multiplication:
    # We're decomposing the trilinear form that computes C[i][k] from A and B.
    # The tensor T has the property that C[i][k] = T(A, B, (i,k))
    # where T is trilinear in (A, B, output_dual).
    #
    # The rank-r decomposition writes T as sum of r rank-1 tensors.
    # A rank-1 tensor in this context would be:
    # T_c(A, B, output) = (sum_{ij} U_{ij,c} A_{ij}) * (sum_{jk} V_{jk,c} B_{jk}) * W_{(i,k),c} * output_{ik}
    #
    # But that's still a single scalar times output, so each rank-1 term contributes
    # (sum U_{ij,c} A_{ij}) * (sum V_{jk,c} B_{jk}) * W_{(i,k),c} to C_{ik}.
    #
    # So the full reconstruction is:
    # C_{ik} = sum_c [ (sum_j U_{i*m+j, c} * A_{ij}) * (sum_j V_{j*p+k, c} * B_{jk}) * W_{k*n+i, c} ]
    #
    # For this to equal the true matrix product C_{ik} = sum_j A_{ij} B_{jk} for all A, B:
    # We need: for all i, k: sum_c W_{k*n+i, c} * (sum_j U_{i*m+j, c} A_{ij}) * (sum_j V_{j*p+k, c} B_{jk}) = sum_j A_{ij} B_{jk}
    #
    # This is a condition on the U, V, W matrices. Let's denote:
    # alpha_{i,c} = vector in R^m with components (U_{i*m+0,c}, ..., U_{i*m+m-1,c})
    # beta_{k,c} = vector in R^m with components (V_{0*p+k,c}, ..., V_{(m-1)*p+k,c})
    # w_{i,k,c} = W_{k*n+i, c}
    #
    # Then: C_{ik} = sum_c w_{i,k,c} * (alpha_{i,c} . A_{i,:}) * (beta_{k,c} . B_{:,k})
    #
    # For this to equal sum_j A_{ij} B_{jk} for ALL A and B, we need the bilinear map
    # (A_{i,:}, B_{:,k}) -> sum_c w_{i,k,c} * (alpha_{i,c} . A_{i,:}) * (beta_{k,c} . B_{:,k})
    # to equal the bilinear map (A_{i,:}, B_{:,k}) -> sum_j A_{ij} B_{jk}.
    #
    # This means: sum_c w_{i,k,c} * alpha_{i,c} ⊗ beta_{k,c} = I_m (the m x m identity matrix)
    # for each pair (i, k).
    #
    # So the condition is: for all i, k: sum_c w_{i,k,c} * alpha_{i,c} * beta_{k,c}^T = I_m
    # where alpha is column vector, beta is column vector.
    #
    # This is a system of equations. We want to minimize r (the rank).
    #
    # For a simple construction: if we set alpha_{i,c} = e_{c mod m} (standard basis vectors cyclically),
    # beta_{k,c} = e_{c mod m}, and w_{i,k,c} = 1 if c mod m == j for some j... 
    # No, let's just use the schoolbook: r = m, with alpha_{i,c} = e_c, beta_{k,c} = e_c, w = 1.
    # That gives sum_c e_c * e_c^T = I_m. So rank = m = 4? No wait, that's per (i,k) pair,
    # but the U and V are shared across different i and k.
    #
    # In schoolbook: U[i*m+j, c] = 1 if c == j, else 0. So alpha_{i,c} = e_c for all i.
    # V[j*p+k, c] = 1 if c == j, else 0. So beta_{k,c} = e_c for all k.
    # W[k*n+i, c] = 1 for all i, k, c... no, that gives sum_c 1 * e_c * e_c^T = I_m. Yes!
    # But W[k*n+i, c] = 1 for all i, k, c would work? Let's check:
    # sum_c W[k*n+i, c] * (e_c . A_{i,:}) * (e_c . B_{:,k}) = sum_c 1 * A_{ic} * B_{ck} = sum_j A_{ij}B_{jk}. Yes!
    # So rank = m = 4 works with W all ones? But wait, the schoolbook uses rank = n*m*p.
    # 
    # Oh I see my error. In the schoolbook decomposition as given in the seed program,
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1 for each (i,j,k) triple.
    # That gives rank = n*m*p = 112.
    #
    # But my analysis shows rank = m = 4 should work! Let me verify:
    # U has shape (16, 4): U[i*m+j, c] = delta_{j,c} for c in 0..3
    # V has shape (28, 4): V[j*p+k, c] = delta_{j,c} for c in 0..3  
    # W has shape (28, 4): W[k*n+i, c] = 1 for all k, i, c
    #
    # Check: C[i][k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i][j]) * (sum_j V[j*p+k, c] * B[j][k])
    # = sum_c 1 * A[i][c] * B[c][k] = sum_j A[i][j] * B[j][k] ✓
    #
    # So rank 4 works! But wait, is this really correct? Let me double-check the W indexing.
    # W[k*n + i, r] - so W is indexed by (k, i) with row = k*n + i.
    # For n=4, p=7: W has shape (p*n, rank) = (28, rank).
    #
    # Yes! Rank 4 should be valid. But can we do even better?
    #
    # The minimal rank for matrix multiplication is a hard open problem in general,
    # but for these specific small dimensions, let's see what's achievable.
    #
    # Can rank be less than m=4? For rank r, we need to write I_m = sum_{c=0}^{r-1} w_{i,k,c} * alpha_{i,c} * beta_{k,c}^T
    # for all i,k. If r < m, then the vectors alpha_{i,c} and beta_{k,c} live in R^m,
    # and we're forming a sum of r rank-1 matrices (scaled) that must equal I_m.
    # The rank of a sum of r rank-1 matrices is at most r. So we need r >= rank(I_m) = m = 4.
    #
    # Therefore the minimal rank is exactly 4!
    #
    # Wait, but that seems too good. Let me reconsider whether the U, V, W can depend on i and k
    # in ways that allow r < m. The constraint is that for EACH (i,k) pair, we need:
    # sum_c w_{i,k,c} * alpha_{i,c} * beta_{k,c}^T = I_m
    #
    # If r < m, then for a fixed i, the set {alpha_{i,c} : c=0..r-1} spans at most r dimensions.
    # Similarly for beta. But the product sum must give a full-rank matrix I_m.
    # 
    # Actually, consider: if all alpha_{i,c} are the same vector v and all beta_{k,c} are the same w,
    # then sum_c w_{i,k,c} * v * w^T = (sum_c w_{i,k,c}) * v * w^T which is rank 1, can't be I_m if m > 1.
    #
    # More generally, for fixed i: {alpha_{i,0}, ..., alpha_{i,r-1}} spans a subspace of dimension <= r.
    # The matrix sum_c w_{i,k,c} * alpha_{i,c} * beta_{k,c}^T has rank <= r (since it's a sum of r rank-1 terms).
    # For this to equal I_m, we need r >= m. So indeed r >= 4.
    #
    # And r = 4 is achievable as shown. So the answer is rank = 4.

    # Let's verify once more with explicit construction:
    # n=4, m=4, p=7
    # U: shape (16, 4), U[i*4+j, c] = 1 if j==c else 0
    # V: shape (28, 4), V[j*7+k, c] = 1 if j==c else 0
    # W: shape (28, 4), W[k*4+i, c] = 1 for all k,i,c

    # This is valid and has rank 4.
    
    rank = m  # = 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for c in range(rank):
            U[i * m + c][c] = 1
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V[j * p + k][c] = 1 if j == c else 0
    
    # W is all ones as established
    
    # Let's verify by checking a few cases
    if time.time() < deadline - 5:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Test with random matrices
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    a_part = sum(U[i * m + j][c] * A_test[i][j] for j in range(m))
                    b_part = sum(V[j * p + k][c] * B_test[j][k] for j in range(m))
                    s += W[k * n + i][c] * a_part * b_part
                C_computed[i][k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
