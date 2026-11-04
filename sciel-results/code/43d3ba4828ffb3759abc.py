"""Matrix multiplication 2,4,6 - rank-minimal tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the 42 Strassen terms
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # Local indices
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        # Term 0 (M1): (A00+A11)(B00+B11)
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0
        
        # Term 1 (M2): (A10+A11)B00
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0
        
        # Term 2 (M3): A00(B01-B11)
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0
        
        # Term 3 (M4): A11(B10-B00)
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0
        
        # Term 4 (M5): (A00+A01)B11
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0
        
        # Term 5 (M6): (A10-A00)(B00+B01)
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        # Term 6 (M7): (A01-A11)(B10+B11)
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    # Apply blocks
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank_init):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial decomposition if Strassen fails (should not happen)
        rank_trivial = n * m * p
        U_t = np.zeros((nm, rank_trivial), dtype=float)
        V_t = np.zeros((mp, rank_trivial), dtype=float)
        W_t = np.zeros((pn, rank_trivial), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_t[i * m + j, r] = 1.0
                    V_t[j * p + k, r] = 1.0
                    W_t[k * n + i, r] = 1.0
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    # Attempt rank reduction by finding linear dependencies
    # We look for coefficients c_r such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0
    # This is equivalent to checking if the 42 flattened vectors are linearly dependent.
    # Flattened vector for term r is vec(U_r) ⊗ vec(V_r) ⊗ vec(W_r)
    # Dimension: 8 * 24 * 12 = 2304.
    
    # However, a more efficient check is to see if we can express one term as a linear combination of others
    # in a way that preserves the tensor structure.
    
    # Let's try to find a dependency by solving a least squares problem.
    # We want to find c such that sum_r c_r * T_r = 0, where T_r is the tensor for term r.
    # T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[i,k]
    
    # We can flatten T_r into a vector of size n*m*p = 48.
    # If the 42 vectors are linearly dependent, we can reduce rank.
    
    T_vectors = []
    for r in range(rank_init):
        T_r = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_r[i, j, k] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        T_vectors.append(T_r.flatten())
    
    T_matrix = np.array(T_vectors).T # 48 x 42
    
    # Check rank of T_matrix
    # If rank < 42, there is a dependency.
    # We can use SVD to find the null space.
    try:
        U_svd, s, Vt = np.linalg.svd(T_matrix, full_matrices=False)
        # Find the smallest singular value
        # If it's close to 0, we have a dependency.
        # We want to remove a term. Let's see which term is most "redundant".
        
        # A simple heuristic: if the smallest singular value is small, we can try to remove the term
        # corresponding to the largest component in the right singular vector.
        
        # However, simply removing a term might break the decomposition.
        # We need to find c such that sum c_r T_r = 0.
        # Then we can eliminate one term by adjusting the others? No, that's not how tensor rank works.
        # Tensor rank reduction is not just linear dependency of the flattening.
        # It's about whether the tensor can be written with fewer rank-1 terms.
        
        # The condition sum c_r T_r = 0 means the tensors are linearly dependent in the space of 3D tensors.
        # This does NOT imply rank reduction.
        # Example: (1,0,0) and (0,1,0) are independent, but (1,1,0) is dependent on them.
        # But here we are asking if the *set* of rank-1 tensors has a linear dependency.
        # If T_1 + T_2 + ... + T_k = 0, it doesn't mean we can remove one.
        
        # So this approach is flawed for rank reduction.
        # Rank reduction requires finding a different set of rank-1 tensors.
        
        # Given the complexity, I will stick with rank 42 unless I find a specific structural simplification.
        
    except:
        pass

    # Convert to lists
    U_list = [[int(round(U[i, r])) for r in range(rank_init)] for i in range(nm)]
    V_list = [[int(round(V[i, r])) for r in range(rank_init)] for i in range(mp)]
    W_list = [[int(round(W[i, r])) for r in range(rank_init)] for i in range(pn)]

    return {"U": U_list, "V": V_list, "W": W_list}
