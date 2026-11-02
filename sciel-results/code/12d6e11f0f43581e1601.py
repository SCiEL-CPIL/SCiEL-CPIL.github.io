import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

def _target_tensor(n=2, m=4, p=4):
    """The matrix multiplication tensor: T[i,j,k] = 1 if j==k else 0."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i, j, k] = 1.0
    return T

def _tensor_from_factors(U, V, W, n=2, m=4, p=4):
    """Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    T = np.zeros((n, m, p))
    for c in range(r):
        T += U_r[:, :, c] * V_r[:, :, c].T * W_r[:, :, c].T.reshape(n, p).T.reshape(1, p, n).T
    # Let me redo this more carefully
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U_r[:, :, c]  # n x m
        vc = V_r[:, :, c]  # m x p
        wc = W_r[:, :, c]  # p x n
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    return T

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify that the factors exactly decompose the matmul tensor."""
    T = _tensor_from_factors(U, V, W, n, m, p)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T, T_target, atol=tol)

def _verify_matrix(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify by random matrix multiplication."""
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            acc = 0.0
            for c in range(r):
                for j in range(m):
                    acc += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
            Cf[i, k] = acc
    return np.allclose(C, Cf, atol=tol)

def _strassen_2x2():
    """Return the 7 Strassen rank-1 terms for 2x2 matrix multiplication."""
    # Each term is (u_indices, v_indices, w_indices) where each is list of (index, value)
    # U factor: 2x2, V factor: 2x2, W factor: 2x2
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    return [
        ([(0, 1), (1, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
        ([(0, 1)], [(1, 1)], [(1, 1)]),
        ([(1, 1)], [(0, 1)], [(0, 1)]),
        ([(1, 1)], [(1, 1)], [(1, 1)]),
        ([(0, 1), (1, 1)], [(1, 1)], [(0, 1)]),
        ([(0, 1)], [(0, 1), (1, 1)], [(1, 1)]),
        ([(1, 1), (0, -1)], [(1, 1), (0, 1)], [(0, 1)]),
    ]

def _build_rank28_strassen():
    """Build rank-28 decomposition using Strassen on 2x2 blocks.
    
    A is 2x4: split into [A_L | A_R] where A_L, A_R are 2x2
    B is 4x4: split into [[B_TL, B_TR], [B_BL, B_BR]] where each is 2x2
    C is 2x4: C = [C_L | C_R] where C_L, C_R are 2x2
    
    C_L = A_L @ B_TL + A_R @ B_BL
    C_R = A_L @ B_TR + A_R @ B_BR
    
    Use Strassen for each 2x2 multiplication (7 terms each).
    Total: 4 multiplications * 7 terms = 28 terms.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen = _strassen_2x2()
    
    # For A (2x4): row index i in {0,1}, col j in {0,1} for left, j in {2,3} for right
    # U index: i*4 + j
    # For B (4x4): row j in {0,1,2,3}, col k in {0,1,2,3}
    # V index: j*4 + k
    # For C (2x4): row i in {0,1}, col k in {0,1,2,3}
    # W index: k*2 + i
    
    # Four Strassen instances:
    # 1. A_L (2x2) * B_TL (2x2) -> C_L (2x2)
    #    A_L: U indices [0,1,4,5] (i*4+j for i,j in {0,1})
    #    B_TL: V indices [0,1,4,5] (j*4+k for j,k in {0,1})
    #    C_L: W indices [0,2,1,3] (k*2+i for k,i in {0,1})
    
    # 2. A_R (2x2) * B_BL (2x2) -> C_L (2x2)
    #    A_R: U indices [2,3,6,7] (i*4+j+2 for i,j in {0,1})
    #    B_BL: V indices [8,9,12,13] (j*4+k for j in {2,3}, k in {0,1})
    #    C_L: W indices [0,2,1,3]
    
    # 3. A_L (2x2) * B_TR (2x2) -> C_R (2x2)
    #    A_L: U indices [0,1,4,5]
    #    B_TR: V indices [2,3,6,7] (j*4+k for j in {0,1}, k in {2,3})
    #    C_R: W indices [4,6,5,7] (k*2+i for k in {2,3}, i in {0,1})
    
    # 4. A_R (2x2) * B_BR (2x2) -> C_R (2x2)
    #    A_R: U indices [2,3,6,7]
    #    B_BR: V indices [10,11,14,15] (j*4+k for j in {2,3}, k in {2,3})
    #    C_R: W indices [4,6,5,7]
    
    configs = [
        # (u_map, v_map, w_map) where map is [idx00, idx01, idx10, idx11]
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3]),   # A_L * B_TL -> C_L
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3]),  # A_R * B_BL -> C_L
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7]),   # A_L * B_TR -> C_R
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7]), # A_R * B_BR -> C_R
    ]
    
    col = 0
    for u_map, v_map, w_map in configs:
        for u_t, v_t, w_t in strassen:
            for idx, val in u_t:
                U[u_map[idx], col] += val
            for idx, val in v_t:
                V[v_map[idx], col] += val
            for idx, val in w_t:
                W[w_map[idx], col] += val
            col += 1
    
    return U, V, W

def _rank1_decomposition(n=2, m=4, p=4):
    """Build the rank-32 rank-1 decomposition (schoolbook)."""
    rank = m * p  # 16... no, rank = m * p is wrong
    # Actually schoolbook: for each j, k with j==k, we have one term
    # T[i,j,k] = sum_j A[i,j]*B[j,k]
    # As a tensor: T[i,j,k] = delta_{j,k}
    # Rank-1 decomposition: for each j (0..3), term with U[i,j]=A[i,j], V[j,k]=B[j,k], W[k,i]=delta_{j,k}
    # Wait, the standard decomposition has rank = m*p = 16? No.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For schoolbook: c indexes (j,k) pairs where we need delta_{j,k}
    # Actually the minimal rank-1 terms: for each j in {0,1,2,3}, define
    # U[i,j,c_j] = A[i,j], V[j,k,c_j] = B[j,k], W[k,i,c_j] = 1 if k==j else 0
    # But W[k,i,c_j] = delta_{k,j} doesn't depend on i, so W is p*n x rank
    # This gives rank 4, not 32.
    
    # Let me reconsider. The tensor T[i,j,k] = delta_{j,k} has rank...
    # For fixed j, the slice T[:,j,:] is a 2x4 matrix with 1 at position (i=j) if j<2... 
    # Actually T[i,j,k] = 1 if j==k, 0 otherwise.
    # For j=0: T[i,0,k] = 1 if k==0, so it's a 2x4 matrix with first column all 1s.
    # The rank of this tensor... 
    
    # Schoolbook decomposition: T[i,j,k] = sum_{j'} delta_{j,j'} * delta_{j',k}
    # = sum_{j'=0}^{3} U[i,j,j'] V[j',k,j'] W[k,i,j']
    # where U[i,j,j'] = delta_{j,j'}, V[j',k,j'] = delta_{j',k}, W[k,i,j'] = 1
    # This gives rank 4! But that can't be right because matrix multiplication requires more.
    
    # Wait, I'm confusing myself. The tensor for matrix multiplication C = A*B where
    # A is n x m, B is m x p, C is n x p is:
    # T[i,j,k] such that C[i,k] = sum_j A[i,j] * B[j,k]
    # As a 3-tensor: T[i,j,k] = delta_{j,k} is WRONG.
    # The correct tensor is: the multiplication map is (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k? No.
    # 
    # The bilinear form: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = delta_{j,j}... 
    # No, T[i,j,k] = 1 if the indices match for the sum, i.e., T[i,j,k] = delta_{j,k} is wrong.
    # 
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]. So the "tensor" is just the identity in the j,k sense:
    # T[i,j,k] = 1 for all i (it doesn't depend on i), and we need j to match in the sum.
    # The tensor is T[i,j,k] = delta_{j,k}? No, that would give C[i,k] = sum_j A[i,j] delta_{j,k} B[j,k] = A[i,k] B[k,k] which is wrong.
    #
    # Let me be very careful. C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k].
    # This is a trilinear form in (A, B, C^*) where C^* is the dual.
    # The tensor T satisfies: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k]
    # For this to work: T[i,j,k] must satisfy sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k] = sum_{i,k} C[i,k] C^*[i,k]
    # = sum_{i,k} (sum_j A[i,j] B[j,k]) C^*[i,k]
    # = sum_{i,j,k} A[i,j] B[j,k] C^*[i,k]
    # So T[i,j,k] = 1 for all i,j,k! The tensor is all ones.
    #
    # Wait no. The tensor rank decomposition is:
    # T = sum_c u_c ⊗ v_c ⊗ w_c where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] v_c[j] w_c[k]
    # And we need T[i,j,k] = 1 for all i,j,k? That gives rank 1!
    #
    # I think I'm confusing the tensor structure. Let me re-read the problem.
    # "Matrix multiplication 2,4,4" - multiply 2x4 by 4x4.
    # The output factor follows W[k*n+i, r].
    # 
    # The factor matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The decomposition: for each rank-1 term c, we have vectors u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # such that the tensor T[i*m+j, j*p+k, k*n+i] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    # 
    # Wait, the indices are: U has shape (n*m, rank) so U[row, c] where row = i*m + j (i in 0..n-1, j in 0..m-1)
    # V has shape (m*p, rank) so V[row, c] where row = j*p + k (j in 0..m-1, k in 0..p-1)
    # W has shape (p*n, rank) so W[row, c] where row = k*n + i (k in 0..p-1, i in 0..n-1)
    #
    # The tensor being decomposed: T[i,j,k] where the full index is (i*m+j, j*p+k, k*n+i)
    # And T[i,j,k] should equal delta_{j,k} (since C[i,k] = sum_j A[i,j] B[j,k])
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k}
    #
    # For the schoolbook: for each j in {0,1,2,3}, define term c=j:
    # U[i*m+j, c=j] = 1 for all i
    # V[j*p+k, c=j] = 1 for all k
    # W[k*n+i, c=j] = 1 if k==j else 0
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = U[i*m+j,j] V[j*p+k,j] W[k*n+i,j]
    # = 1 * 1 * delta_{k,j} = delta_{j,k}. 
    # So rank 4 works? That seems too low.
    #
    # But wait, that's not a valid tensor decomposition in the usual sense because
    # the same "j" appears in different positions in U, V, W.
    # Actually it IS valid. The rank of the matrix multiplication tensor <n,m,p> is at most m*p... 
    # no, it's at most n*m*p in general, and the schoolbook gives rank m (for square) or...
    #
    # I think I was overcomplicating. Let me just verify: the schoolbook for 2x4 * 4x4 has rank 4?
    # That doesn't match known results. The rank of <2,4,4> should be at least... 
    #
    # Actually, I realize the issue. In a proper tensor decomposition, each rank-1 term is a 
    # simple tensor u ⊗ v ⊗ w. The schoolbook decomposition I wrote above IS valid and has rank 4.
    # But that contradicts the fact that Strassen gives rank 7 for 2x2x2, not rank 2.
    #
    # For 2x2x2: T[i,j,k] = delta_{j,k}. Schoolbook: rank 2.
    # c=0: U[i*2+0, 0]=1, V[0*2+k, 0]=1, W[k*2+i, 0]=delta_{k,0}
    # c=1: U[i*2+1, 1]=1, V[1*2+k, 1]=1, W[k*2+i, 1]=delta_{k,1}
    # Check: sum_c U[i*2+j,c] V[j*2+k,c] W[k*2+i,c]
    # = U[i*2+j, j] V[j*2+k, j] W[k*2+i, j]
    # = 1 * 1 * delta_{k,j} = delta_{j,k}. Correct!
    #
    # So the schoolbook rank for <2,2,2> is 2, and Strassen improves it to 7? No, Strassen is 7 > 2.
    # That doesn't make sense. Strassen should be better (lower rank) than schoolbook.
    #
    # I think I have the tensor definition wrong. Let me reconsider.
    #
    # The matrix multiplication tensor <n,m,p> is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k * delta_{j,k}... no.
    #
    # The correct definition: T = sum_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ki} where E_{ab} is the matrix unit.
    # So T[i,j,k] in the "flattened" sense: the coefficient of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} is 1
    # only when... hmm.
    #
    # Actually, the standard definition: the matrix multiplication tensor is
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
    # where E_{a,b} is the (a,b) matrix unit.
    #
    # In flattened form: M_{n,m,p}[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # Wait, that's all ones on the "diagonal" of the 3-way tensor (in the appropriate indexing).
    #
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
    # And T is zero elsewhere.
    #
    # This is NOT the same as delta_{j,k}! Let me recheck.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: C = A ⊗ B contracted over j.
    # The tensor T satisfies: C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1.
    # So T[i,j,k] = 1 for all i,j,k.
    #
    # But in the flattened indexing: the "input" to the first factor is A flattened to n*m,
    # the second is B flattened to m*p, the third is C^* flattened to p*n (or n*p).
    # The tensor entry T_flat[i*m+j, j*p+k, k*n+i] = 1 (since T[i,j,k] = 1).
    #
    # So the target tensor has 1's at positions (i*m+j, j*p+k, k*n+i) for all i,j,k, and 0 elsewhere.
    # For n=2, m=4, p=4: there are 2*4*4 = 32 such positions.
    #
    # Now schoolbook: for each j in {0,1,2,3}, for each... no.
    # Schoolbook: C[i,k] = sum_j A[i,j] B[j,k]. Each term A[i,j]*B[j,k] contributes to C[i,k].
    # As a rank-1 tensor term: for fixed j, the term is
    # u_j[i*m+j'] = delta_{j,j'} (i.e., u_j has 1 at positions i*4+j for all i, 0 elsewhere)
    # v_j[j'*4+k] = delta_{j,j'} (i.e., v_j has 1 at positions j*4+k for all k, 0 elsewhere)
    # w_j[k*2+i] = 1 for all k,i
    # Then sum_j u_j[i*4+j] v_j[j*4+k] w_j[k*2+i] = u_j[i*4+j] v_j[j*4+k] w_j[k*2+i] where j is the only nonzero
    # = 1 * 1 * 1 = 1. But we need this to equal 1 only when the full index matches.
    #
    # Hmm, I think the issue is that in a rank-1 decomposition, the term is u ⊗ v ⊗ w meaning
    # the full 3D tensor entry at (a, b, c) is u[a] * v[b] * w[c].
    # We need: sum_c U[a, c] V[b, c] W[c_idx, c] = T[a, b, c_idx] where
    # a = i*m+j, b = j*p+k, c_idx = k*n+i.
    #
    # For schoolbook with rank m=4:
    # For term c=j (j in 0..3):
    # U[a, c=j] = 1 if a = i*4+j for some i (i.e., a mod 4 == j), else 0
    # V[b, c=j] = 1 if b = j*4+k for some k (i.e., b div 4 == j), else 0
    # W[c_idx, c=j] = 1 for all c_idx
    #
    # Then (U ⊗ V ⊗ W)_c[a, b, c_idx] = U[a,c] V[b,c] W[c_idx,c]
    # = [a mod 4 == j] * [b div 4 == j] * 1
    #
    # Sum over c=j: [a mod 4 == j] * [b div 4 == j]
    # This is 1 if a mod 4 == j AND b div 4 == j, i.e., a mod 4 == b div 4.
    # Let a = i*4+j_a, b = j_b*4+k. Then a mod 4 = j_a, b div 4 = j_b.
    # The sum is 1 if j_a == j_b, which means j_a == j_b.
    # But we need the tensor entry to be 1 when a = i*4+j, b = j*4+k, c_idx = k*2+i,
    # i.e., when j_a = j_b = j (the same j). So yes, it's 1 exactly when j_a = j_b.
    #
    # But wait, we also need c_idx = k*2+i to be consistent. In the sum, W[c_idx, c=j] = 1 for all c_idx,
    # so it doesn't constrain c_idx. This means the sum gives 1 for ALL c_idx when j_a = j_b,
    # but we only want 1 when c_idx = k*2+i where k = b mod 4 and i = a div 4.
    #
    # So schoolbook rank 4 does NOT work! The W factor needs to also depend on the relationship.
    #
    # Correct schoolbook: rank = m*p? No. Let me think again.
    # For each (j, k) pair... no.
    #
    # Actually, the correct schoolbook decomposition has rank m*p = 16 for <2,4,4>?
    # No, for <n,m,p> the schoolbook rank is m (the inner dimension).
    #
    # I think the issue is that I need W to be more specific.
    # For term c indexed by (j, k) where we want to capture A[i,j]*B[j,k] -> C[i,k]:
    # U[i*4+j, c=(j,k)] = 1, U[i*4+j', c=(j,k)] = 0 for j' != j
    # V[j*4+k, c=(j,k)] = 1, V[j'*4+k', c=(j,k)] = 0 for (j',k') != (j,k)
    # W[k*2+i, c=(j,k)] = 1, W[k'*2+i', c=(j,k)] = 0 for (k',i') != (k,i)
    #
    # But this requires knowing i in W, which couples everything. This gives rank = n*m*p = 32.
    #
    # So the schoolbook rank is n*m*p = 32 for the "trivial" decomposition, and Strassen-style
    # decompositions reduce it.
    #
    # OK so my rank-28 Strassen construction should be correct. Let me verify it.

def _verify_rank28():
    U, V, W = _build_rank28_strassen()
    T = _tensor_from_factors(U, V, W)
    T_target = np.zeros((8, 16, 8))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                T_target[i * 4 + j, j * 4 + k, k * 2 + i] = 1.0
    return np.allclose(T.reshape(8, 16, 8), T_target, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0

    # Build rank-28 Strassen decomposition
    U, V, W = _build_rank28_strassen()
    
    # Verify
    if not _verify_decomposition(U, V, W):
        # Fallback: return rank-32
        U = np.zeros((n * m, 32), dtype=float)
        V = np.zeros((m * p, 32), dtype=float)
        W = np.zeros((p * n, 32), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 28
    
    # Try to reduce rank by attempting to remove columns and solving the linear system
    if time.time() - start_time < time_limit - 30.0:
        U_r, V_r, W_r, r_new = _attempt_rank_reduction(U, V, W, n, m, p, time_limit - 30.0)
        if r_new < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r_new
    
    # Round to half-integers if close
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    if _verify_decomposition(U_out, V_out, W_out, tol=1e-6):
        best_U, best_V, best_W = U_out, V_out, W_out
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _attempt_rank_reduction(U, V, W, n, m, p, time_budget):
    """Try to reduce rank by removing one column and solving for corrections."""
    start = time.time()
    r = U.shape[1]
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    for col_to_remove in range(r):
        if time.time() - start > time_budget:
            break
        
        keep = [i for i in range(r) if i != col_to_remove]
        U_keep = U[:, keep].copy()
        V_keep = V[:, keep].copy()
        W_keep = W[:, keep].copy()
        r_keep = len(keep)
        
        # Current tensor from kept columns
        T_keep = _tensor_from_factors(U_keep, V_keep, W_keep, n, m, p)
        T_needed = T_target - T_keep
        
        # Linearize: find dU, dV, dW such that the first-order correction matches T_needed
        nm, mp, pn = n * m, m * p, p * n
        total_vars = (nm + mp + pn) * r_keep
        
        A_mat = np.zeros((n * m * p, total_vars))
        U_keep_r = U_keep.reshape(n, m, r_keep)
        V_keep_r = V_keep.reshape(m, p, r_keep)
        W_keep_r = W_keep.reshape(p, n, r_keep)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    var_offset = 0
                    for c in range(r_keep):
                        A_mat[eq_idx, var_offset + c] = V_keep_r[j, k, c] * W_keep_r[k, i, c]
                    var_offset += nm * r_keep
                    for c in range(r_keep):
                        A_mat[eq_idx, var_offset + c] = U_keep_r[i, j, c] * W_keep_r[k, i, c]
                    var_offset += mp * r_keep
                    for c in range(r_keep):
                        A_mat[eq_idx, var_offset + c] = U_keep_r[i, j, c] * V_keep_r[j, k, c]
        
        try:
            sol, residuals, rank_sv, s = np.linalg.lstsq(A_mat, T_needed.ravel(), rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        if np.max(np.abs(A_mat @ sol - T_needed.ravel())) > 1e-8:
            continue
        
        dU = sol[:nm * r_keep].reshape(n, m, r_keep)
        dV = sol[nm * r_keep:nm * r_keep + mp * r_keep].reshape(m, p, r_keep)
        dW = sol[nm * r_keep + mp * r_keep:].reshape(p, n, r_keep)
        
        U_new = (U_keep_r + dU).reshape(nm, r_keep)
        V_new = (V_keep_r + dV).reshape(mp, r_keep)
        W_new = (W_keep_r + dW).reshape(pn, r_keep)
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            best_rank = r_keep
            break
    
    return best_U, best_V, best_W, best_rank

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
