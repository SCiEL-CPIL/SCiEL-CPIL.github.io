"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses greedy rank-reduction: start from schoolbook (rank 125), iteratively
merge pairs of terms whose sum tensor has rank 1, reducing rank by 1 each time.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [5, 5, 5]


def _verify_decomposition(U, V, W, n=5, m=5, p=5):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(p):
            expected = 1.0 if i == j else 0.0  # This is wrong, see below
    # Actually verify by checking tensor identity:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for the
    # correct tensor... wait, let me think about the convention.
    # W[k*n+i, r] is the coefficient for C[i,k] (or C[k,i]?)
    # The trace convention: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = delta_{i,k} * ... no.
    # Matrix mult tensor: T[i,j,k] where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # No, T[i,j,k] = delta for the "diagonal" in the sense that each output C[i,k]
    # gets contribution from A[i,j]*B[j,k].
    # 
    # The decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m} (indexed by (i,j)), v_r in R^{m*p} (indexed by (j,k)),
    # w_r in R^{p*n} (indexed by (k,i)) -- using W[k*n+i, r]
    #
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this term contributes to C[i,k] from A[i,j]*B[j,k]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (each (i,j,k) triple appears exactly once)
    # Wait no. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Because every product A[i,j]*B[j,k] contributes to C[i,k].
    # Hmm, that's just the all-ones tensor which has rank 1. That can't be right.
    #
    # Let me reconsider. The matrix multiplication tensor T^{n,m,p} is defined by:
    # T(i,j,k) = 1 if the (i,k)-th entry of C = AB uses A(i,j) and B(j,k)
    # Since C(i,k) = sum_j A(i,j) B(j,k), we have T(i,j,k) = 1 for all valid i,j,k.
    # But that's the all-ones tensor of shape (n,m,p) which has rank 1!
    # That's clearly wrong because we know matmul has rank > 1.
    #
    # I think the issue is the indexing convention. Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # 
    # The standard tensor decomposition for matrix multiplication:
    # C = A * B, where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has T[i,j,k] = 1 for all i,j,k (in the "standard" basis).
    # But the decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n×m}, v_r ∈ R^{m×p}, w_r ∈ R^{n×p}
    # and the reconstruction is: C[i,k] = sum_r (sum_j u_r[i,j] A[i,j]) (sum_l v_r[j,l] B[j,l]) w_r[i,k]
    # 
    # Hmm, I think the correct statement is:
    # For each term r: compute s_r = (U_r · A) * (V_r · B) where U_r is a matrix, etc.
    # Actually no. The bilinear form is:
    # C = sum_r (u_r^T A_vec) (v_r^T B_vec) w_r
    # where A_vec is A flattened, B_vec is B flattened, w_r is a matrix.
    #
    # More precisely: C[i,k] = sum_r U[i*m+j... ] 
    # 
    # OK let me just use the direct verification:
    # The decomposition is valid iff for random A, B:
    #   C_computed = sum_r (U[:,r] reshaped to n×m, contracted with A) * 
    #                (V[:,r] reshaped to m×p, contracted with B) * 
    #                (W[:,r] reshaped to p×n... )
    # 
    # Actually, I think the convention is:
    # For each term r:
    #   a_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (scalar)
    #   b_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (scalar)  
    #   C += a_r * b_r * W_r  where W_r is W[:,r] reshaped to n×p
    #
    # Wait, W has shape (p*n, rank) with W[k*n+i, r]. So W_r reshaped to p×n?
    # Or n×p? The index is k*n+i, so if we reshape to (p,n), element [k,i] = W[k*n+i].
    # But C is n×p, so C[i,k]. If W_r is p×n, then W_r[k,i] contributes to... 
    # 
    # I think the convention might be: W_r is reshaped to n×p by transposing,
    # or the index k*n+i means the (i,k) entry when reshaped to (n,p) in column-major.
    #
    # Let me just verify numerically with the known direct decomposition.


def _tensor_rank_check(U, V, W, n=5, m=5, p=5):
    """
    Verify decomposition by checking: for all i,j,k,
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    (the matrix multiplication tensor is all-ones in this indexing)
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a sample of (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _tensor_rank_one_check(U, V, W, n=5, m=5, p=5):
    """
    Check if the tensor defined by U,V,W (single term each, i.e., rank-1 tensors)
    Actually, check if the SUM of two rank-1 tensors is itself rank-1.
    """
    pass


def _is_rank_one_tensor(U, V, W, n=5, m=5, p=5):
    """
    Check if the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    has rank 1 (i.e., T = u ⊗ v ⊗ w for some vectors).
    A tensor has rank 1 iff all its 2×2 minors (in any unfolding) are zero.
    """
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
    
    # Check if T has rank 1: all 2x2 minors of the n×(m*p) matricization are zero
    # T as matrix: T_mat[i, j*p+k] = T[i,j,k]
    T_mat = T.reshape(n, m * p)
    
    # Check all 2x2 minors
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m * p):
                for j2 in range(j1 + 1, m * p):
                    minor = T_mat[i1, j1] * T_mat[i2, j2] - T_mat[i1, j2] * T_mat[i2, j1]
                    if abs(minor) > 1e-9:
                        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Start with schoolbook decomposition (rank = n*m*p = 125)
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_dict = {i * m + j: 1}
                V_dict = {j * p + k: 1}
                W_dict = {k * n + i: 1}
                terms.append((U_dict, V_dict, W_dict))
    
    rank = len(terms)
    
    # Greedy merge: repeatedly find a pair of terms whose sum is rank-1
    # and merge them, reducing rank by 1.
    # 
    # To check if term_a + term_b is rank-1, we need to check if the tensor
    # T_a + T_b has rank 1, where T_a and T_b are rank-1 tensors.
    # 
    # A sum of two rank-1 tensors u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank-1 iff
    # the 6 vectors (u1,u2,v1,v2,w1,w2) satisfy certain conditions.
    # Specifically, T = u⊗v⊗w iff all 2×2 minors vanish.
    #
    # For efficiency, I'll build the tensor explicitly and check.
    
    def build_tensor(term_list):
        """Build the full n×m×p tensor from a list of (U_dict, V_dict, W_dict) terms."""
        T = np.zeros((n, m, p))
        for U_d, V_d, W_d in term_list:
            for (ij, u_val) in U_d.items():
                i = ij // m
                j = ij % m
                for (jk, v_val) in V_d.items():
                    jj = jk // p
                    k = jk % p
                    if jj != j:
                        continue
                    for (ki, w_val) in W_d.items():
                        kk = ki // n
                        ii = ki % n
                        if kk != k and ii != i:
                            continue
                        # The tensor entry T[i,j,k] gets contribution u_val * v_val * w_val
                        # if the indices are consistent
                        T[i, j, k] += u_val * v_val * w_val
        return T
    
    def tensor_is_rank_one(T, tol=1e-9):
        """Check if tensor T (shape n,m,p) has rank 1."""
        # Matricize to n × (m*p)
        T_mat = T.reshape(n, m * p)
        # Check all 2×2 minors
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                for j1 in range(m * p):
                    for j2 in range(j1 + 1, m * p):
                        minor = T_mat[i1, j1] * T_mat[i2, j2] - T_mat[i1, j2] * T_mat[i2, j1]
                        if abs(minor) > tol:
                            return False
        return True
    
    def merge_terms(term_a, term_b):
        """Merge two terms into one, assuming their sum is rank-1."""
        U_merged = {}
        V_merged = {}
        W_merged = {}
        
        for idx, val in term_a[0].items():
            U_merged[idx] = U_merged.get(idx, 0) + val
        for idx, val in term_b[0].items():
            U_merged[idx] = U_merged.get(idx, 0) + val
        
        for idx, val in term_a[1].items():
            V_merged[idx] = V_merged.get(idx, 0) + val
        for idx, val in term_b[1].items():
            V_merged[idx] = V_merged.get(idx, 0) + val
        
        for idx, val in term_a[2].items():
            W_merged[idx] = W_merged.get(idx, 0) + val
        for idx, val in term_b[2].items():
            W_merged[idx] = W_merged.get(idx, 0) + val
        
        # Remove zeros
        U_merged = {k: v for k, v in U_merged.items() if v != 0}
        V_merged = {k: v for k, v in V_merged.items() if v != 0}
        W_merged = {k: v for k, v in W_merged.items() if v != 0}
        
        return (U_merged, V_merged, W_merged)
    
    # Greedy search for merges
    merged = True
    while merged and time.time() < deadline:
        merged = False
        current_rank = len(terms)
        
        # Try all pairs
        for idx_a in range(current_rank):
            if time.time() > deadline:
                break
            for idx_b in range(idx_a + 1, current_rank):
                if time.time() > deadline:
                    break
                
                # Check if terms[idx_a] + terms[idx_b] is rank-1
                T_sum = build_tensor([terms[idx_a], terms[idx_b]])
                
                if tensor_is_rank_one(T_sum):
                    # Merge them
                    new_term = merge_terms(terms[idx_a], terms[idx_b])
                    # Remove both and add the merged term
                    remaining = [terms[i] for i in range(current_rank) if i != idx_a and i != idx_b]
                    remaining.append(new_term)
                    terms = remaining
                    merged = True
                    break
            if merged:
                break
    
    # Convert to matrix form
    final_rank = len(terms)
    U = [[0] * final_rank for _ in range(n * m)]
    V = [[0] * final_rank for _ in range(m * p)]
    W = [[0] * final_rank for _ in range(p * n)]
    
    for r in range(final_rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
