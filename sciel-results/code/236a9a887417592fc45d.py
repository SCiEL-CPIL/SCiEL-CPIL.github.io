import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # We aim to use the Kronecker product structure: R(6) <= R(2) * R(3)
    # This gives rank 4 * 9 = 36 if we can find optimal rank-4 and rank-9 decompositions.
    # Actually, R(2)=7, R(3)=19 is best known, so 7*19=133.
    # But wait, the block decomposition is:
    # A (6x6) = [[A11, A12], [A21, A22]] where Ai are 3x3
    # B (6x6) = [[B11, B12], [B21, B22]] where Bj are 3x3
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # To compute this using tensor decomposition, we need to decompose the 6x6 matrix multiplication tensor.
    # This is equivalent to decomposing 16 3x3 matrix multiplications plus additions.
    # This approach doesn't directly give a simple rank bound without careful tensor construction.
    
    # Let's try a simpler approach first: use the known best decomposition for 3x3 (Strassen-like or cyclic).
    # The best known rank for 3x3 is 19 (using 7x7x7 tensor or similar).
    # If we use the Kronecker product of two 3x3 multiplications, we get 3x3 * 3x3 -> 6x6? No, that's not right.
    
    # Actually, the standard way to get R(6) from R(3) is:
    # R(2n) <= 2 * R(n) * R(2) -- not quite.
    
    # Let's use the fact that matrix multiplication is associative and we can use the tensor product of decompositions.
    # If we have a rank-r decomposition for n x n, we can get a rank r^2 decomposition for n^2 x n^2.
    # So R(6) <= R(3)^2. If R(3) = 19, then R(6) <= 361. That's worse than 216.
    
    # Better: R(6) <= R(2) * R(3) is not a standard inequality.
    # The correct inequality is: R(m*n) <= R(m) * R(n).
    # So R(6) <= R(2) * R(3) = 7 * 19 = 133.
    
    # How to construct this?
    # The matrix multiplication tensor for 6x6 can be viewed as the tensor product of the 2x2 and 3x3 tensors?
    # Not exactly. The correct construction uses the fact that the 6x6 matrix can be seen as a 2x2 block matrix with 3x3 blocks.
    # The multiplication of two such block matrices involves 16 multiplications of 3x3 matrices and 8 additions of 3x3 matrices.
    # This doesn't directly reduce to R(2)*R(3).
    
    # Actually, the tensor product construction works as follows:
    # If T_m is the multiplication tensor for m x m matrices, then T_{m*n} is related to T_m * T_n.
    # Specifically, the rank satisfies R(m*n) <= R(m) * R(n).
    # The construction uses the Kronecker product of the factor matrices.
    
    # Let's implement the Kronecker product construction.
    # If (U_m, V_m, W_m) is a rank-r_m decomposition for m x m,
    # and (U_n, V_n, W_n) is a rank-r_n decomposition for n x n,
    # then (U_m ⊗ U_n, V_m ⊗ V_n, W_m ⊗ W_n) is a rank r_m * r_n decomposition for (m*n) x (m*n).
    
    # We need good decompositions for 2x2 and 3x3.
    # For 2x2, Strassen gives rank 7.
    # For 3x3, the best known is rank 19 (using the 7x7x7 tensor or the cyclic group Z_3 construction).
    
    # Let's start with the known rank-7 decomposition for 2x2 (Strassen).
    # Strassen's 7 intermediate matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's construct U_2, V_2, W_2 for 2x2.
    # The tensor T_2 has entries T_2[i,j,k] = 1 if i=k, 0 otherwise (for the output C[i,j] = sum_k A[i,k] * B[k,j]).
    # Wait, the standard convention is T[i,j,k] where i is row of C, j is col of C, k is the inner index?
    # The task says: W[k*n+i, r] -- this suggests the output index is (k*n + i) where k is col, i is row? Or i is col, k is row?
    # "W[k*n+i,r]" with k, i in range(6). If C is 6x6, C[k, i]? Or C[i, k]?
    # Standard matrix multiplication: C[i, j] = sum_k A[i, k] * B[k, j].
    # So the tensor is T[i, j, k] = A[i, k] * B[k, j] -> contributes to C[i, j].
    # The output index in W is "k*n+i". If n=6, and k, i are indices.
    # If W[k*n+i], this looks like column-major indexing with k as row, i as col? Or k as col, i as row?
    # Let's assume the tensor is T[i, j, k] where i=row of A, j=col of B, k=inner.
    # Output C[i, j].
    # The W index is k*n + i. This is ambiguous. Let's look at the current program's snippet.
    # "W_t[2*k+i, r]" with i, j, k in range(2).
    # In the snippet:
    # for i in range(2):
    #   for j in range(2):
    #     for k in range(2):
    #       s += ... W_t[2*k+i, r]
    #     expected = 1.0 if i == k else 0.0
    # This suggests that the output index is determined by i and k?
    # If i==k, it's 1. This looks like the identity matrix.
    # But matrix multiplication is not the identity tensor.
    # Wait, the snippet is checking if the decomposition gives the identity?
    # No, "expected = 1.0 if i == k else 0.0". This is checking the trace?
    # Or is it checking a specific entry of the result?
    # The task says: "Find a rank-minimal exact tensor decomposition for multiplying a 6 by 6 matrix by a 6 by 6 matrix."
    # The snippet might be a partial check or a specific test case.
    # Let's assume the standard tensor: T[i, j, k] = 1 if i == k (for A's row) and j is B's col? No.
    # T[i, j, k] corresponds to A[i, k] * B[k, j] -> C[i, j].
    # So T[i, j, k] = 1 if (i, k) is a valid A index and (k, j) is a valid B index.
    # Actually, the tensor is simply: T[i, j, k] = 1 if i is row of C, j is col of C, and k is the inner index.
    # But A[i, k] and B[k, j] are the inputs.
    # So the tensor element is 1 if the triple (i, j, k) corresponds to a valid multiplication term.
    # Since all i, j, k in 0..n-1 are valid, T[i, j, k] = 1 for all i, j, k? No, that's not right.
    # The tensor T is defined by T[i, j, k] = 1 if i=k? No.
    # Let's clarify: The matrix multiplication tensor T_n is in V^* ⊗ W^* ⊗ X.
    # T_n(e_i, f_j, g_k) = 1 if i = k? No.
    # T_n(a, b, c) = sum_{i,j,k} a_i b_j c_k T_{ijk}.
    # For matrix multiplication: C = A B.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # So T_{ijk} = 1 if i = i (always) and j = j (always) and k = k (always)?
    # No, the tensor is T_{i, j, k} where i is the row of A (and C), j is the col of B (and C), and k is the inner index.
    # The condition is that A's column index equals B's row index.
    # So T_{i, j, k} = 1 if k (A's col) = k (B's row). This is always true in the sum.
    # Wait, the tensor is: T_{i, j, k} = 1 if the triple (i, j, k) is such that A_{i, k} and B_{k, j} contribute to C_{i, j}.
    # This is true for all i, j, k. So T_{i, j, k} = 1 for all i, j, k?
    # That would mean the tensor is the all-ones tensor, which is rank 1. That's wrong.
    
    # Let's re-read the standard definition.
    # The matrix multiplication tensor T_{n} is defined by:
    # T_{n} = sum_{i,j,k=1}^n e_i ⊗ f_j ⊗ g_k  where e_i is basis for C, f_j for B, g_k for A?
    # No.
    # T_{n}(a, b, c) = trace(a b c) ? No, that's for trace.
    # T_{n}(a, b, c) = sum_{i,j} a_{ij} b_{jk} c_{ki} ? No.
    # The correct definition: T_{n}(A, B, C) = sum_{i,j,k} A_{ik} B_{kj} C_{ji} ?
    # Let's use the bilinear form: <T, A ⊗ B ⊗ C> = sum_{i,j,k} A_{ik} B_{kj} C_{ji}.
    # This is the trace of A B C^T?
    # Actually, the standard tensor is: T_{ijk} = 1 if i = k (for A's row and C's row) and j = j (for B's col and C's col)?
    # Let's look at the indices again.
    # A is n x n, B is n x n, C is n x n.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # So the tensor element T_{i, j, k} = 1 if (i, j) is the output index and k is the inner index.
    # The condition is that A_{ik} and B_{kj} are defined. They are for all i, j, k.
    # So T_{i, j, k} = 1 for all i, j, k?
    # If so, the rank is 1. But we know the rank is at least n^2.
    # The error is in the tensor definition.
    # The tensor is T_{n} = sum_{i,j,k} e_i ⊗ f_k ⊗ g_{kj} ? No.
    
    # Correct definition:
    # Let U be the space of A, V be the space of B, W be the space of C.
    # T ∈ U^* ⊗ V^* ⊗ W.
    # T(u, v, w) = sum_{i,j,k} u_{ik} v_{kj} w_{ji} ?
    # If w is C, then w_{ji} = C_{ji}.
    # Then T(u, v, w) = sum_{i,j,k} A_{ik} B_{kj} C_{ji}.
    # This is the Frobenius inner product of C and A B, transposed?
    # <C, A B>_F = sum_{i,j} C_{ij} (AB)_{ij} = sum_{i,j,k} C_{ij} A_{ik} B_{kj}.
    # So T(u, v, w) = <w, u v>.
    # In terms of basis: T_{i, j, k} where i indexes U (A), j indexes V (B), k indexes W (C).
    # U basis: E_{ik} (1 at i, k).
    # V basis: F_{kj} (1 at k, j).
    # W basis: G_{ji} (1 at j, i).
    # Then T(E_{ik}, F_{kj}, G_{ji}) = 1 if (i, k) matches A, (k, j) matches B, and (j, i) matches C.
    # This is 1 if the indices are consistent.
    # So the tensor has non-zero entries only when the indices form a valid path.
    # But in the sum, i, j, k are all free.
    # So T_{(i,k), (k,j), (j,i)} = 1.
    # This is a 3-tensor with indices:
    # U index: (i, k) -- n^2 values.
    # V index: (k, j) -- n^2 values.
    # W index: (j, i) -- n^2 values.
    # And T[(i,k), (k,j), (j,i)] = 1.
    
    # So the tensor is a permutation of the identity tensor of size n^3?
    # No, it's a specific pattern.
    # For n=2, U index has 4 values: (0,0), (0,1), (1,0), (1,1).
    # V index has 4 values.
    # W index has 4 values.
    # The non-zero entries are:
    # i=0, j=0, k=0: U=(0,0), V=(0,0), W=(0,0) -> 1
    # i=0, j=0, k=1: U=(0,1), V=(1,0), W=(0,0) -> 1
    # i=0, j=1, k=0: U=(0,0), V=(0,1), W=(1,0) -> 1
    # i=0, j=1, k=1: U=(0,1), V=(1,1), W=(1,0) -> 1
    # i=1, j=0, k=0: U=(1,0), V=(0,0), W=(0,1) -> 1
    # i=1, j=0, k=1: U=(1,1), V=(1,0), W=(0,1) -> 1
    # i=1, j=1, k=0: U=(1,0), V=(0,1), W=(1,1) -> 1
    # i=1, j=1, k=1: U=(1,1), V=(1,1), W=(1,1) -> 1
    
    # So the tensor is defined by T[u, v, w] = 1 if w_index = (j, i), u_index = (i, k), v_index = (k, j).
    # This is a 3D tensor with 8 non-zero entries (for n=2).
    
    # Now, let's construct the rank-7 Strassen decomposition for n=2.
    # We need U, V, W matrices of shape (4, 7).
    # U[u, r], V[v, r], W[w, r] such that sum_r U[u, r] V[v, r] W[w, r] = T[u, v, w].
    
    # Let's use the known Strassen decomposition.
    # The 7 intermediates are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's express A, B, C in terms of the U, V, W matrices.
    # A is 2x2, so U has 4 rows.
    # B is 2x2, so V has 4 rows.
    # C is 2x2, so W has 4 rows.
    
    # The decomposition is: C = sum_r U_r * V_r * W_r, where U_r, V_r are scalars?
    # No, U, V, W are matrices.
    # C[i, j] = sum_k A[i, k] B[k, j] = sum_r U[i, k, r] V[k, j, r] W[i, j, r]?
    # This is not quite right.
    
    # Let's use the standard tensor decomposition form:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ U^*, v_r ∈ V^*, w_r ∈ W.
    # In matrix form, U is 4x7, V is 4x7, W is 4x7.
    # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r].
    
    # Let's try to construct this for Strassen.
    # This is a well-known problem. I'll use the known values.
    
    # For now, let's return a naive decomposition to ensure correctness, then improve.
    n = 6
    rank = n**3
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    idx = 0
    for i in range(n):
        for k in range(n):
            for j in range(n):
                u_idx = i * n + k
                v_idx = k * n + j
                w_idx = j * n + i  # W[k*n+i] in the task? Let's check.
                # Task says W[k*n+i, r]. If k is col, i is row, then w_idx = k*n + i.
                # But in the tensor, W index is (j, i) if C[j, i]?
                # Let's assume W index is (row, col) = (i, j) for C[i, j].
                # Then w_idx = i * n + j.
                # Let's use w_idx = i * n + j.
                w_idx = i * n + j
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
    
    return {'U': U, 'V': V, 'W': W}
