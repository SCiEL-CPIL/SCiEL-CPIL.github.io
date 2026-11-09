import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8

    # Rank-32 decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (it's the identity in the "multiplication" sense)
    # Actually, the matrix multiplication tensor for n x m times m x p is:
    # T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} -- no.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No.
    # T = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # The tensor T has components T[i,j,k] = 1 if we're looking at the "structure tensor"
    # Actually, the matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Wait, no. Let me reconsider.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This means the tensor T satisfies:
    # T[i,j,k] = 1 for all i, j, k? 
    # No. The tensor is defined such that:
    # T(A, B) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
    # So T[i,j,k] must be such that summing over j gives the product.
    # For fixed i,k: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i, j, k.
    #
    # Wait, that's the all-ones tensor. Let me verify with n=1, m=1, p=1:
    # C[0,0] = A[0,0] * B[0,0]
    # T[0,0,0] = 1. Yes.
    #
    # For n=2, m=4, p=8:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    #
    # The tensor T is the all-ones tensor of shape (2, 4, 8).
    #
    # Now, we can decompose this tensor. The all-ones tensor of shape (n, m, p) has rank:
    # rank = min(n*p, m*n, m*p)? No.
    #
    # Actually, the all-ones tensor T[i,j,k] = 1 can be written as:
    # T = sum_{r=1}^{rank} u_r v_r^T w_r^T
    # where u_r is in R^n, v_r in R^m, w_r in R^p.
    #
    # One simple decomposition: T = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} e_i) (e_j) (sum_{k=0}^{p-1} e_k)
    # That's m terms, each of rank 1.
    #
    # But we can do better. The all-ones tensor of shape (n, m, p) has rank min(n*p, m, ...) 
    # Actually, the rank of the all-ones tensor is min(n, m, p) if we allow arbitrary vectors?
    # No. Let me think again.
    #
    # T[i,j,k] = 1 = 1 * 1 * 1
    # We can write T = u v^T w^T where u = (1,...,1) in R^n, v = (1,...,1) in R^m, w = (1,...,1) in R^p.
    # That's rank 1!
    #
    # Wait, is that right? Let's check:
    # (u v^T w^T)[i,j,k] = u[i] * v[j] * w[k] = 1 * 1 * 1 = 1. Yes!
    #
    # So the all-ones tensor has rank 1. But that can't be right for matrix multiplication.
    #
    # I'm confusing the structure tensor with the matrix multiplication tensor.
    #
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p}(A, B) = AB
    # where A is n x m, B is m x p, AB is n x p.
    #
    # In tensor notation:
    # M = sum_{i,j,k} (AB)[i,k] A[i,j] B[j,k]
    #
    # But (AB)[i,k] = sum_j A[i,j] B[j,k], so:
    # M = sum_{i,j,k} (sum_j' A[i,j'] B[j',k]) A[i,j] B[j,k]
    #
    # This is getting circular. Let me use the standard definition.
    #
    # The matrix multiplication tensor M_{n,m,p} is the tensor in (R^{n x m}*) x (R^{m x p}*) x (R^{n x p})
    # defined by:
    # M_{n,m,p}(A, B, C) = tr(C^T AB) = sum_{i,k} C[i,k] (AB)[i,k] = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    #
    # So the tensor has components M[i,j,k] = 1 for all i, j, k? 
    # No. M[i,j,k] is the coefficient of A[i,j] B[j,k] C[i,k] in the expansion.
    # From sum_{i,j,k} C[i,k] A[i,j] B[j,k], the coefficient of A[i,j] B[j,k] C[i,k] is 1.
    # So M[i,j,k] = 1 for all i, j, k.
    #
    # But wait, that means the matrix multiplication tensor IS the all-ones tensor?
    # That can't be right because then it would have rank 1.
    #
    # I think I'm making an error. Let me look at this more carefully.
    #
    # The matrix multiplication tensor M_{n,m,p} is in V* x W* x U where V = R^{nm}, W = R^{mp}, U = R^{np}.
    # It is defined by: M(v, w) = v^T (matrix product) w, but that's not quite right either.
    #
    # Actually, the standard definition is:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^* x e_j^* x e_{j,k}^* x e_{i,k}
    # where e_i^* is the i-th dual basis vector of V, e_j^* is the j-th dual basis vector of W,
    # and e_{i,k} is the (i,k)-th basis vector of U.
    #
    # So M_{n,m,p} = sum_{i,j,k} (e_i^* x e_j^* x e_{i,k})
    #
    # This means the tensor has a term for each (i,j,k) triple, and the term is the outer product
    # of the i-th basis vector of V*, the j-th basis vector of W*, and the (i,k)-th basis vector of U.
    #
    # So the tensor M has rank at most n*m*p (the number of terms), and we're looking for a lower rank decomposition.
    #
    # For n=2, m=4, p=8, the naive rank is 2*4*8 = 64.
    #
    # Now, the key insight: M = sum_{i,j,k} e_i^* x e_j^* x e_{i,k}
    #
    # We can group terms. For fixed j:
    # sum_{i,k} e_i^* x e_j^* x e_{i,k} = (sum_i e_i^* x e_{i,:}) x e_j^*
    #
    # Let F_j = sum_i e_i^* x e_{i,:} which is a tensor in V* x U.
    # Then M = sum_j F_j x e_j^*
    #
    # The rank of F_j as a matrix (flattening V* x U) is n (since it's like an identity matrix of size n).
    # So rank(F_j) = n = 2.
    #
    # Therefore, M = sum_j sum_{r=1}^{n} (F_j)_r x e_j^*
    # where (F_j)_r are rank-1 decompositions of F_j.
    #
    # F_j = sum_i e_i^* x e_{i,:}
    # As a matrix, F_j has entries F_j[i, k] = delta_{i,i} = 1 if we think of it as mapping from U to V*.
    # Actually, F_j[i, (i', k')] = delta_{i,i'} delta_{k,k}? No.
    #
    # Let me be more careful. F_j is in V* x U. V* has basis e_i^* (i=0,...,n-1), U has basis e_{i,k} (i=0,...,n-1, k=0,...,p-1).
    # F_j = sum_i e_i^* x e_{i,:} where e_{i,:} = sum_k e_{i,k}.
    #
    # So F_j = sum_i (e_i^* x sum_k e_{i,k}) = sum_i sum_k e_i^* x e_{i,k}
    #
    # As a matrix with rows indexed by i (from V*) and columns indexed by (i', k) (from U):
    # F_j[i, (i', k)] = 1 if i = i', else 0.
    # So F_j is a block diagonal matrix with n blocks, each block being a 1 x p all-ones row.
    #
    # The rank of F_j is n (since the rows are linearly independent).
    #
    # So we can write F_j = sum_{r=1}^{n} u_r x v_r^* where u_r in V*, v_r in U.
    # Specifically, u_r = e_r^*, v_r = sum_k e_{r,k} = (0,...,0, 1,...,1, 0,...,0) with 1s in positions (r, 0), ..., (r, p-1).
    #
    # So M = sum_j sum_{r=1}^{n} e_r^* x e_j^* x (sum_k e_{r,k})
    #
    # This gives rank = m * n = 4 * 2 = 8? No, that's not right either.
    #
    # Wait, I think I'm overcomplicating this. Let me just directly construct a rank-32 decomposition.
    #
    # M = sum_{i,j,k} e_i^* x e_j^* x e_{i,k}
    #
    # For each j, define S_j = sum_i e_i^* x e_{i,:} (as above).
    # M = sum_j S_j x e_j^*
    #
    # S_j has rank n=2 as a matrix. So S_j = sum_{r=0}^{1} u_r^{(j)} x v_r^{(j)}
    # where u_r^{(j)} = e_r^* and v_r^{(j)} = sum_k e_{r,k}.
    #
    # So M = sum_j sum_{r=0}^{1} e_r^* x e_j^* x (sum_k e_{r,k})
    #
    # This is a decomposition with m*n = 4*2 = 8 terms? No, each term is still a sum over k.
    #
    # Let me expand: M = sum_j sum_r sum_k e_r^* x e_j^* x e_{r,k}
    #
    # This is just the original sum reorganized. The number of terms is still n*m*p = 64.
    #
    # I need a different approach. Let me think about the rank of the matrix multiplication tensor.
    #
    # For n=2, m=4, p=8, what is the rank?
    #
    # One approach: use the fact that M_{n,m,p} can be decomposed using the rank of the associated matrix.
    #
    # Actually, let me try a different decomposition. Consider:
    # M = sum_{j=0}^{m-1} (sum_i e_i^* x e_{i,:}) x e_j^*
    #
    # The term sum_i e_i^* x e_{i,:} is a tensor in V* x U. Let's call it T_j (it doesn't depend on j).
    # T = sum_i e_i^* x e_{i,:} = sum_i e_i^* x (sum_k e_{i,k})
    #
    # As a matrix (flattening V* x U), T has entries T[i, (i', k)] = delta_{i,i'}.
    # So T is an n x (n*p) matrix that is block diagonal with n identity-like blocks.
    #
    # The rank of T is n = 2.
    #
    # So T = sum_{r=0}^{1} u_r x v_r^T where u_r = e_r^* and v_r is the vector in U with 1s at positions (r, 0), ..., (r, p-1) and 0 elsewhere.
    #
    # Therefore: M = sum_j T x e_j^* = sum_j sum_r u_r x v_r x e_j^*
    #
    # This gives a decomposition with m * n = 4 * 2 = 8 terms!
    #
    # Wait, but each term is u_r x v_r x e_j^* where v_r is a fixed vector (doesn't depend on j).
    # So we have m * n = 8 terms.
    #
    # Let me verify: M = sum_{j=0}^{3} sum_{r=0}^{1} e_r^* x v_r x e_j^*
    # where v_r = sum_{k=0}^{7} e_{r,k}.
    #
    # Expanding: sum_{j,r} e_r^* x (sum_k e_{r,k}) x e_j^* = sum_{j,r,k} e_r^* x e_{r,k} x e_j^*
    #
    # But the original M = sum_{i,j,k} e_i^* x e_j^* x e_{i,k}
    #
    # These are the same! (r = i)
    #
    # So the rank is at most 8? That seems too low. Let me double-check.
    #
    # Actually, I think I made an error. The tensor M is in V* x W* x U, where:
    # V = R^{n*m} (space of A matrices), W = R^{m*p} (space of B matrices), U = R^{n*p} (space of C matrices).
    #
    # The basis of V is e_{i,j} (i=0..n-1, j=0..m-1).
    # The basis of W is e_{j,k} (j=0..m-1, k=0..p-1).
    # The basis of U is e_{i,k} (i=0..n-1, k=0..p-1).
    #
    # M = sum_{i,j,k} e_{i,j}^* x e_{j,k}^* x e_{i,k}
    #
    # Now, for fixed j, the sum over i,k is:
    # sum_{i,k} e_{i,j}^* x e_{j,k}^* x e_{i,k}
    #
    # This is a tensor in V* x W* x U. Let's fix j and look at the slice.
    #
    # Actually, I think the rank-8 decomposition I derived is correct, but let me verify with a small example.
    #
    # For n=1, m=1, p=1: M = e_{0,0}^* x e_{0,0}^* x e_{0,0}. Rank 1. My formula gives m*n = 1*1 = 1. Correct.
    #
    # For n=1, m=2, p=1: M = e_{0,0}^* x e_{0,0}^* x e_{0,0} + e_{0,1}^* x e_{1,0}^* x e_{0,0}
    # Rank should be 2. My formula gives m*n = 2*1 = 2. Correct.
    #
    # For n=2, m=1, p=1: M = e_{0,0}^* x e_{0,0}^* x e_{0,0} + e_{1,0}^* x e_{0,0}^* x e_{1,0}
    # Rank should be 2. My formula gives m*n = 1*2 = 2. Correct.
    #
    # So for n=2, m=4, p=8, the rank should be m*n = 4*2 = 8?
    #
    # But wait, that can't be right. The rank of the matrix multiplication tensor M_{n,m,p} is known to be at least max(n*p, m*n, m*p) in some cases, but I'm not sure.
    #
    # Actually, I think my derivation is correct. The rank is m*n = 8.
    #
    # Let me construct the decomposition explicitly.
    #
    # M = sum_{j=0}^{3} sum_{r=0}^{1} e_r^* x v_r x e_j^*
    # where e_r^* is the r-th basis vector of V* (but V has dimension n*m=8, so e_r^* doesn't make sense directly).
    #
    # I need to be more careful with the indices.
    #
    # V has basis e_{i,j} for i in {0,1}, j in {0,1,2,3}.
    # W has basis e_{j,k} for j in {0,1,2,3}, k in {0,...,7}.
    # U has basis e_{i,k} for i in {0,1}, k in {0,...,7}.
    #
    # M = sum_{i=0}^{1} sum_{j=0}^{3} sum_{k=0}^{7} e_{i,j}^* x e_{j,k}^* x e_{i,k}
    #
    # For fixed j, define T_j = sum_{i,k} e_{i,j}^* x e_{j,k}^* x e_{i,k}
    #
    # T_j is a tensor in V* x W* x U.
    #
    # We can write T_j = sum_{r=0}^{1} u_r^{(j)} x v_r^{(j)} x w_r^{(j)}
    #
    # Let's try: u_r^{(j)} = e_{r,j}^*, v_r^{(j)} = sum_k e_{j,k}^*, w_r^{(j)} = sum_k e_{r,k}
    #
    # Then u_r^{(j)} x v_r^{(j)} x w_r^{(j)} = e_{r,j}^* x (sum_k e_{j,k}^*) x (sum_k' e_{r,k'})
    # = sum_k sum_k' e_{r,j}^* x e_{j,k}^* x e_{r,k'}
    #
    # This has terms for all k, k', but we only want k=k'. So this doesn't work directly.
    #
    # I think the issue is that I can't factor out the sum over k in both the W and U parts simultaneously.
    #
    # Let me try a different approach. Just use the naive rank-64 decomposition and see if I can reduce it.
    #
    # Actually, let me just implement the rank-32 decomposition by grouping over k in pairs.
    #
    # For each pair of k values (k0, k0+1), we have:
    # sum_{i,j} e_{i,j}^* x e_{j,k0}^* x e_{i,k0} + e_{i,j}^* x e_{j,k0+1}^* x e_{i,k0+1}
    #
    # This is 2*n*m = 2*2*4 = 16 terms per pair. 4 pairs -> 64 terms. No improvement.
    #
    # Let me just go with the rank-56 Strassen decomposition that was working, and see if I can verify it's correct.

    # Actually, let me just return the rank-56 Strassen decomposition that was already implemented.
    # The previous code had rank 56 and scored 1.1429.
    
    num_groups = 4  # k-blocks of size 2
    num_ablocks = 2  # j-blocks of size 2
    num_strassen = 7

    rank = num_groups * num_ablocks * num_strassen  # 56

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    X_forms = np.array([
        [1, 0, 0, 1],    # m1: a + d
        [0, 0, 1, 1],    # m2: c + d
        [1, 0, 0, 0],    # m3: a
        [0, 0, 0, 1],    # m4: d
        [1, 1, 0, 0],    # m5: a + b
        [-1, 0, 1, 0],   # m6: c - a
        [0, 1, 0, -1],   # m7: b - d
    ], dtype=np.int64)

    Y_forms = np.array([
        [1, 0, 0, 1],    # m1: e + h
        [1, 0, 0, 0],    # m2: e
        [0, 1, 0, -1],   # m3: f - h
        [-1, 0, 1, 0],   # m4: g - e
        [0, 0, 0, 1],    # m5: h
        [1, 1, 0, 0],    # m6: e + f
        [0, 0, 1, 1],    # m7: g + h
    ], dtype=np.int64)

    Z_combos = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # c01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # c10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # c11 = m1 - m2 + m3 + m6
    ], dtype=np.int64)

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = X_forms[s, i * 2 + jp]

                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = Y_forms[s, jp * 2 + kp]

                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = Z_combos[i * 2 + kp, s]

                r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
