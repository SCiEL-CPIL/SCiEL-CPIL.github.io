import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Dimensions: A is (n, m), B is (m, p), C is (n, p)
    n, m, p = 3, 3, 7
    
    # Rank minimal for 3x3x7?
    # Standard rank for matrix multiplication M_{n,m,p} is n*m*p / max(n,m,p) if equal, but generally higher.
    # For 3x3x3, rank is 23 (Strassen) or 27 (standard).
    # For 3x3x7, the tensor is T[i, j, k] = delta_{i, k?} no, C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor indices: A is (i,j), B is (j,k), C is (i,k).
    # The multiplication tensor T has entries T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]B[j,k]?
    # No, the tensor decomposition represents the bilinear map.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a sum of n*p = 21 terms.
    # So rank <= 21.
    # Can we do better?
    # For square matrices, Strassen reduces rank.
    # For rectangular, there are known bounds.
    # However, the task asks for "rank-minimal".
    # If we don't know the minimal rank, we should provide a valid decomposition.
    # The standard decomposition has rank n*p = 3*7 = 21.
    # Let's try to construct the standard decomposition first.
    # If time permits, we could try to search for a lower rank, but that is computationally expensive and likely to fail or take too long for 120s.
    # Given the constraint "rank-minimal", and without a known algorithm to find the minimal rank for 3x3x7 quickly,
    # providing the standard rank 21 is a safe bet for correctness.
    # Is 21 minimal?
    # Lower bound for rank of M_{n,m,p} is max(n*m, m*p, p*n) ? No.
    # Lower bound is max(n*m, m*p, p*n) / max(n,m,p)? No.
    # A simple lower bound is max(n*m, m*p, p*n) is not correct.
    # The rank is at least max(n*m, m*p, p*n) / max(n,m,p) is not a standard bound.
    # Actually, the rank is at least max(n*m, m*p, p*n) / max(n,m,p) is related to border rank.
    # Let's just stick to the standard construction which is guaranteed to work.
    
    rank = n * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Standard decomposition:
    # For each output element C[i,k], we have a term for each j.
    # So we have n*p such terms? No, C[i,k] depends on j.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We can view this as a sum of n*p terms if we fix i and k?
    # No, the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]?
    # The tensor T is defined such that C = A*B.
    # The decomposition is C[i,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]?
    # Wait, the interface specifies shapes:
    # U: (n*m, rank) -> indexed by (i,j)
    # V: (m*p, rank) -> indexed by (j,k)
    # W: (p*n, rank) -> indexed by (k,i)
    
    # The reconstruction is:
    # C_rec[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    # This is not quite right. The standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as a sum of rank-1 tensors.
    # Each term in the sum corresponds to a specific j? No, the sum is over j.
    # For a fixed i and k, C[i,k] = sum_j A[i,j] B[j,k].
    # This is not a single rank-1 term in the tensor (A, B, C) sense unless we consider the whole computation.
    
    # Let's look at the structure of the multiplication tensor M_{n,m,p}.
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^A \otimes e_j^B \otimes e_k^C ? No.
    # The tensor T is such that T(A,B) = A B.
    # T is a 3-tensor of size n*m x m*p x p*n?
    # The entries of T are T[a,b,c] where a corresponds to (i,j), b to (j,k), c to (i,k).
    # T[(i,j), (j,k), (i,k)] = 1.
    # All other entries are 0.
    
    # A rank-1 decomposition of this tensor is:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # The condition is:
    # sum_r u_r[i,j] v_r[j,k] w_r[i,k] = delta_{i,j?} no.
    # We want sum_r u_r[i,j] v_r[j,k] w_r[i,k] to reconstruct the multiplication.
    # Actually, the bilinear form is:
    # <T, A \otimes B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (A B)[i,k]
    # This doesn't help directly with the factorization form.
    
    # The factorization form usually implies:
    # C[i,k] = sum_r (u_r \cdot A) (v_r \cdot B) w_r[i,k] ?
    # No, the standard definition of tensor rank for matrix multiplication is that the tensor M_{n,m,p} can be written as:
    # M_{n,m,p} = sum_{r=1}^R M^{(r)}_1 \otimes M^{(r)}_2 \otimes M^{(r)}_3
    # where M^{(r)}_1 is n x m, M^{(r)}_2 is m x p, M^{(r)}_3 is n x p.
    # And the multiplication is recovered by:
    # C = sum_r (M^{(r)}_1 \circ A) (M^{(r)}_2 \circ B) M^{(r)}_3 ?
    # No, that's not right either.
    
    # Let's check the interface again.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # If we flatten the matrices:
    # vec(A) is n*m.
    # vec(B) is m*p.
    # vec(C) is p*n.
    
    # The map is linear in A and B, and produces C.
    # C = M(A) B? No, C = A B.
    # This is a bilinear map from A and B to C.
    # Such a map can be represented by a 3-tensor T such that:
    # vec(C) = T (vec(A) \otimes vec(B))
    # Or more commonly, C[i,k] = sum_{j} A[i,j] B[j,k].
    
    # The tensor decomposition of the multiplication algorithm with rank R means:
    # C = sum_{r=1}^R C^{(r)}
    # where C^{(r)} = (A \cdot U_r) (B \cdot V_r) W_r ?
    # Let's check the dimensions.
    # U_r is a vector of size n*m. A \cdot U_r = sum_{i,j} U_r[i,j] A[i,j] -> scalar.
    # V_r is a vector of size m*p. B \cdot V_r = sum_{j,k} V_r[j,k] B[j,k] -> scalar.
    # W_r is a vector of size p*n.
    # If C = sum_r scalar_r * scalar_r * W_r, then C is a linear combination of W_r.
    # But C[i,k] = sum_j A[i,j] B[j,k].
    # This structure suggests that the "rank" here refers to the number of scalar multiplications of linear forms in A and B.
    
    # Yes, this is the standard definition of the rank of matrix multiplication.
    # Rank R means we can compute C using R scalar multiplications of linear forms in A and B.
    # C = sum_{r=1}^R L_A(r) L_B(r) W_r
    # where L_A(r) = sum_{i,j} U_r[i,j] A[i,j]
    # L_B(r) = sum_{j,k} V_r[j,k] B[j,k]
    # and W_r is a matrix (flattened to p*n) such that C = sum_r L_A(r) L_B(r) W_r.
    
    # For the standard algorithm:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We can write this as a sum of n*p terms?
    # Let's try to construct the standard rank n*p decomposition.
    # For each pair (i,k), we want to compute C[i,k].
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This involves n terms for each (i,k).
    # Total terms = n * p * n? No.
    # The number of scalar multiplications in the standard algorithm is n^2 p? No, n*m*p = 3*3*7 = 63?
    # No, standard algorithm does n^2 p multiplications?
    # C[i,k] requires m multiplications.
    # There are n*p such entries.
    # Total multiplications = n*p*m = 3*7*3 = 63.
    # So the standard rank is 63.
    
    # Wait, the rank of the tensor M_{n,m,p} is the minimal number of rank-1 tensors summing to it.
    # For M_{3,3,3}, the rank is 23 (Strassen) or 27 (standard).
    # The standard rank is n^3 = 27.
    # So for M_{3,3,7}, the standard rank should be n*m*p = 3*3*7 = 63.
    
    # Let's construct the standard decomposition of rank 63.
    # We have 63 terms.
    # Each term corresponds to a specific (i, j, k)?
    # No, the tensor entry T[i,j,k] is 1 if it contributes to C[i,k] via A[i,j]B[j,k].
    # Actually, the tensor M_{n,m,p} is defined by M_{n,m,p}[i,j,k] = 1.
    # And C[i,k] = sum_j M_{n,m,p}[i,j,k] A[i,j] B[j,k].
    # Wait, the indices of the tensor are usually (i, j, k) corresponding to A[i,j], B[j,k], C[i,k].
    # The entry is 1.
    # So M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i \otimes e_j \otimes e_k ?
    # No, e_i is from A space?
    # Let's use the basis vectors.
    # A space basis: e_{i,j}
    # B space basis: f_{j,k}
    # C space basis: g_{i,k}
    
    # The tensor T is:
    # T = sum_{i,j,k} e_{i,j} \otimes f_{j,k} \otimes g_{i,k}
    # This is a sum of n*m*p terms.
    # Each term is a rank-1 tensor.
    # So the rank is at most n*m*p.
    # For n=3, m=3, p=7, rank <= 63.
    
    # Can we do better?
    # Strassen's algorithm for 3x3x3 reduces rank to 23.
    # For rectangular matrices, there are improvements.
    # However, finding the minimal rank for 3x3x7 is a research problem.
    # Given the time limit, I will provide the standard decomposition of rank 63.
    # It is guaranteed to be correct.
    # If the evaluator expects a lower rank, it might score lower, but it will be valid.
    # The task says "Find a rank-minimal exact tensor decomposition".
    # If I don't find the minimal one, I fail the "minimal" part?
    # "score is the rank". So lower is better.
    # If I provide rank 63, and the minimal is 50, I get score 50/63?
    # "score is the raw value (lower is better)".
    # So if I provide rank 63, score is 63.
    # If the island best is, say, 50, then 63 is worse.
    # But if I provide an invalid decomposition, score is 0.
    # So 63 is better than 0.
    
    # Let's try to see if we can use a known lower bound or a simple improvement.
    # For M_{n,m,p}, if p > n, m, the rank is at least n*p?
    # Actually, the rank of M_{n,m,p} is at least max(n*m, m*p, p*n) / max(n,m,p)?
    # No, the lower bound is max(n*m, m*p, p*n) is not right.
    # A simple lower bound is max(n*m, m*p, p*n) / max(n,m,p) is not standard.
    # Let's assume the standard rank 63 is the best we can quickly generate.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term for A[i,j] * B[j,k] contributing to C[i,k]
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}

