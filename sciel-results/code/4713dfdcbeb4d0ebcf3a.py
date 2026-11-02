"""Tensor decomposition search for matrix multiplication 3x3 * 3x6."""
import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 6]

def _build_matrices(n, m, p, rank, alpha, beta, gamma):
    """Construct U, V, W from factor coefficients."""
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                if idx >= rank:
                    return None, None, None
                U[i*m+j, idx] = alpha[idx]
                V[j*p+k, idx] = beta[idx]
                W[k*n+i, idx] = gamma[idx]
    return U, V, W

def _check_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta[i,k]."""
    # The tensor T_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal delta_{i,k} (independent of j).
    total_terms = 0
    max_rank = W.shape[1]
    for c in range(max_rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # Only consider non-zero entries
        nz_u = np.nonzero(uc)[0]
        nz_v = np.nonzero(vc)[0]
        nz_w = np.nonzero(wc)[0]
        if len(nz_u) == 0 or len(nz_v) == 0 or len(nz_w) == 0:
            continue
        total_terms += 1
        for i_u in nz_u:
            for j_v in nz_v:
                for k_w in nz_w:
                    # i_u = i*m + j, j_v = j*p + k, k_w = k'*n + i'
                    i = i_u // m
                    j_u = i_u % m
                    j = j_v // p
                    k = j_v % p
                    i2 = k_w % n
                    k2 = k_w // n
                    if j_u == j and i == i2:
                        # This term contributes to tensor[i, j, k]
                        pass
    # Actually, let's just compute the full contraction for small cases
    T = np.zeros((n, p))  # T[i,k] should be delta[i,k]
    # T[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for c in range(W.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # s should equal 1 if k == i? Wait, the target is A(3x3) @ B(3x6) -> C(3x6)
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is T_{i,j,k} = delta_{i,k}? No.
            # The multiplication tensor has T_{i,j,k} = 1 if k==i (in the output index sense)? 
            # No. The standard decomposition: M_{i,j,k} = 1 means C[i,k] += A[i,j]*B[j,k].
            # So the tensor is T_{i,j,k} = delta_{i,k}? No, it's delta_{i, k} only in specific conventions.
            # Actually, the matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k? No.
            # T_{i,j,k} = 1 if the output index k equals... wait.
            # C[i, k] = sum_j A[i, j] B[j, k].
            # So the tensor entries are T_{i,j,k} = 1 for all valid (i,j,k)? No!
            # The tensor is: T = sum_{i,j,k} e_i (in output1) ⊗ e_j (in input1) ⊗ e_k (in input2/output2)
            # The standard fact: C = A * B corresponds to tensor M where M_{i,j,k} = 1 
            # IF the output coordinate i and input coordinate k are related.
            # Actually for general matrices, M_{i,j,k} = 1 for all i, j, k is WRONG.
            # M_{i,j,k} = 1 means: coefficient of (A_i ⊗ B_k) in output position... 
            # Let's be precise. We want C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # The tensor is T = sum_{i,j,k} delta_{ik} * A_i ⊗ B_j ⊗ C_k ? No.
            # The matrix multiplication tensor is defined such that:
            # <T, a ⊗ b> = A(a) b, where A is matrix, a is vector... 
            # In 3-tensor form: T_{i,j,k} = 1 if i == k? No.
            # T_{i,j,k} = 1 for ALL i, j, k? 
            # If T_{i,j,k} = 1, then sum_j T_{i,j,k} a_j b_k = sum_j a_j b_k. That's not matrix mult.
            # 
            # Correct: The tensor for C = AB is T_{i,j,k} = 1 if and only if the indices align.
            # Actually, in the standard convention: T_{i,j,k} = 1 for all i,j,k where
            # the "output" index i and "input2" index k are free, and "input1" index j is summed.
            # Wait, the tensor T in C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k} requires T_{i,j,k} = 1 for all i,j,k? 
            # No! C_{i,k} = sum_j A_{i,j} B_{j,k}. So T_{i,j,k} = 1 for all i,j,k.
            # Yes! The matrix multiplication tensor is the all-ones tensor? No, that's wrong.
            #
            # Let me re-derive. We have three tensors: U (n*m x rank), V (m*p x rank), W (p*n x rank).
            # The decomposition asserts: M = sum_c U_c ⊗ V_c ⊗ W_c
            # where U_c is vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}.
            # M_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
            # For this to equal the matrix multiplication tensor, we need:
            # M_{i,j,k} = 1 if and only if... 
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # In tensor form: C = sum_{i,k} C_{i,k} e_i ⊗ f_k
            # Also C = A * B = sum_{i,j,k} A_{i,j} B_{j,k} e_i ⊗ f_k
            # So the bilinear form gives: <M, a ⊗ b> = A(a) b where a is vector in R^m, b in R^p?
            # No, a in R^{n*m}, b in R^{m*p}.
            # <M, a ⊗ b> = sum_{i,j,k} M_{i,j,k} a_{i*m+j} b_{j*p+k}
            # This should equal <A(a_vec), b_vec> where a_vec is reshaped to n x m, b to m x p,
            # and A(a_vec) b_vec gives n x p, and we need to match with W indexing.
            #
            # The key identity: For the schoolbook, U_{i*m+j, (i,j,k)} = 1, V_{j*p+k, (i,j,k)} = 1, W_{k*n+i, (i,j,k)} = 1.
            # Then M_{i',j',k'} = sum_{c} U_{i'*m+j', c} V_{j'*p+k', c} W_{k'*n+i', c}
            # Non-zero only if there exists c = (i,j,k) such that i*m+j = i'*m+j', j*p+k = j'*p+k', k*n+i = k'*n+i'.
            # This means i=i', j=j', k=k'.
            # So M_{i',j',k'} = 1 if (i',j',k') is a valid index, else 0.
            # Wait, that gives M_{i,j,k} = 1 for all valid (i,j,k). 
            # But that's the all-ones tensor, which is NOT matrix multiplication!
            #
            # I think I'm confusing the conventions. Let me re-read the task:
            # "W[k*n+i,r]" - this is the index convention for W.
            # The decomposition: M_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For schoolbook, this gives M_{i,j,k} = 1 for all (i,j,k) in range.
            #
            # But matrix multiplication tensor should satisfy:
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # Which in tensor form with the given indexing means:
            # M_{i,j,k} = 1? No, that would give sum_j 1 * A_{i,j} * B_{j,k} = sum_j A_{i,j} B_{j,k}. 
            # YES! So M_{i,j,k} = 1 for all valid (i,j,k) IS the matrix multiplication tensor!
            #
            # Wait, is that right? Let's check:
            # <M, a ⊗ b> where a_{i*m+j} = A_{i,j}, b_{j*p+k} = B_{j,k}
            # = sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k} = (AB)_{i,k}
            # Yes! So the target tensor IS the all-ones tensor (for valid indices).
            #
            # But that can't be right either, because then the rank of the all-ones tensor 
            # for any dimensions would be... well, the all-ones tensor of size n*m*p has rank?
            # The all-ones tensor has a decomposition of rank 1? No, because 
            # sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r} can't be made 1 for all i,j,k with rank 1.
            # With rank 1: M_{i,j,k} = u_{i,j} v_{j,k} w_{k,i}. This can't be 1 for all i,j,k 
            # unless n=m=p=1.
            #
            # I think the issue is that the all-ones tensor DOES have higher rank.
            # The rank of the all-ones tensor of size n×m×p is... at most n*m, m*p, p*n?
            #
            # Actually, I realize the schoolbook gives rank n*m*p with M_{i,j,k} = 1.
            # This IS a valid decomposition of the all-ones tensor.
            #
            # So the task is to find a rank-minimal decomposition of the all-ones tensor 
            # T_{i,j,k} = 1 for all valid (i,j,k) with the specific index structure.

    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(W.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T[i,j,k] = s
    
    expected = np.ones((n, m, p))
    if np.allclose(T, expected, atol=1e-8):
        return True
    return False

def _rank_via_np(U, V, W, n, m, p):
    """Compute the tensor T and check if it's all ones."""
    T = np.zeros((n*m, p*n))  # T[(i*m+j), (k*n+i)]... no, let's just do 3D
    T3 = np.zeros((n, m, p))
    for r in range(W.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # T3[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T3[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    return np.allclose(T3, np.ones((n,m,p)), atol=1e-7)

def construct():
    start_time = time.time()
    DEADLINE = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook: rank = n*m*p = 54
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def _to_list(mat):
        return mat.tolist()
    
    # Return schoolbook as fallback
    best = {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}
    best_rank = rank_schoolbook
    
    # Try rank 36 (2x2x2 block Strassen-like)
    # Actually, let's try to find lower rank using random search
    # For rank r, we need to find U, V, W such that the tensor equals all-ones.
    
    # Strategy: Try rank values from 30 up to 53, random initialization, gradient descent
    import numpy as np
    
    for target_rank in range(30, 54):
        if time.time() > DEADLINE:
            break
        
        found = False
        for trial in range(50):
            if time.time() > DEADLINE:
                break
            
            rng = np.random.default_rng(trial + target_rank * 1000)
            
            U = rng.standard_normal((nm, target_rank))
            V = rng.standard_normal((mp, target_rank))
            W = rng.standard_normal((pn, target_rank))
            
            # Normalize
            for c in range(target_rank):
                nU = np.linalg.norm(U[:, c])
                nV = np.linalg.norm(V[:, c])
                nW = np.linalg.norm(W[:, c])
                if nU > 0 and nV > 0 and nW > 0:
                    scale = (nU * nV * nW) ** (1.0/3.0)
                    U[:, c] /= scale ** (1.0/3.0)
                    V[:, c] /= scale ** (1.0/3.0)
                    W[:, c] /= scale ** (1.0/3.0)
            
            # Gradient descent
            lr = 0.01
            for step in range(5000):
                if time.time() > DEADLINE:
                    break
                
                # Compute T
                T = np.zeros((n, m, p))
                for c in range(target_rank):
                    u = U[:, c]
                    v = V[:, c]
                    w = W[:, c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
                
                error = np.sum((T - 1.0) ** 2)
                if error < 1e-6:
                    found = True
                    break
                
                if step % 500 == 0:
                    lr *= 0.95
                
                # Compute gradients
                dU = np.zeros((nm, target_rank))
                dV = np.zeros((mp, target_rank))
                dW = np.zeros((pn, target_rank))
                
                # dT[i,j,k] = 2 * (T[i,j,k] - 1)
                dT = 2.0 * (T - 1.0)
                
                for c in range(target_rank):
                    u = U[:, c]
                    v = V[:, c]
                    w = W[:, c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = dT[i,j,k]
                                if val != 0:
                                    dU[i*m+j, c] += val * v[j*p+k] * w[k*n+i]
                                    dV[j*p+k, c] += val * u[i*m+j] * w[k*n+i]
                                    dW[k*n+i, c] += val * u[i*m+j] * v[j*p+k]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                # Verify
                if _rank_via_np(U, V, W, n, m, p):
                    best_rank = target_rank
                    best = {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
                    break
        
        if time.time() > DEADLINE:
            break
        
        if best_rank == target_rank:
            # Found exact solution at this rank, stop
            break
    
    if time.time() > DEADLINE - 5:
        return best
    
    return best
