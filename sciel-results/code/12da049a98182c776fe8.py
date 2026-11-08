import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # safety margin before 150s kill
    
    # Build the full multiplication tensor T[i, j, k] = delta(i, k) for C = A @ B
    # where A is (n x m), B is (m x p), C is (n x p)
    # T is indexed as T[i, j, k] such that C[i, k] = sum_j A[i, j] * B[j, k]
    # We need to find a rank decomposition: T[i,j,k] = sum_r U[i,j] * V[j,k] * W[r] ... wait
    # Actually the decomposition is: C[i,k] = sum_r (sum_j U[i,j] A[j,r]) (sum_k W[k,r] B[k,r])
    # Let me think more carefully.
    # The tensor T has T[i, j, k] = 1 if i == k (for the diagonal in the "multiplication" sense)
    # Wait, no. For matrix multiplication C = A @ B where A is n x m and B is m x p:
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    # The tensor T[i, j, k] represents the structure such that T[i, j, k] = 1 if we want
    # A[i,j] * B[j,k] to contribute to C[i,k]. So T[i, j, k] = 1 for all i, j, k? No.
    # The rank of the tensor is the minimum r such that T[i,j,k] = sum_{r} U[i,r] * V[j,r] * W[k,r]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r) with the trace convention.
    # Actually, the tensor for matrix multiplication (n x m) * (m x p) -> (n x p) has:
    # T[a, b, c] = delta_{a,c} where a is (i), b is (j), c is (k)? No.
    # The multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if we use A[i,j] and B[j,k] to form C[i,k]
    # So M_{i,j,k} = 1 for all valid i, j, k. Wait, that's not right either.
    # Actually M_{i,j,k} = delta_{i,k} doesn't make sense because i and k have different ranges.
    # The correct tensor is: M[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # and M[i,j,k] = 1 iff the product A[i,j]*B[j,k] contributes to C[i,k]. This is always true,
    # so M[i,j,k] = 1 for all i,j,k. But that can't be right because the rank would be n*m*p.
    # I need to re-examine the tensor definition. The tensor T for matrix multiplication satisfies:
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]. This is always the case.
    # But the rank decomposition is: T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r] where:
    # U has shape (n, r), V has shape (m, r), W has shape (p, r)? No, that doesn't match.
    # Looking at the problem statement: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The decomposition: for each (i, k) in C, C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * ... 
    # I think the correct formulation is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... no.
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The tensor T has T[i, j, k] = delta(i == something) ... 
    # Standard matrix multiplication tensor: T[i, j, k] = 1 if i==k? No, that's for square matrices in a different formulation.
    # For C = A*B where A is (n x m), B is (m x p), C is (n x p):
    # The tensor T of order 3 has T[i, j, k] = 1 for the "interaction" between A[i,j], B[j,k], C[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p). Wait, that's n*m*p ones.
    # But the rank of this all-ones tensor is 1! That can't be the matrix multiplication tensor.
    # I think I'm confusing myself. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M_{i,j,k} = delta_{i,k} but that only works if n=p. 
    # Actually, the correct definition: M_{i,j,k} = 1 iff the term A[i,j] B[j,k] appears in C[i,k].
    # This is true for ALL i,j,k. So M is the all-ones tensor of size n x m x p.
    # But the rank of the all-ones tensor is 1. That's clearly not right for matrix multiplication.
    # I must be misunderstanding the tensor. Let me reconsider.
    # Ah, I see. The tensor M_{n,m,p} for matrix multiplication has:
    # M_{i,j,k} = 1 if i == k (when n=p for square case). For rectangular, it's more subtle.
    # Actually no. The correct tensor is: we want C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T[i,j,k] is defined such that T[i,j,k] = 1 represents the coefficient
    # in the bilinear expression. The "tensor" here is really a 3-tensor with components
    # T[i,j,k] = delta_{?}. I think the standard definition is:
    # T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which is always true.
    # So T[i,j,k] = 1 for all valid indices. The "rank" of this tensor in the sense of
    # tensor rank (minimal number of rank-1 tensors) is indeed 1 (just the all-ones tensor).
    # But that's not what we want. We want the Bilinear Rank, which is different.
    # Bilinear rank: minimal r such that C[i,k] = sum_{r=1}^{r} W_r[k, i] * (sum_j U_r[i,j] A[i,j]) * ... 
    # I'm getting confused. Let me just keep the current working solution which scores 1.0769.
    # The current solution uses rank 39 and works correctly.
    
    # Let me just verify the current approach works and return it, with a deadline guard.
    
    # Rank 39 decomposition:
    # Part 1: 2x2x7 multiplication using 3 Strassen (21) + 4 schoolbook (4) = 25
    # Part 2: 2x1x7 multiplication = 14
    # Total = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] += coeff
        col += 1
    
    def add_strassen_block(k_start):
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])
    
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [1], [b06], [1], [c06], [1])
    add_term([1], [1], [b16], [1], [c06], [1])
    add_term([3], [1], [b06], [1], [c16], [1])
    add_term([4], [1], [b16], [1], [c16], [1])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [1], [b2k], [1], [c0k], [1])
        add_term([5], [1], [b2k], [1], [c1k], [1])
    
    if time.time() - start_time > time_limit:
        raise TimeoutError("Exceeded time limit")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
