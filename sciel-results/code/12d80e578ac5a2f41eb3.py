import time
import sys
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build the rank-44 solution first
    rank = 44
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Now try to find a lower-rank solution by exploiting the rectangular geometry.
    # The multiplication tensor T_{i,a,k} = 1 for all i in {0,1}, a in {0,1,2}, k in {0..7}
    # can be viewed as a direct sum over k-pairs of four 2x3x2 sub-tensors.
    # Each 2x3x2 sub-tensor has rank at most 6 (since 2*3*2=12, but we can do better).
    # Actually, for 2x3x2, the rank is known to be 6 (not 7 as one might guess from Strassen).
    # Wait, let's reconsider. The tensor for multiplying 2x3 by 3x2 has dimensions 2*3*2=12.
    # A rank-6 decomposition would be optimal if it exists.
    # Let's try to construct a rank-6 decomposition for 2x3x2.
    # The tensor is T_{i,a,k} = 1 for i in {0,1}, a in {0,1,2}, k in {0,1}.
    # This is the outer product of three vectors? No, it's not separable.
    # But we can try to find a rank-6 decomposition.
    # Actually, the rank of the 2x3x2 multiplication tensor is 6.
    # Let's verify this by trying to construct it.
    # If we can get rank 6 for each of the 4 blocks, total rank = 24.
    # This would be a huge improvement over 44.

    # Let's try to find a rank-6 decomposition for 2x3x2.
    # The tensor entries are T[i,a,k] = 1 for all i,a,k.
    # We need U (6x6), V (6x6), W (4x6) such that sum_r U[i,a,r]*V[a,k,r]*W[k,i,r] = 1.
    # Wait, the dimensions are: U is (n*m, rank) = (6, rank), V is (m*p, rank) = (6, rank), W is (p*n, rank) = (4, rank).
    # For 2x3x2: n=2, m=3, p=2. So U is 6xR, V is 6xR, W is 4xR.
    # Let's try R=6.

    # We can use a simple ansatz: since all entries are 1, maybe we can use a symmetric decomposition.
    # Alternatively, we can use the fact that 2x3x2 is small and brute-force search for a rank-6 solution.
    # But that might be too slow. Let's try a structured approach.
    # One idea: use 3 terms for k=0 and 3 terms for k=1, but that would be rank 6 only if the terms for k=0 and k=1 are independent.
    # Actually, the tensor is T[i,a,k] = 1. This is the same as the tensor for 2x3x1 multiplied by 2 (for k=0 and k=1).
    # Wait, no. T[i,a,k] = 1 for all k. So it's like having two identical 2x3x1 tensors.
    # The rank of 2x3x1 is 2*3=6 (schoolbook), but maybe it can be done in less?
    # Actually, 2x3x1 is just the outer product of a 2x3 matrix and a 1-vector, so its rank is the rank of the 2x3 matrix, which is 2.
    # Wait, no. The tensor for multiplying 2x3 by 3x1 is T[i,a,k] = A[i,a]*B[a,k]. Since B is a vector, this is just A[i,a]*b_a.
    # So the tensor is T[i,a,0] = A[i,a]*b_a. This is not a multiplication tensor in the usual sense because the output is a 2x1 vector.
    # The rank of this tensor is the rank of the matrix A, which is 2. So we can do it in rank 2.
    # But wait, the tensor is T[i,a,k] = 1 for all i,a,k. This is not the same as A[i,a]*b_a.
    # The tensor for matrix multiplication is T[i,a,k] = 1 if we are computing C[i,k] = sum_a A[i,a]B[a,k].
    # So for 2x3x1, T[i,a,0] = 1 for all i,a. This is a 2x3x1 tensor with all entries 1.
    # This tensor is separable: T[i,a,0] = u_i * v_a * w_0, where u_i=1, v_a=1, w_0=1. So rank 1.
    # Wait, that's not right. The tensor is T[i,a,k] = 1 for all i,a,k. For 2x3x1, k is only 0.
    # So T[i,a,0] = 1. This is indeed separable: T[i,a,0] = 1*1*1. So rank 1.
    # But that can't be right because the matrix multiplication tensor is not separable.
    # I think I'm confusing the tensor with the multiplication operation.
    # The multiplication tensor for 2x3 by 3x1 is a 2x3x1 tensor where T[i,a,k] = 1 if and only if the (i,k)-th entry of the product depends on A[i,a] and B[a,k].
    # Since B is 3x1, k is only 0. So T[i,a,0] = 1 for all i,a.
    # So the tensor is indeed all ones. And it is separable: T[i,a,0] = (1 for i)*(1 for a)*(1 for k). So rank 1.
    # But that would mean we can compute C[i,0] = sum_a A[i,a]*B[a,0] with rank 1, which is impossible because it's a dot product of length 3.
    # The issue is that the tensor T[i,a,k] = 1 represents the fact that C[i,k] = sum_a A[i,a]B[a,k].
    # So for each (i,k), the sum is over a. The tensor T[i,a,k] is 1 for all i,a,k, meaning that every A[i,a] and B[a,k] contributes to C[i,k].
    # The rank of this tensor is the minimum number of terms in a decomposition C[i,k] = sum_r U[i,a,r]*V[a,k,r]*W[k,i,r].
    # For 2x3x1, this is the minimum rank of the 2x3x1 all-ones tensor.
    # This tensor is not separable because it's a 3-way tensor. The rank of the all-ones 2x3x1 tensor is 2 (since it's a matrix of all ones, which has rank 1, but as a 3-way tensor, the rank is the matrix rank, which is 1? No, the tensor rank is not the same as matrix rank).
    # Actually, the all-ones 2x3x1 tensor can be written as sum_{r=1}^2 u_r[i] v_r[a] w_r[k], where u_1=[1,1], u_2=[1,0], v_1=[1,1,1], v_2=[0,0,0]... no, that doesn't work.
    # Let's just accept that the rank of the 2x3x2 all-ones tensor is 6 and move on. If we can't find it, we'll stick with 44.

    # Given the complexity, let's try a different approach: use the known rank-6 decomposition for 2x3x2 if we can find it.
    # Since I don't have it readily available, I'll try a random search for a rank-6 solution for 2x3x2.
    # But that might be too slow. Let's instead try to reduce the rank of the 44-term solution by finding linear dependencies.

    # For now, return the rank-44 solution.
    return candidate

# EVOLVE-BLOCK-END
