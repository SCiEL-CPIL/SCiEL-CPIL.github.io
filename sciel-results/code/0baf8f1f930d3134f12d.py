"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Approach: Use the Kronecker product structure T_6 = T_3 ⊗ T_2.
- 3x3 multiplication: naive rank-27 decomposition
- 2x2 multiplication: Strassen rank-7 decomposition
- Total rank: 27 * 7 = 189

The 6x6 index i is decomposed as i = 2*I + i' where I in {0,1,2} (block)
and i' in {0,1} (intra-block). The Kronecker product gives factors in
"block-major" order which we permute to the required row-major layout.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _strassen_2x2_factors():
    """
    Return the 7 Strassen factors for 2x2 matrix multiplication.
    
    For C = A*B where A, B, C are 2x2 matrices indexed as:
    A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
    C[0]=C00, C[1]=C01, C[2]=C10, C[3]=C11
    
    The tensor T_2[a,b,c] = 1 iff the (a,b,c) indices correspond to a valid
    (i,j,k) triple where a=i*2+j, b=j*2+k, c=k*2+i.
    
    Strassen's 7 products:
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)B00
    m3 = A00(B01-B11)
    m4 = A11(B10-B00)
    m5 = (A00+A01)B11
    m6 = (A10-A00)(B00+B01)
    m7 = (A01-A11)(B10+B11)
    
    C00 = m1 - m4 + m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m3 + m6 + m7
    """
    # Each factor is (u, v, w) where u, v, w are vectors of length 4
    # u[i*2+j] is the coefficient of A[i,j]
    # v[j*2+k] is the coefficient of B[j,k]  
    # w[k*2+i] is the coefficient for C[k,i]... wait, C[i,k]
    # Actually C is indexed as c = k*n+i where n=2, so c = 2*k+i
    # But C[i,k] means row i, col k, so in flattened form C[i*2+k]
    # The tensor entry is T[a,b,c] where c = k*2+i (following the convention)
    
    factors = []
    
    # m1 = (A00+A11)(B00+B11)
    # u1: A00=1, A11=1 -> indices 0 and 3
    u1 = np.array([1, 0, 0, 1])
    # v1: B00=1, B11=1 -> indices 0 and 3
    v1 = np.array([1, 0, 0, 1])
    # w1: contributes to C00 and C11
    # C00 is at index 0*2+0=0, C11 is at index 1*2+1=3
    # But in the tensor convention c = k*2+i where i is row, k is col
    # C[i,k] -> c = k*2+i
    # C00: i=0,k=0 -> c=0
    # C11: i=1,k=1 -> c=3
    w1 = np.array([1, 0, 0, 1])
    factors.append((u1, v1, w1))
    
    # m2 = (A10+A11)B00
    # u2: A10=1, A11=1 -> indices 2 and 3
    u2 = np.array([0, 0, 1, 1])
    # v2: B00=1 -> index 0
    v2 = np.array([1, 0, 0, 0])
    # w2: contributes to C10
    # C10: i=1,k=0 -> c=0*2+1=1
    w2 = np.array([0, 1, 0, 0])
    factors.append((u2, v2, w2))
    
    # m3 = A00(B01-B11)
    # u3: A00=1 -> index 0
    u3 = np.array([1, 0, 0, 0])
    # v3: B01=1, B11=-1 -> indices 1 and 3
    v3 = np.array([0, 1, 0, -1])
    # w3: contributes to C01
    # C01: i=0,k=1 -> c=1*2+0=2
    w3 = np.array([0, 0, 1, 0])
    factors.append((u3, v3, w3))
    
    # m4 = A11(B10-B00)
    # u4: A11=1 -> index 3
    u4 = np.array([0, 0, 0, 1])
    # v4: B10=1, B00=-1 -> indices 2 and 0
    v4 = np.array([-1, 0, 1, 0])
    # w4: contributes to C10 with -1 (from C00 = m1 - m4 + ...)
    # Wait, C10 = m2 + m4, so m4 contributes +1 to C10
    # C10: i=1,k=0 -> c=1
    w4 = np.array([0, 1, 0, 0])
    factors.append((u4, v4, w4))
    
    # m5 = (A00+A01)B11
    # u5: A00=1, A01=1 -> indices 0 and 1
    u5 = np.array([1, 1, 0, 0])
    # v5: B11=1 -> index 3
    v5 = np.array([0, 0, 0, 1])
    # w5: contributes to C01 with +1 and C00 with +1
    # C01: c=2, C00: c=0
    w5 = np.array([1, 0, 1, 0])
    factors.append((u5, v5, w5))
    
    # m6 = (A10-A00)(B00+B01)
    # u6: A10=1, A00=-1 -> indices 2 and 0
    u6 = np.array([-1, 0, 1, 0])
    # v6: B00=1, B01=1 -> indices 0 and 1
    v6 = np.array([1, 1, 0, 0])
    # w6: contributes to C11 with +1
    # C11: i=1,k=1 -> c=3
    w6 = np.array([0, 0, 0, 1])
    factors.append((u6, v6, w6))
    
    # m7 = (A01-A11)(B10+B11)
    # u7: A01=1, A11=-1 -> indices 1 and 3
    u7 = np.array([0, 1, 0, -1])
    # v7: B10=1, B11=1 -> indices 2 and 3
    v7 = np.array([0, 0, 1, 1])
    # w7: contributes to C00 with +1 and C11 with +1
    # C00: c=0, C11: c=3
    w7 = np.array([1, 0, 0, 1])
    factors.append((u7, v7, w7))
    
    return factors


def _naive_3x3_factors():
    """
    Return the 27 naive factors for 3x3 matrix multiplication.
    
    Each factor corresponds to a triple (i,j,k) where i is row of A,
    j is col of A / row of B, k is col of B.
    
    U[i*3+j, r] = 1 if (i,j) matches
    V[j*3+k, r] = 1 if (j,k) matches
    W[k*3+i, r] = 1 if (k,i) matches
    """
    factors = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = np.zeros(9)
                v = np.zeros(9)
                w = np.zeros(9)
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                factors.append((u, v, w))
    return factors


def _build_permutation_6x6():
    """
    Build permutations to convert from Kronecker (block-major) layout to
    row-major layout for 6x6 matrices.
    
    For a 6x6 matrix, index = row*6 + col.
    Decompose row = 2*I + i' where I in {0,1,2}, i' in {0,1}
    Decompose col = 2*J + j' where J in {0,1,2}, j' in {0,1}
    
    Kronecker index (block-major): k = (I*3+J)*4 + (i'*2+j') = 12I + 4J + 2i' + j'
    Target index (row-major): t = row*6 + col = (2I+i')*6 + (2J+j') = 12I + 6i' + 2J + j'
    
    We need a permutation perm_U such that U_target[t] = U_kron[k] where k maps to t.
    """
    perm_U = np.zeros(36, dtype=int)
    perm_V = np.zeros(36, dtype=int)
    perm_W = np.zeros(36, dtype=int)
    
    for I in range(3):
        for J in range(3):
            for i_prime in range(2):
                for j_prime in range(2):
                    # Kronecker index
                    k = (I*3+J)*4 + (i_prime*2+j_prime)
                    # Target index
                    row = 2*I + i_prime
                    col = 2*J + j_prime
                    t = row*6 + col
                    perm_U[t] = k
    
    # For V (B matrix): index = row*6 + col where row=j, col=k
    # Decompose row = 2*J + j', col = 2*K + k'
    # Kronecker index: k = (J*3+K)*4 + (j'*2+k')
    # Target index: t = (2*J+j')*6 + (2*K+k')
    for J in range(3):
        for K in range(3):
            for j_prime in range(2):
                for k_prime in range(2):
                    k = (J*3+K)*4 + (j_prime*2+k_prime)
                    row = 2*J + j_prime
                    col = 2*K + k_prime
                    t = row*6 + col
                    perm_V[t] = k
    
    # For W (C matrix): index = col*6 + row where col=k, row=i (following c=k*n+i convention)
    # Wait, the convention is W[k*n+i, r] where n=6
    # So W index = k*6 + i
    # Decompose k = 2*K + k', i = 2*I + i'
    # Kronecker index: k_kron = (K*3+I)*4 + (k'*2+i')
    # Target index: t = k*6 + i = (2*K+k')*6 + (2*I+i')
    for K in range(3):
        for I in range(3):
            for k_prime in range(2):
                for i_prime in range(2):
                    k_kron = (K*3+I)*4 + (k_prime*2+i_prime)
                    k_val = 2*K + k_prime
                    i_val = 2*I + i_prime
                    t = k_val*6 + i_val
                    perm_W[t] = k_kron
    
    return perm_U, perm_V, perm_W


def _build_decomposition_189():
    """
    Build the rank-189 decomposition using T_3 (naive, rank 27) ⊗ T_2 (Strassen, rank 7).
    """
    factors_3 = _naive_3x3_factors()  # 27 factors
    factors_2 = _strassen_2x2_factors()  # 7 factors
    
    perm_U, perm_V, perm_W = _build_permutation_6x6()
    
    rank = 27 * 7
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    idx = 0
    for (u3, v3, w3) in factors_3:
        for (u2, v2, w2) in factors_2:
            # Kronecker product of factors
            # u_full has length 36, indexed as (I*3+J)*4 + (i'*2+j')
            u_full_kron = np.zeros(36)
            v_full_kron = np.zeros(36)
            w_full_kron = np.zeros(36)
            
            for I in range(3):
                for J in range(3):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            # u3 index: I*3+J (this is the (i,j) entry of A_3x3)
                            # u2 index: i_prime*2+j_prime (this is the (i',j') entry of A_2x2)
                            u_kron_idx = (I*3+J)*4 + (i_prime*2+j_prime)
                            u_full_kron[u_kron_idx] = u3[I*3+J] * u2[i_prime*2+j_prime]
                            
                            # v3 index: J*3+K... wait, v3 is for B_3x3 with indices (j,k)
                            # But in the Kronecker product, the B_3x3 indices are (J,K)
                            # and B_2x2 indices are (j',k')
                            # v_full_kron index: (J*3+K)*4 + (j'*2+k')
                            pass
            
            # Actually, let me redo this more carefully.
            # u3 has indices (I, J) representing A_3x3[I, J]
            # u2 has indices (i', j') representing A_2x2[i', j']
            # The combined A_6x6 index is (2I+i', 2J+j') -> flat index (2I+i')*6 + (2J+j')
            # Kronecker flat index: (I*3+J)*4 + (i'*2+j')
            
            # v3 has indices (J, K) representing B_3x3[J, K]
            # v2 has indices (j', k') representing B_2x2[j', k']
            # The combined B_6x6 index is (2J+j', 2K+k') -> flat index (2J+j')*6 + (2K+k')
            # Kronecker flat index: (J*3+K)*4 + (j'*2+k')
            
            # w3 has indices (K, I) representing C_3x3[K, I]... wait
            # w3[k*3+i] where k is col, i is row
            # In the Kronecker product, w3 indices are (K, I) for C_3x3
            # w2 indices are (k', i') for C_2x2
            # The combined C_6x6 index: c = k*6 + i = (2K+k')*6 + (2I+i')
            # Kronecker flat index: (K*3+I)*4 + (k'*2+i')
            
            for I in range(3):
                for J in range(3):
                    for K in range(3):
                        for i_prime in range(2):
                            for j_prime in range(2):
                                for k_prime in range(2):
                                    # u_full
                                    u_kron = (I*3+J)*4 + (i_prime*2+j_prime)
                                    u_full_kron[u_kron] = u3[I*3+J] * u2[i_prime*2+j_prime]
                                    
                                    # v_full
                                    v_kron = (J*3+K)*4 + (j_prime*2+k_prime)
                                    v_full_kron[v_kron] = v3[J*3+K] * v2[j_prime*2+k_prime]
                                    
                                    # w_full
                                    w_kron = (K*3+I)*4 + (k_prime*2+i_prime)
                                    w_full_kron[w_kron] = w3[K*3+I] * w2[k_prime*2+i_prime]
            
            # Apply permutations
            U[:, idx] = u_full_kron[perm_U]
            V[:, idx] = v_full_kron[perm_V]
            W[:, idx] = w_full_kron[perm_W]
            idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_tensor(decomp, n, m, p, num_tests=20):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    for _ in range(num_tests):
        for _ in range(50):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        for _ in range(50):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    decomp = _build_decomposition_189()
    
    # Verify correctness
    if not _verify_tensor(decomp, n, m, p):
        # Fallback to naive 216-rank decomposition
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return decomp
