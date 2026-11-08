import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the full multiplication tensor T[i,j,k,l] = 1 if k==j else 0
    # where i in [0,n), j in [0,m), k in [0,p), l in [0,m)
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # T_{i,j,k}^{l} = delta_{j,l} for C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] is a vector of length m with 1 at position j... no.
    # The tensor T has indices (i, j, k) where i is row of A, j is column of A / row of B, k is column of B.
    # T[i,j,k] = 1 if this term contributes to C[i,k]
    # Actually the standard tensor T in {0,1}^{n x m x p} where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # But wait, the decomposition is: C[i,k] = sum_r U[i,r] * V[k,r] * W[j,r]... no.
    # The tensor T_{i,j,k} = delta_{j, j'} where j' is the shared index.
    # For matrix multiplication: T[i,j,k] = 1 for all valid i,j,k where the shared index is j.
    # More precisely, T is a 3-tensor with T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] is part of the sum for C[i,k].
    # This means T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right. The tensor entries are:
    # T[i,j,k] = 1 if j is the shared index between A[i,j] and B[j,k].
    # So T[i,j,k] = 1 for all i,j,k in their respective ranges.
    # No! The tensor is defined such that:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] where the "1" appears when the indices match the multiplication pattern.
    # Specifically, T[i,j,k] = 1 for all valid (i,j,k) - that would give rank n*m*p which is schoolbook.
    # 
    # Let me reconsider. The tensor T for matrix multiplication of (n x m) by (m x p) matrices is:
    # T[i,j,k] = 1 if and only if the product term A[i,j] * B[j,k] contributes to C[i,k]
    # This is always true for valid indices, so T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # But that's a rank-1 tensor which is wrong.
    #
    # Actually, the correct formulation: The tensor T is in R^{(n*p) x m x m} or similar.
    # Let me use the standard formulation:
    # The matrix multiplication tensor T has entries T[i, j, k] where:
    # - i indexes the row of C (and A): 0 <= i < n
    # - j indexes the column of A (and row of B): 0 <= j < m  
    # - k indexes the column of B (and C): 0 <= k < p
    # T[i,j,k] = 1 (it's always 1 for valid indices)
    # But the tensor decomposition requires: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Wait, that's not quite right either. Let me look at the problem statement again.
    #
    # "output factor follows the official trace convention W[k*n+i,r]"
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # So the indices are:
    # U[i*m + j, r] for i in [0,n), j in [0,m)
    # V[j*p + k, r] for j in [0,m), k in [0,p)
    # W[k*n + i, r] for k in [0,p), i in [0,n)
    #
    # The decomposition: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[i,j,k]
    # where T[i,j,k] = 1 if j == j' and i == i' and k == k'... 
    #
    # Actually, the standard tensor decomposition for matrix multiplication:
    # T[i, j, k] = delta_{j, j'} (the shared index must match)
    # The tensor is: T is a 3-tensor with T[i,j,k] = 1 for all valid i,j,k
    # But we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, I think the tensor entries are:
    # For each triple (i, j, k) where i in [0,n), j in [0,m), k in [0,p):
    # T[i,j,k] = 1
    # And we want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Hmm, but that doesn't seem right because U, V, W all have the same "r" index.
    # Let me reconsider.
    #
    # The correct formulation: The matrix multiplication tensor T ∈ R^{(nm) × (mp) × (pn)}
    # where T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i', else 0.
    # No, that would make it a permutation-like tensor.
    #
    # Actually, I think the correct formulation is:
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # But that gives a tensor that's just the identity in each mode, which has rank nm (or similar).
    #
    # Let me think differently. The matrix multiplication can be written as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be seen as a bilinear (actually trilinear) form.
    # The tensor T has entries indexed by (i,j) for A, (j,k) for B, and (i,k) for C.
    # T[(i,j), (j',k), (i',k')] = 1 if j=j', i=i', k=k' else 0.
    #
    # So T is a 3-tensor where T[a, b, c] = 1 if the indices match appropriately.
    # Specifically, if we flatten:
    # A index: a = i*m + j (0 <= a < n*m)
    # B index: b = j'*p + k (0 <= b < m*p)
    # C index: c = k'*n + i' (0 <= c < p*n)
    #
    # T[a, b, c] = 1 if j = j' and i = i' and k = k', else 0.
    #
    # So T[a, b, c] = 1 iff (a % m) == (b // p) and (a // m) == (c % n) and (b % p) == (c // n)
    #
    # We want to find rank r and matrices U (nm x r), V (mp x r), W (pn x r) such that:
    # T[a, b, c] = sum_{r_idx=0}^{r-1} U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
    
    # Build the tensor
    nm = n * m
    mp = m * p
    pn = p * n
    
    # T is a 3-tensor of size nm x mp x pn
    # T[a, b, c] = 1 if (a % m) == (b // p) and (a // m) == (c % n) and (b % p) == (c // n)
    
    # For efficiency, let's build it as a list of non-zero entries
    # Each non-zero entry is (a, b, c) where the conditions hold
    nonzeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzeros.append((a, b, c))
    
    # Now we need to find the minimum rank decomposition.
    # This is a hard problem in general, but for this specific case we can try:
    # 1. Use the current hardcoded solution (rank 39)
    # 2. Try to find a better solution using linear algebra
    
    # The current solution has rank 39. Let's verify it and see if we can do better.
    
    # First, let's construct the current solution's U, V, W
    U_curr = [[0] * 39 for _ in range(nm)]
    V_curr = [[0] * 39 for _ in range(mp)]
    W_curr = [[0] * 39 for _ in range(pn)]
    
    col = 0
    def add_term(u_indices, v_indices, w_indices, sign_w=None, sign_v=None, sign_u=None):
        nonlocal col
        for idx in u_indices:
            U_curr[idx][col] += 1
        for idx in v_indices:
            V_curr[idx][col] += 1
        for idx in w_indices:
            W_curr[idx][col] += 1
        if sign_w:
            for idx, s in sign_w:
                W_curr[idx][col] = s
        if sign_v:
            for idx, s in sign_v:
                V_curr[idx][col] = s
        if sign_u:
            for idx, s in sign_u:
                U_curr[idx][col] = s
        col += 1
    
    # Strassen blocks for k=0,1; 2,3; 4,5
    for k_start in [0, 2, 4]:
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11], sign_w=[(c11, -1)])
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11], sign_v=[(b11, -1)])
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10], sign_v=[(b00, -1)])
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01], sign_w=[(c00, -1)])
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11], sign_u=[(0, -1)])
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00], sign_u=[(4, -1)])
    
    # Block for k=6
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part 2: j=2
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
    
    # Verify the current solution
    def verify_solution(U, V, W, rank):
        for (a, b, c) in nonzeros:
            s = 0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 1:
                return False
        # Check that non-nonzero entries are 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in set(nonzeros):
                        s = 0
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
        return True
    
    # The verification above is O(nm * mp * pn * rank) which is 42*21*14*39 ≈ 386,000 operations
    # That's fast enough.
    
    # Let's try to find a better solution using a greedy approach or linear algebra.
    # One approach: try to reduce the rank by combining terms.
    
    # For now, let's return the current solution if it's valid.
    if verify_solution(U_curr, V_curr, W_curr, 39):
        return {"U": U_curr, "V": V_curr, "W": W_curr}
    
    # If the current solution is invalid, fall back to schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(nm)]
    V_sb = [[0] * rank_schoolbook for _ in range(mp)]
    W_sb = [[0] * rank_schoolbook for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
