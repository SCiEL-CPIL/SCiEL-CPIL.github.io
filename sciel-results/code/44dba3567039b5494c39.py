"""
Strategy: Algebraic Structural Search via Block Decomposition.

Instead of random search over the full coefficient space, we exploit the
block structure of matrix multiplication. A 2x3 * 3x7 multiplication can be
viewed as a sum of smaller multiplications if we partition the indices.
Specifically, we look for decompositions of the form:
  C = A_1 B_1 + A_2 B_2 + ...
where each term is a valid matrix multiplication of smaller dimensions.
For example, if we partition the 3 intermediate dimension into 1 and 2,
we get (2x1)*(1x7) and (2x2)*(2x7).
The rank of the total decomposition is the sum of the ranks of the parts.
We know the schoolbook rank for small matrices:
  R(1,1,1) = 1
  R(2,2,2) = 7 (Strassen)
  R(1,1,7) = 7
  R(2,2,7) = ? (likely 14 or 15, schoolbook is 14)
  
We will systematically check all possible partitions of the dimensions
(n, m, p) = (2, 3, 7) and use known minimal ranks for the sub-problems
to see if we can beat the schoolbook rank of 42.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Base case: Build a decomposition for a given (n, m, p) using schoolbook
    def build_sub(n_sub, m_sub, p_sub):
        rank = n_sub * m_sub * p_sub
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        c = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j, c] = 1.0
                    V[j * p_sub + k, c] = 1.0
                    W[k * n_sub + i, c] = 1.0
                    c += 1
        return U, V, W

    # Combine two decompositions into one larger decomposition
    # U1, V1, W1 are for (n1, m1, p1)
    # U2, V2, W2 are for (n2, m2, p2)
    # We assume the indices are aligned such that the first m1 of the m dimension
    # are used by part 1, and the remaining m2 by part 2.
    # This corresponds to block diagonal matrices.
    def combine(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n_total, m_total, p_total):
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n_total * m_total, rank))
        V = np.zeros((m_total * p_total, rank))
        W = np.zeros((p_total * n_total, rank))
        
        # Place part 1
        # U1 rows: i in [0, n1), j in [0, m1) -> index i*m_total + j
        # V1 rows: j in [0, m1), k in [0, p1) -> index j*p_total + k
        # W1 rows: k in [0, p1), i in [0, n1) -> index k*n_total + i
        for i in range(n1):
            for j in range(m1):
                for c in range(U1.shape[1]):
                    U[i * m_total + j, c] = U1[i * m1 + j, c]
        
        for j in range(m1):
            for k in range(p1):
                for c in range(V1.shape[1]):
                    V[j * p_total + k, c] = V1[j * p1 + k, c]
        
        for k in range(p1):
            for i in range(n1):
                for c in range(W1.shape[1]):
                    W[k * n_total + i, c] = W1[k * n1 + i, c]
                    
        # Place part 2
        # Offset columns by U1.shape[1]
        # U2 rows: i in [0, n2), j in [0, m2)
        # In the total matrix, these correspond to i in [0, n_total) (since n1=n_total) 
        # and j in [m1, m1+m2)
        # Wait, the block diagonal assumption usually implies disjoint index sets.
        # If we split M into M1 and M2, then A is (N x M1) and (N x M2)? No.
        # A is (N x M). If we split M into M1+M2, A = [A1, A2].
        # B is (M x P). B = [[B11, B12], [B21, B22]]? No, B is just (M x P).
        # The product C = A B.
        # C = A1 B1 + A2 B2 if B is block diagonal? No.
        # If B = [[B1, 0], [0, B2]] and A = [A1, A2], then C = A1 B1 + A2 B2.
        # This requires P to be split as well? No, B1 is (M1 x P1) and B2 is (M2 x P2)?
        # If P is not split, B1 is (M1 x P) and B2 is (M2 x P).
        # Then C = A1 B1 + A2 B2.
        # This works if we define B1 as the top M1 rows of B and B2 as the bottom M2 rows.
        # And A1 as the left N x M1 block of A, A2 as the right N x M2 block.
        # This is always a valid decomposition of the multiplication tensor!
        # The rank is R(N, M1, P) + R(N, M2, P).
        
        # So we can recursively split M.
        # Similarly, we can split N or P?
        # If we split N: A = [[A1], [A2]], B = B. C = [[C1], [C2]].
        # C1 = A1 B, C2 = A2 B.
        # Rank = R(N1, M, P) + R(N2, M, P).
        
        # If we split P: A = A, B = [B1, B2]. C = [C1, C2].
        # C1 = A B1, C2 = A B2.
        # Rank = R(N, M, P1) + R(N, M, P2).
        
        # So the problem is separable!
        # R(N, M, P) = min over partitions of N, M, P of sum of R(parts).
        # Base case: R(1, 1, 1) = 1.
        # We can use dynamic programming to find the minimal rank assuming
        # separability.
        
        # Let's implement the combination for the specific split we found.
        # For now, this function is a placeholder for a specific split.
        pass

    # Dynamic Programming for Separable Rank
    # R[n, m, p] = min ( R[n1, m, p] + R[n2, m, p] for n1+n2=n,
    #                    R[n, m1, p] + R[n, m2, p] for m1+m2=m,
    #                    R[n, m, p1] + R[n, m, p2] for p1+p2=p,
    #                    n*m*p )
    
    # We only need R for dimensions up to (2, 3, 7).
    # Let's compute R for all (n, m, p) up to these limits.
    
    def compute_min_rank(n_max, m_max, p_max):
        R = {}
        for n in range(1, n_max + 1):
            for m in range(1, m_max + 1):
                for p in range(1, p_max + 1):
                    # Base case: schoolbook
                    best = n * m * p
                    # Split N
                    for n1 in range(1, n):
                        n2 = n - n1
                        if (n1, m, p) in R and (n2, m, p) in R:
                            val = R[(n1, m, p)] + R[(n2, m, p)]
                            if val < best:
                                best = val
                    # Split M
                    for m1 in range(1, m):
                        m2 = m - m1
                        if (n, m1, p) in R and (n, m2, p) in R:
                            val = R[(n, m1, p)] + R[(n, m2, p)]
                            if val < best:
                                best = val
                    # Split P
                    for p1 in range(1, p):
                        p2 = p - p1
                        if (n, m, p1) in R and (n, m, p2) in R:
                            val = R[(n, m, p1)] + R[(n, m, p2)]
                            if val < best:
                                best = val
                    R[(n, m, p)] = best
        return R

    # Compute minimal separable ranks
    R = compute_min_rank(2, 3, 7)
    target_rank = R[(2, 3, 7)]
    
    # If the separable rank is better than schoolbook, we need to construct the decomposition.
    # We can do this recursively.
    def construct_decomp(n, m, p):
        # Returns U, V, W for (n, m, p)
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            return U, V, W
            
        best_rank = R[(n, m, p)]
        
        # Try splits
        # Split N
        for n1 in range(1, n):
            n2 = n - n1
            if R[(n1, m, p)] + R[(n2, m, p)] == best_rank:
                U1, V1, W1 = construct_decomp(n1, m, p)
                U2, V2, W2 = construct_decomp(n2, m, p)
                # Combine
                rank = best_rank
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Part 1: Rows 0..n1-1 of A
                # U1 is (n1*m, r1). Map to U[i*m+j, c] for i in [0, n1)
                for i in range(n1):
                    for j in range(m):
                        U[i * m + j, :U1.shape[1]] = U1[i * m + j, :]
                
                # V is the same for both parts because B is shared?
                # Wait, if we split N, A = [[A1], [A2]]. B is the same.
                # C1 = A1 B, C2 = A2 B.
                # V is the same matrix B in both cases.
                # But in the tensor decomposition, V is a factor matrix.
                # T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # If we split N, the index i is in U and W.
                # For part 1 (i < n1), we use U1, W1. V is shared?
                # No, the decomposition is a sum of terms.
                # T = T1 + T2.
                # T1[i, j, k] = sum_{c1} U1[i,j,c1] V1[j,k,c1] W1[k,i,c1]
                # T2[i, j, k] = sum_{c2} U2[i,j,c2] V2[j,k,c2] W2[k,i,c2]
                # If we use the same V for both, we can just concatenate U, V, W columns.
                # But V must be the same for the "same" B?
                # Actually, if we split N, the multiplication is:
                # C1 = A1 B and C2 = A2 B.
                # The tensor for C1 is (n1, m, p). The tensor for C2 is (n2, m, p).
                # The V factor for C1 is B. The V factor for C2 is B.
                # So we can reuse the same V columns?
                # No, the rank adds up. We need separate V columns if we want to be safe,
                # unless we can prove they are identical and can be merged.
                # If V1 == V2, we can merge the columns by summing U*W?
                # sum U1 V W1 + sum U2 V W2 = sum (U1 W1 + U2 W2) V? No, V is in the middle.
                # It's (U1 V W1) + (U2 V W2). This is not a simple concatenation if V is shared.
                # Actually, we can just duplicate V. The rank will be R1 + R2.
                
                # Let's just duplicate V to be safe and simple.
                # U: concat U1, U2 (padded with 0s for the other's rows)
                # V: concat V1, V2 (padded with 0s? No, V is (m, p). It's the same size.
                #     We just put V1 in first R1 cols, V2 in next R2 cols.
                # W: concat W1, W2 (padded with 0s for the other's rows)
                
                for i in range(n1):
                    for j in range(m):
                        for c in range(U1.shape[1]):
                            U[i * m + j, c] = U1[i * m + j, c]
                            # V is shared? No, we duplicate.
                            # V[j*p+k, c] = V1[j*p+k, c]
                            # W[k*n+i, c] = W1[k*n1+i, c] -- Wait, W index i is local to n1.
                            # In the combined W, the row index is k*n + i.
                            # For part 1, i is in [0, n1).
                            # For part 2, i is in [n1, n).
                            
                for j in range(m):
                    for k in range(p):
                        for c in range(V1.shape[1]):
                            V[j * p + k, c] = V1[j * p + k, c]
                            
                for k in range(p):
                    for i in range(n1):
                        for c in range(W1.shape[1]):
                            W[k * n + i, c] = W1[k * n1 + i, c]
                            
                # Part 2
                for i in range(n2):
                    for j in range(m):
                        for c in range(U2.shape[1]):
                            U[i * m + j, U1.shape[1] + c] = U2[i * m + j, c]
                            
                for j in range(m):
                    for k in range(p):
                        for c in range(V2.shape[1]):
                            V[j * p + k, U1.shape[1] + c] = V2[j * p + k, c]
                            
                for k in range(p):
                    for i in range(n2):
                        for c in range(W2.shape[1]):
                            W[k * n + (i + n1), U1.shape[1] + c] = W2[k * n2 + i, c]
                            
                return U, V, W
                
        # Split M
        for m1 in range(1, m):
            m2 = m - m1
            if R[(n, m1, p)] + R[(n, m2, p)] == best_rank:
                U1, V1, W1 = construct_decomp(n, m1, p)
                U2, V2, W2 = construct_decomp(n, m2, p)
                rank = best_rank
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Part 1: j in [0, m1)
                for i in range(n):
                    for j in range(m1):
                        for c in range(U1.shape[1]):
                            U[i * m + j, c] = U1[i * m1 + j, c]
                            
                for j in range(m1):
                    for k in range(p):
                        for c in range(V1.shape[1]):
                            V[j * p + k, c] = V1[j * p1 + k, c] # Wait, V1 is (m1, p).
                            # In V1, row index is j*p + k? No, V1 is (m1*p, r1).
                            # Row j in V1 corresponds to j*p + k in V?
                            # Yes, if we align k.
                            V[j * p + k, c] = V1[j * p + k, c]
                            
                for k in range(p):
                    for i in range(n):
                        for c in range(W1.shape[1]):
                            W[k * n + i, c] = W1[k * n + i, c]
                            
                # Part 2: j in [m1, m)
                for i in range(n):
                    for j in range(m2):
                        for c in range(U2.shape[1]):
                            U[i * m + (j + m1), U1.shape[1] + c] = U2[i * m2 + j, c]
                            
                for j in range(m2):
                    for k in range(p):
                        for c in range(V2.shape[1]):
                            V[(j + m1) * p + k, U1.shape[1] + c] = V2[j * p + k, c]
                            
                for k in range(p):
                    for i in range(n):
                        for c in range(W2.shape[1]):
                            W[k * n + i, U1.shape[1] + c] = W2[k * n + i, c]
                            
                return U, V, W
                
        # Split P
        for p1 in range(1, p):
            p2 = p - p1
            if R[(n, m, p1)] + R[(n, m, p2)] == best_rank:
                U1, V1, W1 = construct_decomp(n, m, p1)
                U2, V2, W2 = construct_decomp(n, m, p2)
                rank = best_rank
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Part 1: k in [0, p1)
                for i in range(n):
                    for j in range(m):
                        for c in range(U1.shape[1]):
                            U[i * m + j, c] = U1[i * m + j, c]
                            
                for j in range(m):
                    for k in range(p1):
                        for c in range(V1.shape[1]):
                            V[j * p + k, c] = V1[j * p1 + k, c]
                            
                for k in range(p1):
                    for i in range(n):
                        for c in range(W1.shape[1]):
                            W[k * n + i, c] = W1[k * n + i, c]
                            
                # Part 2: k in [p1, p)
                for i in range(n):
                    for j in range(m):
                        for c in range(U2.shape[1]):
                            U[i * m + j, U1.shape[1] + c] = U2[i * m + j, c]
                            
                for j in range(m):
                    for k in range(p2):
                        for c in range(V2.shape[1]):
                            V[j * p + (k + p1), U1.shape[1] + c] = V2[j * p2 + k, c]
                            
                for k in range(p2):
                    for i in range(n):
                        for c in range(W2.shape[1]):
                            W[(k + p1) * n + i, U1.shape[1] + c] = W2[k * n + i, c]
                            
                return U, V, W
                
        # Fallback: Schoolbook
        return build_sub(n, m, p)

    # Attempt to construct the optimal separable decomposition
    try:
        U, V, W = construct_decomp(n, m, p)
        if verify(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception as e:
        pass

    # Fallback to schoolbook
    U, V, W = build_schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
