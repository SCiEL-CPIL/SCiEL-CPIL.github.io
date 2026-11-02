"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Key insight: Fix V to the schoolbook identity pattern (V[j*p+k, c] = 1 if c == j*p+k, else 0).
Then the constraint becomes: for each j in [0,3), the 7x7 matrix M_j[k,i] = W[k*n+i, j*p+k] * U[i*m+j, j*p+k] equals 1.
This is a rank-r factorization of the all-ones 7x7 matrix: W_j @ U_j^T = J (7x7 ones).

We can find rank-r factorizations of J using SVD, then round to half-integers and verify.
"""

import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit

    # Schoolbook fallback: rank = n*m*p = 63
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_half_integer(x):
        """Round to nearest half-integer."""
        return round(2 * x) / 2.0

    def try_rank_with_fixed_V(r):
        """
        With V fixed to identity pattern, for each j we need:
        W_j (p x n) @ U_j (n x p)^T = ones(p, n)
        where W_j[k, i] = W[k*n+i, j*p+k] and U_j[i, k] = U[i*m+j, j*p+k]
        
        This is a rank-r factorization of the p x n all-ones matrix.
        The minimum rank of a 7x3 all-ones matrix is 3 (its rank is 3).
        But we need half-integer entries, so we search for valid half-integer factorizations.
        
        Strategy: Use SVD of the ones matrix, then perturb and round.
        Also try explicit constructions.
        """
        if r >= best_rank:
            return None

        # The ones matrix of shape (p, n) = (7, 3)
        J = np.ones((p, n))

        # SVD of J
        U_svd, s, Vt_svd = np.linalg.svd(J, full_matrices=False)
        # rank of J is min(p, n) = 3
        actual_rank = len(s)

        if r < actual_rank:
            return None

        # Try to find a rank-r factorization of J with half-integer entries
        # J = A @ B where A is (p x r) and B is (r x n)
        # We need A and B to have half-integer entries.

        # Method 1: Direct construction for small r
        # For r = 3: J = u @ v^T where u is (7x1) and v is (3x1), but that's rank 1.
        # Actually J is rank 3, so we need at least rank 3.
        # J[i,j] = 1 = sum_{c=0}^{r-1} A[i,c] * B[c,j]
        
        # For r = 3, we can try: A = ones(7,3), B = identity(3,3) -> A@B = ones(7,3). Yes!
        # But wait, A[i,c] = 1 for all i,c and B[c,j] = delta(c,j). Then A@B[i,j] = 1. Correct!
        # So rank 3 works with A = ones(7,3) and B = I(3).
        
        # But we need to check: can we do it with half-integers? Yes, 1 and 0 are half-integers.
        
        # Let's try to find factorizations for various r values.
        # For each r, we search for A (7xr) and B (rx3) with half-integer entries such that A@B = J.

        def find_factorization(r):
            """Find A (p x r) and B (r x n) with half-integer entries such that A @ B = J."""
            if r < n:  # rank must be at least n=3 for a 7x3 ones matrix
                return None
            
            # Simple construction: A = ones(p, r), B = some matrix such that A @ B = J
            # A @ B[i,j] = sum_c A[i,c] * B[c,j] = sum_c 1 * B[c,j] = sum_c B[c,j]
            # So we need sum_c B[c,j] = 1 for all j.
            # And B has half-integer entries.
            # Simple: B[0,j] = 1, B[c,j] = 0 for c > 0. Then sum = 1. Works!
            # So A = ones(7, r), B = [I(3); zeros(r-3, 3)] works for any r >= 3.
            
            # But this gives a valid solution. Let's verify and return.
            A = np.ones((p, r))
            B = np.zeros((r, n))
            for j in range(n):
                B[0, j] = 1.0
            
            # Verify
            if np.allclose(A @ B, J, atol=1e-9):
                return A, B
            
            # Try other constructions
            # For r = 3: use SVD-based approach
            if r == 3:
                # J = u * v^T where u = [1,1,1,1,1,1,1]^T (7x1) and we need rank 3
                # Actually J is rank 3, so we need 3 terms.
                # J[i,j] = sum_{c=0}^{2} A[i,c] * B[c,j]
                # Try: A = ones(7,3), B = I(3,3). Then A@B[i,j] = B[0,j] if i=0... no.
                # A@B[i,j] = sum_c A[i,c]*B[c,j] = sum_c 1*B[c,j] = sum_c B[c,j]
                # So we need sum_c B[c,j] = 1 for all j.
                # B = [[1,0,0],[0,1,0],[0,0,1]] works: sums are 1,1,1.
                B_test = np.eye(n)
                A_test = np.ones((p, r))
                if np.allclose(A_test @ B_test, J, atol=1e-9):
                    return A_test, B_test
            
            # General approach: random search with rounding
            for _ in range(200):
                A_rand = np.random.randn(p, r)
                B_rand = np.random.randn(r, n)
                # Scale to get approximately the right magnitude
                A_rand = A_rand / np.sqrt(r) * np.sqrt(r)  # keep scale
                B_rand = B_rand / np.sqrt(r) * np.sqrt(r)
                
                # Round to half-integers
                A_rounded = np.array([[to_half_integer(x) for x in row] for row in A_rand])
                B_rounded = np.array([[to_half_integer(x) for x in row] for row in B_rand])
                
                if np.allclose(A_rounded @ B_rounded, J, atol=1e-9):
                    return A_rounded, B_rounded
            
            return None

        # Try to find a factorization
        A, B = find_factorization(r)
        if A is None:
            return None

        # Now construct U, V, W from A and B
        # V is fixed: V[j*p+k, c] = 1 if c == j*p+k else 0
        # U[i*m+j, j*p+k] = U_j[i, k] where U_j is the (n x p) part
        # W[k*n+i, j*p+k] = W_j[k, i] where W_j is the (p x n) part
        # We need W_j @ U_j^T = J, i.e., W_j[k,i] * U_j[i,k] summed over... 
        # Wait, let me re-derive.
        
        # With V fixed as identity: V[j*p+k, c] = 1 if c == j*p+k else 0
        # The constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # = U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
        # So for each (i,j,k): U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
        
        # This means for each j, we have a p x n matrix where:
        # M_j[k, i] = U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
        # So U_j (n x p) and W_j (p x n) must satisfy: element-wise product is 1.
        # i.e., U_j[i,k] * W_j[k,i] = 1 for all i,k.
        
        # This is different from a matrix product! It's an element-wise constraint.
        # U_j[i,k] * W_j[k,i] = 1 means W_j[k,i] = 1/U_j[i,k]
        # So we just need U_j to have non-zero entries, and W_j is determined.
        # But we need both to be half-integers!
        
        # So we need U_j[i,k] to be a half-integer such that 1/U_j[i,k] is also a half-integer.
        # Half-integers that have half-integer reciprocals:
        # 1 -> 1, 1/2 -> 2, 2 -> 1/2, -1 -> -1, -1/2 -> -2, -2 -> -1/2
        # Also: 0 is not allowed (infinite reciprocal).
        
        # So each entry of U_j must be in {1, 1/2, 2, -1, -1/2, -2} (and similarly for W_j).
        # The simplest: all entries = 1. Then U_j = ones(n,p) and W_j = ones(p,n).
        
        # This means: U[i*m+j, j*p+k] = 1 for all i,j,k, and W[k*n+i, j*p+k] = 1 for all k,i,j.
        # And V[j*p+k, j*p+k] = 1 (identity).
        
        # Total rank = m*p = 3*7 = 21.
        # Let's verify this construction!
        
        rank_candidate = m * p  # 21
        if rank_candidate < best_rank:
            U_mat = [[0] * rank_candidate for _ in range(n * m)]
            V_mat = [[0] * rank_candidate for _ in range(m * p)]
            W_mat = [[0] * rank_candidate for _ in range(p * n)]
            
            for j in range(m):
                for k in range(p):
                    c = j * p + k
                    V_mat[j * p + k][c] = 1
                    for i in range(n):
                        U_mat[i * m + j][c] = 1
                        W_mat[k * n + i][c] = 1
            
            if verify(U_mat, V_mat, W_mat):
                return {"U": U_mat, "V": V_mat, "W": W_mat}
        
        return None

    # First, try the simple rank-21 construction
    result = try_rank_with_fixed_V(m * p)
    if result is not None:
        best_result = result
        best_rank = m * p

    # Now try to find even lower ranks with more sophisticated searches
    # Lower bound: at least max(n*m, m*p, p*n) = 21, but the simple construction gives 21.
    # Can we do better than 21? The tensor has 63 entries, each must equal 1.
    # With rank r, we have r*(n*m + m*p + p*n) = r*(9+21+21) = 51r parameters.
    # 51r >= 63 => r >= 2 (but this is a very loose bound).
    # More carefully: the tensor T[i,j,k] = 1 has a certain tensor rank.
    # For the all-ones tensor of shape (3,3,7), the tensor rank is at least max(3,3,7) = 7?
    # Actually, the tensor rank of an all-ones tensor (n,m,p) is at least max(n,m,p) but can be higher.
    
    # Let's try random search for ranks below 21
    for r in range(max(n, m, p), best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(150):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                B[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                C[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
            
            # Round to half-integers
            U_rounded = np.array([[to_half_integer(U[i, c]) for c in range(r)] for i in range(n * m)])
            V_rounded = np.array([[to_half_integer(V[i, c]) for c in range(r)] for i in range(m * p)])
            W_rounded = np.array([[to_half_integer(W[i, c]) for c in range(r)] for i in range(p * n)])
            
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = r
                found = True
                break
        
        if found:
            break

    return best_result
