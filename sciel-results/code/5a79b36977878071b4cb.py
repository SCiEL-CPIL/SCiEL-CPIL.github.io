import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit

    def time_left():
        return deadline - (time.time() - start_time)

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, k*p+l, r*n+s] = 1 iff j==k and i==r
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i * m + j, j * p + l, l * n + i] = 1

    def verify(U, V, W, tol=1e-9):
        """Check that sum_r U[:,r] outer V[:,r] outer W[:,r] == T exactly."""
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        r = U_arr.shape[1]
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(r):
            recon += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        # Check half-integer entries
        for M in (U_arr, V_arr, W_arr):
            for v in M.flat:
                if abs(v) > 1e-12 and v != int(v) and 2 * v != int(2 * v):
                    return False
        return True

    def make_rank22_solution():
        """Known rank-22 solution using two Strassen blocks + outer products."""
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A: 00:0, 01:1, 10:3, 11:4
        # B: 00:0, 01:1, 10:4, 11:5
        # C: 00:0, 01:2, 10:1, 11:3
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B: 02:2, 03:3, 12:6, 13:7
        # C: 02:4, 03:6, 12:5, 13:7
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-21 solution by attempting to merge/eliminate one term
    # Strategy: try linear combinations of the 22 terms to see if one becomes redundant
    # This is a linear algebra problem: find if there's a non-trivial linear dependency
    
    def try_rank21():
        """Attempt to find a rank-21 decomposition."""
        base = make_rank22_solution()
        U0 = np.array(base["U"], dtype=np.float64)
        V0 = np.array(base["V"], dtype=np.float64)
        W0 = np.array(base["W"], dtype=np.float64)
        
        # The tensor can be written as sum_{c=0}^{21} U0[:,c] ⊗ V0[:,c] ⊗ W0[:,c]
        # If there's a linear dependency, we can reduce the rank.
        # Check if the 22 vectors in the tensor space are linearly independent.
        
        # Flatten the tensor space: dimension is 6*12*8 = 576
        # Each term is a vector in R^576
        terms = []
        for c in range(22):
            term = np.outer(np.outer(U0[:, c], V0[:, c]), W0[:, c]).flatten()
            terms.append(term)
        
        terms_arr = np.array(terms)  # 22 x 576
        
        # Check rank of the 22x576 matrix
        # If rank < 22, there's a dependency and we can reduce
        try:
            rank = np.linalg.matrix_rank(terms_arr, tol=1e-10)
        except:
            return None
            
        if rank < 22:
            # There's a linear dependency. Find it.
            # Use SVD to find the null space
            U_svd, s, Vt = np.linalg.svd(terms_arr, full_matrices=False)
            # The last row of Vt corresponds to the smallest singular value
            # If s[-1] < tol, we have a dependency
            if s[-1] < 1e-10:
                dep = Vt[-1, :]  # coefficients for the dependency
                # sum_c dep[c] * term_c = 0
                # This means we can eliminate one term
                # Find the term with the largest |dep[c]|
                max_idx = np.argmax(np.abs(dep))
                if abs(dep[max_idx]) > 1e-10:
                    # Eliminate term max_idx by redistributing
                    # New term for c != max_idx: term_c' = term_c - (dep[c]/dep[max_idx]) * term_max_idx
                    # But this changes the U, V, W matrices
                    # Actually, we need to find new U, V, W of rank 21
                    # This is more complex - the dependency in the tensor space
                    # doesn't directly give us a rank-21 U,V,W decomposition
                    
                    # Alternative: try to find a 21-dimensional subspace
                    # that spans the same tensor
                    pass
        
        # Try a different approach: parameter search for rank 21
        # This is computationally expensive, so we'll use a heuristic
        
        # Heuristic: try to combine the two Strassen blocks more efficiently
        # by sharing some of the A-side linear combinations
        
        # The key observation: both Strassen blocks use the same A matrix.
        # The A-side vectors in Strassen are:
        # Block 1: (A00+A11), (A10+A11), A00, A11, (A00+A01), (A10-A00), (A01-A11)
        # Block 2: same A-side vectors
        # So the A-side has at most 7 distinct vectors, used twice.
        
        # If we could find a way to use fewer than 14 A-side vectors total,
        # we might reduce the rank.
        
        # For now, return None to indicate no improvement found
        return None

    # First, verify the rank-22 solution works
    base_sol = make_rank22_solution()
    if not verify(base_sol["U"], base_sol["V"], base_sol["W"]):
        # Fallback: construct a trivial rank-24 solution
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    c = i * m * p + j * p + l
                    U[i * m + j][c] = 1
                    V[j * p + l][c] = 1
                    W[l * n + i][c] = 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-21 solution
    if time_left() > 30:
        result_21 = try_rank21()
        if result_21 is not None:
            if verify(result_21["U"], result_21["V"], result_21["W"]):
                return result_21

    # Return the verified rank-22 solution
    return base_sol
