import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Index mappings
    def idxA(i, j):
        return i * m + j
    def idxB(j, k):
        return j * p + k
    def idxC(k, i):
        return k * n + i

    # Build the multiplication tensor T[a,b,c] = 1 if c = A[a]*B[b] contributes
    # Actually T is the structure tensor: T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # More precisely: for each j (inner dimension), for each i in [0,n), k in [0,p):
    #   A[i,j] * B[j,k] -> C[k,i]
    # So the tensor has entries T[idxA(i,j), idxB(j,k), idxC(k,i)] = 1

    # We want to find a rank-r decomposition.
    # Approach: use a greedy/random search for rank-r tensors with small integer entries.
    # Since the problem is small, we can try a structured approach.

    # First, let's compute the trivial schoolbook rank:
    # rank_schoolbook = n * m * p = 2 * 3 * 7 = 42

    # Try to find a lower rank decomposition.
    # We'll use a randomized approach: generate random factor matrices and check if they work.
    # But checking exactness is hard. Instead, let's use a structured approach based on
    # the algebraic structure.

    # The tensor is T[i,j,k] = sum over the "path" i->j->k
    # This is a 3-tensor with a specific structure.

    # Let's try a different approach: use the fact that the tensor can be written as
    # sum_r U_a,r * V_b,r * W_c,r for all (a,b,c)
    # where a = idxA(i,j), b = idxB(j,k), c = idxC(k,i)

    # The constraint is:
    # sum_r U[idxA(i,j),r] * V[idxB(j,k),r] * W[idxC(k,i),r] = 1  (for all valid i,j,k)
    # and 0 for all other (a,b,c)

    # Let's try rank = 39 (which is what the current program achieves) and see if we can do better.
    # Actually, let's try to find the minimal rank by trying ranks from 38 down.

    # For each rank r, we can try to solve the system of polynomial equations.
    # This is hard in general, but for small r we can try.

    # Let's try a different approach: use the known bounds.
    # For 2x3x7, the trivial bound is 42.
    # Strassen-like improvements might reduce this.

    # Let's try to construct a decomposition with rank 38 or less.

    # One approach: use the fact that the tensor is a "matrix multiplication tensor"
    # and use known results from algebraic complexity theory.

    # For 2x2x2, Strassen gives rank 7 (vs 8 schoolbook).
    # For 2x3x7, we can try to partition and apply Strassen where possible.

    # Partition the 3x7 matrix into:
    # - First 2 columns: 2x2x2 blocks (3 of them if we pair columns)
    # - Remaining column: schoolbook

    # Actually, let's just try to find a good decomposition by searching.

    # Let's use a randomized search with local improvement.

    import random
    import time

    best_rank = nm * mp  # 2 * 3 * 7 = 42 (wait, that's not right)
    # Actually, schoolbook rank for 2x3 * 3x7 is 2*3*7 = 42? No.
    # Schoolbook: for each i in [0,2), j in [0,3), k in [0,7):
    #   C[k,i] += A[i,j] * B[j,k]
    # So there are 2*3*7 = 42 terms, each contributing to one C entry.
    # So rank_schoolbook = 42.

    # Wait, that's the number of multiplications, which equals the rank of the schoolbook decomposition.

    # Let's try to find a lower rank.

    # We'll do a randomized search for ranks from 38 to 41.
    for r in range(38, 42):
        # Try to find a rank-r decomposition
        found = False
        for attempt in range(100):
            # Generate random factor matrices with small integer entries
            U = np.random.randint(-2, 3, size=(nm, r))
            V = np.random.randint(-2, 3, size=(mp, r))
            W = np.random.randint(-2, 3, size=(pn, r))

            # Check if this decomposition is valid
            # We need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all (a,b,c)
            # T[a,b,c] is 1 if there exists i,j,k such that a=idxA(i,j), b=idxB(j,k), c=idxC(k,i)
            # and 0 otherwise.

            # Compute the decomposition
            decomposed = np.tensordot(U, V, axes=1)  # shape (nm, mp, r)
            decomposed = np.tensordot(decomposed, W, axes=1)  # shape (nm, mp, pn)

            # Compute the target tensor
            T = np.zeros((nm, mp, pn), dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[idxA(i,j), idxB(j,k), idxC(k,i)] = 1

            # Check if decomposed matches T
            if np.array_equal(decomposed, T):
                found = True
                break

        if found:
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, fall back to the schoolbook decomposition (rank 42)
    # or the known rank-39 decomposition from the current program.

    # Let's construct the schoolbook decomposition (rank 42) as a fallback.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = idxA(i, j)
                b = idxB(j, k)
                c = idxC(k, i)
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
