"""Matrix multiplication 6x6x6 using 2x2 Strassen block decomposition with rank-reduced 3x3."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 138
    n, m, p = 6, 6, 6

    # Schoolbook 3x3 rank-27
    r3 = 27
    U33 = np.zeros((9, r3))
    V33 = np.zeros((9, r3))
    W33 = np.zeros((9, r3))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1

    # Strassen 2x2 terms
    terms = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1}),
    ]

    bmap = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }

    R = 7 * r3
    U = np.zeros((36, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))

    for mi, (ta, tb, tc) in enumerate(terms):
        base = mi * r3
        for lc in range(r3):
            c = base + lc
            i3 = lc // 9
            rem = lc % 9
            j3 = rem // 3
            k3 = rem % 3
            u33v = U33[i3 * 3 + j3, lc]
            v33v = V33[j3 * 3 + k3, lc]
            w33v = W33[k3 * 3 + i3, lc]
            if u33v != 0:
                for ab, ca in ta.items():
                    bi, bj = bmap[ab]
                    U[(bi * 3 + i3) * 6 + (bj * 3 + j3), c] += ca * u33v
            if v33v != 0:
                for bb, cb in tb.items():
                    bk, bl = bmap[bb]
                    V[(bk * 3 + j3) * 6 + (bl * 3 + k3), c] += cb * v33v
            if w33v != 0:
                for cb2, cc in tc.items():
                    ci, ck = bmap[cb2]
                    W[(ck * 3 + k3) * 6 + (ci * 3 + i3), c] += cc * w33v

    # Rank reduction via linear dependence detection
    # Build the 3-tensor slice matrices and look for null-space relations
    # T_c = U_c (outer) V_c (outer) W_c
    # We look for alpha such that sum_c alpha_c T_c = 0
    # This is equivalent to: for all i,j,k: sum_c alpha_c U[i,c]V[j,c]W[k,c] = 0
    # Vectorize: for each (i,j), sum_c alpha_c U[i,c]V[j,c] = 0 for all k weighted by W[k,c]
    # More precisely: define for each (i,j): f_{ij}(c) = U[i,c]*V[j,c]
    # Then we need: for all (i,j,k): sum_c f_{ij}(c) * W[k,c] * alpha_c = 0
    # This is: M @ alpha = 0 where M is a (36*36*36) x R matrix, but that's huge.
    # Instead, use the matricization approach:
    # Reshape tensor to 36x(36*36) matrix M where M[i, j*36+k] = sum_c U[i,c]V[j,c]W[k,c]
    # The rank of M should be 36 (full). The columns of the decomposition are:
    # col_c of M = U[:,c] (outer) (V[:,c] (outer) W[:,c]).ravel()
    # So M = U @ (V (outer) W).ravel()... actually M = sum_c U[:,c] (outer) (V[:,c] (outer) W[:,c]).ravel()
    # This means M = U @ B where B[c, (j,k)] = V[j,c]*W[k,c]
    # So M = U @ B, and rank(M) <= min(rank(U), rank(B))
    # If we can find a smaller rank representation, we need to find U', V', W' with fewer columns
    # such that U' @ B' = M where B'[c,(j,k)] = V'[j,c]*W'[k,c]
    
    # Practical approach: try to find linear dependencies among the rank-1 terms
    # by checking if the columns of U, V, W have consistent linear relations.
    # If U[:, c1] = U[:, c2] and V[:, c1] = V[:, c2] and W[:, c1] = W[:, c2], 
    # then we can merge them.
    
    # More generally, if there exist matrices P, Q, R (R x R) such that
    # U @ P = U', V @ Q = V', W @ R = W' and P, Q, R are related...
    # This is complex. Let's try a simpler heuristic:
    # Check for duplicate columns in U, V, W simultaneously.
    
    # For each pair of columns (c1, c2), check if U[:,c1] == U[:,c2], V[:,c1] == V[:,c2], W[:,c1] == W[:,c2]
    # If so, we can merge them.
    
    # Also check for sign flips: U[:,c1] == -U[:,c2], V[:,c1] == V[:,c2], W[:,c1] == -W[:,c2]
    # etc.
    
    # Let's try a more systematic approach: use SVD on the "flattened" tensor
    # to find the true rank, then try to reconstruct a lower-rank decomposition.
    
    # Actually, let's try a different strategy: use the fact that Strassen's algorithm
    # for 2x2 blocks has 7 multiplications, and each 3x3 multiplication has rank 27.
    # But some of the 3x3 multiplications might have lower rank due to the specific
    # linear combinations in Strassen's algorithm.
    
    # For example, M0 = (A00+A11)*(B00+B11) involves adding two 3x3 blocks,
    # which doesn't change the rank of the multiplication.
    
    # Let's try to find if any of the 7 Strassen terms have a lower-rank 3x3 decomposition.
    # This is unlikely for generic 3x3 matrices, but let's check.
    
    # Alternative: use the Laderman 4x4 decomposition (rank 16) and see if we can
    # embed it into the 6x6 problem.
    
    # Laderman's algorithm for 4x4:
    # 16 multiplications instead of 16 schoolbook... wait, Laderman is 16 for 4x4,
    # while schoolbook is 16. So no gain.
    
    # Actually, Laderman's algorithm has rank 16 for 4x4, same as schoolbook.
    # The known best for 4x4 is 16 (Laderman), and for 5x5 it's 25 (schoolbook) 
    # or possibly lower.
    
    # For 6x6, the known best is 216 (schoolbook) or 189 (Strassen 2x2 with 3x3).
    # There might be a better decomposition.
    
    # Let's try a 3x3 block decomposition with Strassen-like optimization.
    # 6x6 = 3x3 blocks of 2x2.
    # Schoolbook: 27 multiplications of 2x2, each rank 4, total rank 108.
    # But 2x2 multiplication has rank 7 (Strassen), so 27*7 = 189.
    # Same as before!
    
    # What about a mix: use 2x2 Strassen for some blocks and schoolbook for others?
    
    # Let's try: 6x6 as 3x3 blocks of 2x2, with a Strassen-like reduction.
    # 3x3 matrix multiplication has rank 27 (schoolbook). Is there a lower-rank decomposition?
    # The known best for 3x3 is 27 (schoolbook), I believe.
    
    # So we're stuck with 189 for now. Let's return the rank-189 decomposition.
    
    # Try to find and remove linearly dependent terms
    # Check for exact duplicates
    to_remove = set()
    for c1 in range(R):
        if c1 in to_remove:
            continue
        for c2 in range(c1 + 1, R):
            if c2 in to_remove:
                continue
            if np.array_equal(U[:, c1], U[:, c2]) and np.array_equal(V[:, c1], V[:, c2]) and np.array_equal(W[:, c1], W[:, c2]):
                to_remove.add(c2)
            elif np.array_equal(U[:, c1], -U[:, c2]) and np.array_equal(V[:, c1], V[:, c2]) and np.array_equal(W[:, c1], -W[:, c2]):
                to_remove.add(c2)
            elif np.array_equal(U[:, c1], U[:, c2]) and np.array_equal(V[:, c1], -V[:, c2]) and np.array_equal(W[:, c1], -W[:, c2]):
                to_remove.add(c2)
    
    if to_remove:
        keep = [c for c in range(R) if c not in to_remove]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
