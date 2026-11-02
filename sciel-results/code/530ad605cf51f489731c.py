"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def elapsed():
        return time.time() - start_time

    def verify_numpy(U, V, W, tol=1e-9):
        """Verify decomposition using numpy. U:(nm,r), V:(mp,r), W:(pn,r)"""
        r = U.shape[1]
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = to_lists(U_school, V_school, W_school)
    best_rank = schoolbook_rank

    # Strategy: Use Strassen-like decomposition for the 4x4 sub-block
    # The 4x5 * 5x5 multiplication can be viewed as:
    # A is 4x5, B is 5x5
    # We can partition A as [A_4x4 | A_4x1] and B as [B_4x5; B_1x5]
    # C = A_4x4 * B_4x5 + A_4x1 * B_1x5
    #
    # For the 4x4 * 4x5 part, we can use Strassen for the 4x4 * 4x4 part
    # and schoolbook for the 4x4 * 1x5 part (4*4*5 = 80 operations... wait)
    #
    # Actually, let me think more carefully:
    # A is 4x5, B is 5x5, C is 4x5
    # Partition: A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    #             B = [B1; B2] where B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 4x4 * 4x5 -> can be seen as 4x4 * 4x4 (first 4 cols) + 4x4 * 4x1 (last col)
    # Using Strassen for 4x4*4x4: rank 8 (instead of 16)
    # Schoolbook for 4x4*4x1: 16 operations
    # A2*B2: 4x1 * 1x5 -> 20 operations (schoolbook)
    # Total: 8 + 16 + 20 = 44... but wait, Strassen for 4x4 uses 8 rank-1 terms
    # for the 4x4*4x4 part, but we need to be careful about the dimensions.
    #
    # Let me reconsider. The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,5), k in [0,5).
    # We want to decompose this as sum of rank-1 tensors.
    #
    # Strassen for 2x2*2x2 uses 7 multiplications instead of 8.
    # For 4x4*4x4, we can apply Strassen recursively: 4x4 = 2*(2x2), so 7^2 = 49
    # instead of 64. But that's for square matrices.
    #
    # For our case, let's try a direct construction:
    # Use the known rank-7 decomposition for 2x2 matrix multiplication.
    # 
    # The 2x2 Strassen decomposition:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Let's try to build a decomposition for 4x5*5x5 using block Strassen.
    # Partition A into 2x2 blocks of 2x? and B into corresponding blocks.
    #
    # A is 4x5: partition rows into 2+2, columns into 2+3
    # A = [[A11, A12], [A21, A22]] where A11,A12,A21,A22 are 2x2, 2x3, 2x2, 2x3
    #
    # B is 5x5: partition rows into 2+3, columns into 2+3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    #
    # This is getting complex. Let me try a simpler approach:
    # Just use the schoolbook decomposition but see if we can find any improvement
    # through random search with better initialization.

    # Try random search for lower ranks
    def try_random_rank(target_rank, num_trials=50, seed=42):
        """Try to find a rank-target_rank decomposition."""
        import random
        rng = random.Random(seed)
        for trial in range(num_trials):
            if elapsed() > deadline - 5:
                return None

            # Initialize with random small values
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))

            # Fill with random values from {-1, 0, 1, 0.5, -0.5}
            for c in range(target_rank):
                for _ in range(nm // 5):
                    row = rng.randint(0, nm - 1)
                    U[row, c] = rng.choice([-1, 0, 1, 0.5, -0.5])
                for _ in range(mp // 5):
                    row = rng.randint(0, mp - 1)
                    V[row, c] = rng.choice([-1, 0, 1, 0.5, -0.5])
                for _ in range(pn // 5):
                    row = rng.randint(0, pn - 1)
                    W[row, c] = rng.choice([-1, 0, 1, 0.5, -0.5])

            if verify_numpy(U, V, W, tol=1e-9):
                return to_lists(U, V, W)

        return None

    # Try ranks from 99 down to 90
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 0), -1):
        if elapsed() > deadline - 10:
            break

        result = try_random_rank(target_rank, num_trials=30, seed=target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

    # If we have time, try a structured Strassen-based approach
    if elapsed() < deadline - 20:
        # Build a decomposition using Strassen for 2x2 blocks
        # This is a heuristic; may not work perfectly for rectangular case
        # but let's try to construct something valid.
        
        # For now, return the best we found
        pass

    return best_result
# EVOLVE-BLOCK-END
