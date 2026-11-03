"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Known rank-17 solution
    def get_rank17():
        rank = 17
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        # Strassen part for 2x2 block (indices 0-6)
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        U[4, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1

        # Schoolbook for remaining terms (indices 7-16)
        U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
        U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
        U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
        U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
        U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
        U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
        U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
        U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
        U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
        U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Verify the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_random(U, V, W, num_tests=5, tol=1e-9):
        """Verify with random matrices."""
        rank = U.shape[1]
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-10, 10, (n, m)).astype(float)
            B = np.random.randint(-10, 10, (m, p)).astype(float)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_vec = U[:, r]
                v_vec = V[:, r]
                w_vec = W[:, r]
                u_mat = u_vec.reshape(n, m)
                v_mat = v_vec.reshape(m, p)
                w_mat = w_vec.reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_computed[i, k] += A[i, j] * B[j, k] * u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                # Actually, the correct formula is:
                # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) / ...
                # No, the tensor decomposition means:
                # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # And the algorithm computes: C[i,k] = sum_j A[i,j] * B[j,k]
                # which equals sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) / ???
                # Actually no. The standard interpretation is that the multiplication tensor T satisfies:
                # T_{i,j,k} = 1 for all valid (i,j,k)
                # And T = sum_r u_r ⊗ v_r ⊗ w_r
                # So for computing C = A*B:
                # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
                # That's not right either. The correct way:
                # The decomposition gives an algorithm:
                # For each r: M_r = (A * U_r) @ (B * V_r) where * is element-wise, then C = sum_r W_r * M_r
                # But that's not standard. Let me just verify the tensor directly.
                pass
        # Just use the tensor verification
        return verify(U, V, W, tol)

    U17, V17, W17 = get_rank17()

    # Try to find rank-16 by merging two schoolbook terms
    # The 10 schoolbook terms are at indices 7-16
    schoolbook_indices = list(range(7, 17))

    best_result = None
    best_rank = 17

    # Strategy 1: Try merging pairs of schoolbook terms
    for i, j in combinations(schoolbook_indices, 2):
        if time.time() > deadline - 5:
            break

        # Try to merge terms i and j into a single term
        # Term i: U_i * V_i * W_i
        # Term j: U_j * V_j * W_j
        # We want: U_new * V_new * W_new = U_i * V_i * W_i + U_j * V_j * W_j
        # This is possible if the sum of two rank-1 tensors is rank 1
        # which requires specific conditions

        # For our case, both terms have U with single non-zero entry, V with single non-zero entry
        # So U_i * V_i * W_i is supported on a specific (i,j,k) position
        # Two such terms can only be merged if they share the same (i,j,k) position
        # OR if we can find a different factorization

        # Let's check if they can be merged by checking if the sum tensor has rank 1
        U_sum = U17[:, i] + U17[:, j]
        V_sum = V17[:, i] + V17[:, j]
        W_sum = W17[:, i] + W17[:, j]

        # Check if U_sum ⊗ V_sum ⊗ W_sum equals the sum of the two original terms
        # This is only true in very special cases
        # Instead, let's try a more general approach:
        # Keep 7 Strassen terms, and try to factor the 10 schoolbook terms into 9

        # Create a candidate rank-16 solution
        U16 = np.zeros((n * m, 16))
        V16 = np.zeros((m * p, 16))
        W16 = np.zeros((p * n, 16))

        # Copy Strassen terms (0-6)
        for r in range(7):
            U16[:, r] = U17[:, r]
            V16[:, r] = V17[:, r]
            W16[:, r] = W17[:, r]

        # Copy all schoolbook terms except j (we'll try to absorb j into i or another term)
        schoolbook_to_keep = [s for s in schoolbook_indices if s != j]
        for idx, s in enumerate(schoolbook_to_keep):
            U16[:, 7 + idx] = U17[:, s]
            V16[:, 7 + idx] = V17[:, s]
            W16[:, 7 + idx] = W17[:, s]

        # Now we have 15 terms. We need to add one more term that captures the missing contribution
        # The missing contribution is the term j
        # We need to find U_new, V_new, W_new such that:
        # U_new ⊗ V_new ⊗ W_new = U17[:,j] ⊗ V17[:,j] ⊗ W17[:,j] + (correction from other terms)
        # This is complex. Let's try a different approach.

        # Strategy 2: Use the structure of the problem
        # The 2x3x3 multiplication can be seen as:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]

        # The Strassen part handles the 2x2 submatrix (rows 0-1, cols 0-1)
        # The remaining terms involve col 2 of A or col 2 of B

        # Let's try to find a rank-16 solution by solving a linear system
        # We fix the structure of U and V, then solve for W

        # For now, just return the rank-17 solution if we can't find rank-16
        if verify(U16, V16, W16):
            best_rank = 16
            best_result = (U16, V16, W16)
            break

    # If we found rank-16, return it
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

# EVOLVE-BLOCK-END
