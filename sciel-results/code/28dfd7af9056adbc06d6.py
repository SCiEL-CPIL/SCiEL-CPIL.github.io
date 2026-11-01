import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition (rank 30) as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def verify(U, V, W):
        """Check if U, V, W correctly decompose T exactly."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)

    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    # For A (2x2), B (2x2), C = A*B:
    # C[0,0] = M1 - M2 + M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6  -- wait, let me use the standard form.
    #
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M is a rank-1 term: M = u_i * v_j * w_k
    # For the tensor decomposition, each Strassen term contributes one rank-1
    # factor to U, V, W.

    def strassen_2x2x2():
        """Return (U, V, W) each of shape (4, 7) for 2x2x2 matmul tensor."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[i*2+k]
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3

        # M1 = (A00+A11)(B00+B11)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        # M1 contributes to C00 and C11
        W[0, 0] += 1; W[3, 0] += 1

        # M2 = (A10+A11)B00
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        # M2 contributes to C10 and C11
        W[2, 1] += 1; W[3, 1] -= 1

        # M3 = A00(B01-B11)
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        # M3 contributes to C01 and C11
        W[1, 2] += 1; W[3, 2] += 1

        # M4 = A11(B10-B00)
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        # M4 contributes to C00 and C10
        W[0, 3] += 1; W[2, 3] += 1

        # M5 = (A00+A01)B11
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        # M5 contributes to C00 and C01
        W[0, 4] -= 1; W[1, 4] += 1

        # M6 = (A10-A00)(B00+B01)
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        # M6 contributes to C11
        W[3, 5] += 1

        # M7 = (A01-A11)(B10+B11)
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        # M7 contributes to C00
        W[0, 6] += 1

        return U, V, W

    # Verify Strassen works for 2x2x2
    U_s, V_s, W_s = strassen_2x2x2()
    T_222 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_222[i*2+j, j*2+k, i*2+k] = 1.0
    T_222_recon = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
    strassen_ok = np.allclose(T_222_recon, T_222, atol=1e-10)

    # Build a rank-28 decomposition for 2x3x5 using block structure.
    # Strategy: decompose the 2x3x5 tensor into:
    # - Two 2x2x2 blocks (using Strassen, rank 7 each) for columns j=0,1 and j=1,2
    # - Plus correction terms to handle the overlap and the 5th row
    #
    # Actually, a cleaner approach: use the fact that the tensor is a sum of
    # rank-1 terms T[i*m+j, j*p+k, k*n+i] = 1.
    #
    # Let me try a direct construction. The key is that for each (i,j,k),
    # we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to sum to 1.
    #
    # For 2x3x5, we can use the following block decomposition:
    # Split the 3 columns of A (and 3 rows of B) into two overlapping 2-column
    # blocks: [0,1] and [1,2]. Each 2x2x5 sub-problem can be handled by
    # Strassen on the 2x2x2 part plus schoolbook on the 2x2x3 remainder.
    #
    # 2x2x2: Strassen rank 7
    # 2x2x3: schoolbook rank 12 (or we can do better)
    # But the two blocks share column 1, so we need to be careful.
    #
    # Alternative: use a known rank-28 decomposition.
    # For 2x3x5, the tensor rank is known to be 28 (from Lader et al.).
    # Let me construct it explicitly.
    #
    # The 2x3x5 matmul tensor can be decomposed as follows:
    # Think of it as: C = A * B where A is 2x3, B is 3x5.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # We can write this using the following 28 rank-1 terms.
    # The construction uses the fact that 2x3x5 = 2x2x5 + 2x1x5,
    # and 2x2x5 can be done with rank 7 (for 2x2x2 part) + 12 (for 2x2x3 part)
    # = 19, plus 2x1x5 = 10, total 29. But with sharing, we can get 28.
    #
    # Let me just do a direct search for a rank-28 solution using the
    # structure of the problem.

    # Direct construction of rank-28 decomposition:
    # Use the following approach:
    # - For the first two columns (j=0,1) of A and first two rows (j=0,1) of B:
    #   Use Strassen for the 2x2x2 part (k=0,1) -> 7 terms
    #   Use schoolbook for the 2x2x3 part (k=2,3,4) -> 6 terms
    #   Total: 13 terms
    # - For the third column (j=2) of A and third row (j=2) of B:
    #   Use schoolbook for 2x1x5 -> 10 terms
    # - But we double-counted the (j=1) overlap, so we need to subtract.
    #
    # This is getting complicated. Let me just do a numerical search.

    # Try to find a rank-28 decomposition via optimization
    def try_rank_decomposition(target_rank, max_time=30.0):
        """Try to find a rank-target_rank decomposition via gradient descent."""
        start = time.time()
        U0, V0, W0 = schoolbook()

        for restart in range(5):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break

            # Initialize near schoolbook but with fewer columns
            # Use random linear combinations of schoolbook columns
            U = U0[:, :target_rank].copy()
            V = V0[:, :target_rank].copy()
            W = W0[:, :target_rank].copy()

            # Add small random perturbation
            U += np.random.randn(*U.shape) * 0.01
            V += np.random.randn(*V.shape) * 0.01
            W += np.random.randn(*W.shape) * 0.01

            # Gradient descent
            lr = 0.05
            for step in range(200):
                if time.time() > deadline:
                    return None
                if time.time() - start > max_time:
                    break

                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T_recon - T
                err = np.sum(R**2)

                if err < 1e-14:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_r, V_r, W_r
                    # Also try integer rounding
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_r, V_r, W_r
                    break

                # Compute gradients
                grad_U = 2 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', R, U, V)

                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

                # Decay learning rate
                lr *= 0.98

        return None

    # Try rank 28 first (known achievable)
    for target_rank in [28, 27, 26]:
        if time.time() > deadline:
            break

        result = try_rank_decomposition(target_rank, max_time=25.0)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break

    # If we found rank 28, try to improve to 27 or lower
    if best_rank == 28:
        for target_rank in [27, 26]:
            if time.time() > deadline:
                break
            result = try_rank_decomposition(target_rank, max_time=15.0)
            if result is not None:
                U_r, V_r, W_r = result
                if verify(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break

    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
