"""Matrix multiplication 2x4x5 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def build_direct_40():
        """Direct rank-40 decomposition: each output entry C[k,i] = sum_j A[i,j]*B[j,k]"""
        rank = n * m * p  # 2*4*5 = 40
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j), col] = 1
                    V[v_idx(j, k), col] = 1
                    W[w_idx(k, i), col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        # Check all "1" entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_idx = i * m + j
                    B_idx = j * p + k
                    C_idx = k * n + i
                    computed = np.dot(U[A_idx, :] * V[B_idx, :], W[C_idx, :])
                    if abs(computed - 1) > tol:
                        return False

        # Check that non-matching entries are 0
        # T[A_idx, B_idx, C_idx] = 0 unless j_A == j_B and k_B == k_C and i_A == i_C
        # A_idx = i*m+j, B_idx = j2*p+k2, C_idx = k3*n+i3
        # Need j == j2, k2 == k3, i == i3
        for A_idx in range(n * m):
            i_A = A_idx // m
            j_A = A_idx % m
            for B_idx in range(m * p):
                j_B = B_idx // p
                k_B = B_idx % p
                for C_idx in range(p * n):
                    k_C = C_idx // n
                    i_C = C_idx % n
                    if not (j_A == j_B and k_B == k_C and i_A == i_C):
                        computed = np.dot(U[A_idx, :] * V[B_idx, :], W[C_idx, :])
                        if abs(computed) > tol:
                            return False
        return True

    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2

    def try_optimize(rank, max_iters=5000, lr=0.01, seed=42):
        """Try to find a rank-r decomposition via gradient descent."""
        if time.time() > deadline:
            return None

        np.random.seed(seed)
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1

        # Collect all the "1" entries
        ones = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ones.append((i * m + j, j * p + k, k * n + i))

        # Also collect a sample of "0" entries to enforce sparsity
        zeros = []
        for _ in range(2000):
            A_idx = random.randint(0, n * m - 1)
            B_idx = random.randint(0, m * p - 1)
            C_idx = random.randint(0, p * n - 1)
            i_A = A_idx // m
            j_A = A_idx % m
            j_B = B_idx // p
            k_B = B_idx % p
            k_C = C_idx // n
            i_C = C_idx % n
            if not (j_A == j_B and k_B == k_C and i_A == i_C):
                zeros.append((A_idx, B_idx, C_idx))

        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Compute loss: sum of (computed - target)^2 over all entries
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            # "1" entries
            for A_idx, B_idx, C_idx in ones:
                # T = sum_r U[A,r]*V[B,r]*W[C,r]
                UV = U[A_idx, :] * V[B_idx, :]  # shape (rank,)
                T_val = np.dot(UV, W[C_idx, :])
                err = T_val - 1.0
                loss += err * err
                # dT/dU[A,r] = V[B,r]*W[C,r]
                dU = err * V[B_idx, :] * W[C_idx, :]
                grad_U[A_idx, :] += dU
                # dT/dV[B,r] = U[A,r]*W[C,r]
                dV = err * U[A_idx, :] * W[C_idx, :]
                grad_V[B_idx, :] += dV
                # dT/dW[C,r] = U[A,r]*V[B,r]
                dW = err * U[A_idx, :] * V[B_idx, :]
                grad_W[C_idx, :] += dW

            # "0" entries (sampled)
            for A_idx, B_idx, C_idx in zeros:
                UV = U[A_idx, :] * V[B_idx, :]
                T_val = np.dot(UV, W[C_idx, :])
                err = T_val - 0.0
                loss += err * err
                dU = err * V[B_idx, :] * W[C_idx, :]
                grad_U[A_idx, :] += dU
                dV = err * U[A_idx, :] * W[C_idx, :]
                grad_V[B_idx, :] += dV
                dW = err * U[A_idx, :] * V[B_idx, :]
                grad_W[C_idx, :] += dW

            # Gradient descent step
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if best_loss < 1e-10:
                break

        # Round to half-integers and verify
        U_r = round_to_half(best_U)
        V_r = round_to_half(best_V)
        W_r = round_to_half(best_W)

        if verify_decomposition(U_r, V_r, W_r, tol=1e-6):
            return {"U": U_r, "V": V_r, "W": W_r}

        # Also try with original (unrounded) if it's close enough
        if verify_decomposition(best_U, best_V, best_W, tol=1e-6):
            return {"U": best_U, "V": best_V, "W": best_W}

        return None

    def build_strassen_36():
        """Strassen-based rank-36 decomposition."""
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0

        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices

            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1

            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1

            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1

            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1

        # Block 1: A_L * B_11 -> C_11
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)

        # Block 2: A_R * B_21 -> C_11
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)

        # Block 3: A_L * B_12 -> C_12
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)

        # Block 4: A_R * B_22 -> C_12
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)

        # Block 5: A_L * B_15 (2x2 * 2x1) - direct rank 4
        U[u_idx(0,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1

        # Block 6: A_R * B_25 (2x2 * 2x1) - direct rank 4
        U[u_idx(0,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1

        assert col == rank, f"Expected {rank}, got {col}"
        return {"U": U, "V": V, "W": W}

    # Start with the best known baseline
    baseline = build_strassen_36()
    best_rank = 36
    best_result = baseline

    if verify_decomposition(baseline['U'], baseline['V'], baseline['W']):
        pass  # Good, baseline is valid

    # Try to find lower rank decompositions using optimization
    # Try ranks from 35 down to 20
    for target_rank in range(35, 19, -1):
        if time.time() > deadline:
            break

        # Try multiple random seeds
        for seed in [42, 123, 456, 789, 1000, 2024]:
            if time.time() > deadline:
                break

            result = try_optimize(target_rank, max_iters=3000, lr=0.05, seed=seed)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break  # Found valid decomposition at this rank, move to lower

    # If we haven't found anything better, try a different approach:
    # Use the direct rank-40 and try to merge terms
    if best_rank >= 36:
        if time.time() < deadline - 10:
            # Try a structured approach: use rank 32 with careful design
            # For 2x4x5, try to exploit the 2-row structure
            # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
            # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]

            # Try rank 30 with optimization
            for seed in [42, 123, 456]:
                if time.time() > deadline:
                    break
                result = try_optimize(30, max_iters=5000, lr=0.03, seed=seed)
                if result is not None:
                    best_rank = 30
                    best_result = result
                    break

    # Final verification
    if verify_decomposition(best_result['U'], best_result['V'], best_result['W']):
        return best_result
    else:
        # Return the verified baseline
        return baseline

