import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5-second safety margin

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Target tensor for full 3x3x7 problem
    # T[i*3+j, j*7+k, k*3+i] = 1
    T_full = np.zeros((9, 21, 21))
    for i in range(3):
        for j in range(3):
            for k in range(7):
                T_full[i * 3 + j, j * 7 + k, k * 3 + i] = 1.0

    # Schoolbook decomposition (rank 63) as fallback
    def schoolbook():
        U = np.zeros((9, 63))
        V = np.zeros((21, 63))
        W = np.zeros((21, 63))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    U[i * 3 + j, col] = 1.0
                    V[j * 7 + k, col] = 1.0
                    W[k * 3 + i, col] = 1.0
                    col += 1
        return U, V, W, 63

    # Try to find a lower-rank decomposition using gradient-based optimization
    # with a structured initialization that respects the Kronecker product structure

    def try_structured_rank(target_rank, time_budget=30.0):
        """
        Try to find a decomposition with the given rank using structured initialization.
        Uses a Kronecker-product-inspired initialization:
        - Split the 7 columns of B into two 3-column blocks and one 1-column block
        - Initialize with known structure for 3x3x3 and 3x3x1 sub-problems
        - Then refine with gradient descent
        """
        if time.time() - start_time > deadline - 10:
            return None

        r = target_rank

        # Structured initialization based on Kronecker product
        # The 3x7 output can be seen as 3x3 + 3x3 + 3x1
        # For each sub-block, we can use the known rank for that sub-problem
        # 3x3x3 has known rank 19 (or 20), 3x3x1 has rank 9
        # But we're trying to find a single rank-r decomposition for the whole thing

        # Strategy: Initialize with a "block diagonal" structure
        # First r_33 elements correspond to the first 3x3 block
        # Next r_33 elements correspond to the second 3x3 block  
        # Last r_31 elements correspond to the last 3x1 block

        # For a rank-r decomposition, we need r vectors in each of U, V, W
        # Initialize using the known schoolbook for sub-problems, then pad/truncate

        # Better strategy: Use random initialization with symmetry breaking
        np.random.seed(42 + r)
        
        # Initialize with small random values
        U = np.random.randn(9, r) * 0.3
        V = np.random.randn(21, r) * 0.3
        W = np.random.randn(21, r) * 0.3

        # Add structured bias: encourage the decomposition to respect the block structure
        # For the first r//2 columns, bias toward the first 3x3 block
        # For the next r//2 columns, bias toward the second 3x3 block
        # For the last columns, bias toward the 3x1 block
        
        # Actually, let's use a smarter initialization:
        # Start from the schoolbook rank-63 and try to compress
        if r < 63:
            # Use iterative truncation/SVD-based approach
            # Start with rank-63 and try to reduce rank via optimization
            U_init = np.zeros((9, r))
            V_init = np.zeros((21, r))
            W_init = np.zeros((21, r))
            
            # Initialize from a subset of schoolbook terms
            for i in range(r):
                # Select terms from schoolbook that cover the tensor efficiently
                idx = i % 63
                ii = idx // 21
                jj = (idx // 7) % 3
                kk = idx % 7
                U_init[ii * 3 + jj, i] = 1.0
                V_init[jj * 7 + kk, i] = 1.0
                W_init[kk * 3 + ii, i] = 1.0
            
            U = U_init + np.random.randn(9, r) * 0.1
            V = V_init + np.random.randn(21, r) * 0.1
            W = W_init + np.random.randn(21, r) * 0.1

        # Gradient descent with adaptive learning rate
        lr = 0.02
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(5000):
            if time.time() - start_time > deadline - 15:
                break

            # Compute reconstructed tensor
            T_rec = np.zeros((9, 21, 21))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])

            error = T_rec - T_full
            err_norm = np.max(np.abs(error))

            if err_norm < best_error:
                best_error = err_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err_norm < 1e-8:
                break

            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            for r_idx in range(r):
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])

            # Adaptive learning rate
            if it % 100 == 0 and err_norm > best_error * 1.1:
                lr *= 0.9
            elif it % 200 == 0:
                lr = min(lr * 1.05, 0.1)

            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Periodic restart if stuck
            if it > 1000 and it % 500 == 0 and err_norm > 1e-4:
                np.random.seed(42 + r + it)
                U = best_U + np.random.randn(9, r) * 0.05
                V = best_V + np.random.randn(21, r) * 0.05
                W = best_W + np.random.randn(21, r) * 0.05
                lr = 0.01

        # Check if we found a good solution
        if best_error < 1e-6:
            # Final polish with higher precision
            lr = 0.001
            for it in range(2000):
                if time.time() - start_time > deadline - 5:
                    break
                T_rec = np.zeros((9, 21, 21))
                for r_idx in range(r):
                    T_rec += np.einsum('a,b,c->abc', best_U[:, r_idx], best_V[:, r_idx], best_W[:, r_idx])
                error = T_rec - T_full
                if np.max(np.abs(error)) < 1e-10:
                    break
                grad_U = np.zeros_like(best_U)
                grad_V = np.zeros_like(best_V)
                grad_W = np.zeros_like(best_W)
                for r_idx in range(r):
                    grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, best_V[:, r_idx], best_W[:, r_idx])
                    grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, best_U[:, r_idx], best_W[:, r_idx])
                    grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, best_U[:, r_idx], best_V[:, r_idx])
                best_U -= lr * grad_U
                best_V -= lr * grad_V
                best_W -= lr * grad_W
            return best_U, best_V, best_W, r

        return None

    # Try ranks from low to high
    # Lower bound for 3x3x7 is at least 19 (from 3x3x3) but realistically higher
    # Known results suggest rank around 30-40 might be achievable
    for target_rank in range(25, 64):
        if time.time() - start_time > deadline - 20:
            break
        
        result = try_structured_rank(target_rank, time_budget=25.0)
        if result is not None:
            U, V, W, r = result
            # Verify
            T_rec = np.zeros((9, 21, 21))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            if np.max(np.abs(T_rec - T_full)) < 1e-6:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    U, V, W, r = schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
