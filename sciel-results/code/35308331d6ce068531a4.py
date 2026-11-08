import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0  # Safety margin for 120s limit + overhead
    
    def get_seed_solution():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
            nonlocal col
            if col >= rank:
                raise ValueError("Rank overflow")
            for idx, coeff in zip(u_indices, u_coeffs):
                U[idx][col] += coeff
            for idx, coeff in zip(v_indices, v_coeffs):
                V[idx][col] += coeff
            for idx, coeff in zip(w_indices, w_coeffs):
                W[idx][col] += coeff
            col += 1

        def add_strassen_block(k_start):
            a00, a01, a10, a11 = 0, 1, 3, 4
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
            add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
            add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
            add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
            add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
            add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
            add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [1], [b06], [1], [c06], [1])
        add_term([1], [1], [b16], [1], [c06], [1])
        add_term([3], [1], [b06], [1], [c16], [1])
        add_term([4], [1], [b16], [1], [c16], [1])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [1], [b2k], [1], [c0k], [1])
            add_term([5], [1], [b2k], [1], [c1k], [1])
            
        return {"U": U, "V": V, "W": W}

    def build_tensor():
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    def verify_solution(U, V, W, T, tol=1e-9):
        R = len(U[0])
        # Convert to numpy arrays for vectorized check
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check integer/half-integer
        # val * 2 should be close to an integer
        if not np.allclose(U_np * 2, np.round(U_np * 2), atol=tol):
            return False
        if not np.allclose(V_np * 2, np.round(V_np * 2), atol=tol):
            return False
        if not np.allclose(W_np * 2, np.round(W_np * 2), atol=tol):
            return False

        # Check residual using einsum
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to reshape U, V, W to match indices
        # U is (n*m, R). Index i*m+j.
        # V is (m*p, R). Index j*p+k.
        # W is (p*n, R). Index k*n+i.
        
        # Reshape U to (n, m, R)
        U_reshaped = U_np.reshape(n, m, R)
        # Reshape V to (m, p, R)
        V_reshaped = V_np.reshape(m, p, R)
        # Reshape W to (p, n, R)
        W_reshaped = W_np.reshape(p, n, R)
        
        # Compute T_reconstructed[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        # A[i, j, k, r] = U[i, j, r] * V[j, k, r]
        A = U_reshaped[:, :, None, :] * V_reshaped[None, :, :, :]
        
        # W_transposed[i, k, r] = W_reshaped[k, i, r]
        W_transposed = W_reshaped.transpose(1, 0, 2) # (n, p, R)
        
        # P[i, j, k, r] = A[i, j, k, r] * W_transposed[i, k, r]
        P = A * W_transposed[:, None, :, :]
        
        # Sum over r
        S = np.sum(P, axis=3) # (n, m, p)
        
        # Check if S is close to 1
        if not np.allclose(S, 1.0, atol=tol):
            return False
            
        return True

    sol = get_seed_solution()
    T = build_tensor()
    
    # Verification step
    if verify_solution(sol["U"], sol["V"], sol["W"], T):
        return sol
    else:
        raise ValueError("Seed solution failed verification")
# EVOLVE-BLOCK-END
