import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Dimensions
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6

    # Build the target tensor T of shape (dim_U, dim_V, dim_W)
    # T[i, j, k] = 1 if A[i, j] * B[j, k] contributes to C[i, k]
    # In the decomposition: C = sum_r U[:, r] * V[:, r] * W[:, r]
    # The tensor T is defined by T[a, b, c] = 1 if (a, b, c) corresponds to a term in the multiplication.
    # a is index in A (0..5), b is index in B (0..8), c is index in C (0..5).
    # A index a -> i = a // m, j = a % m
    # B index b -> j = b // p, k = b % p
    # C index c -> k = c // n, i = c % n  (Note: Task says W[k*n+i, r], so C index is k*n + i)
    
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Function to check if a set of U, V, W matrices (rank R) decomposes T
    def check_decomposition(U, V, W, R, atol=1e-4):
        # Reconstruct tensor
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can compute this efficiently
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # But we only need to check if it matches T where T is non-zero?
        # Actually, we need exact match everywhere.
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_approx, T, atol=atol):
            return False
        return True

    # Function to snap to half-integers
    def snap_half(X):
        return np.round(X * 2) / 2.0

    # Try to find a rank 16 decomposition
    # If found, return it.
    # Otherwise, return the known rank 17.
    
    best_rank = 17
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's secure the rank 17 solution to have a fallback
    # We can reuse the logic from the current program or just construct it here.
    # To save code, I will implement a quick rank 17 construction or just use the logic from the prompt's current program.
    # However, to keep this self-contained and "different", I will try to find 16 first.
    
    # Search for Rank 16
    target_rank = 16
    if time.time() < deadline - 10:
        # Use scipy if available
        try:
            from scipy.optimize import least_squares
            import random
            
            # Number of variables: R * (dim_U + dim_V + dim_W)
            R = target_rank
            n_vars = R * (dim_U + dim_V + dim_W)
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
            def unpack_params(x):
                U = x[:R*dim_U].reshape(dim_U, R)
                V = x[R*dim_U:R*dim_U + R*dim_V].reshape(dim_V, R)
                W = x[R*dim_U + R*dim_V:].reshape(dim_W, R)
                return U, V, W
            
            def residuals(x):
                U, V, W = unpack_params(x)
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                # Residual is the difference
                return (T_approx - T).flatten()
            
            # Try a few random initializations
            found = False
            for attempt in range(5): # Limit attempts to save time
                if time.time() > deadline - 15:
                    break
                random.seed(attempt)
                np.random.seed(attempt)
                # Initialize with small random values
                x0 = np.random.normal(0, 1, n_vars)
                
                try:
                    res = least_squares(residuals, x0, method='lm', max_nfev=10000)
                    if res.cost < 1e-8:
                        U, V, W = unpack_params(res.x)
                        # Snap to half-integers
                        U_snap = snap_half(U)
                        V_snap = snap_half(V)
                        W_snap = snap_half(W)
                        
                        if check_decomposition(U_snap, V_snap, W_snap, R, atol=1e-6):
                            best_rank = R
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            found = True
                            break
                except Exception:
                    pass
                    
            if found:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
        except ImportError:
            pass

    # If rank 16 not found, construct Rank 17
    # This is the same logic as the current program, but I'll write it cleanly.
    # We know 17 is valid.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: 2x2x2 Strassen (Rank 7)
    U1 = np.zeros((6, 7))
    V1 = np.zeros((9, 7))
    W1 = np.zeros((6, 7))
    
    # M1: (A00 + A11) * (B00 + B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2: (A10 + A11) * B00
    U1[1, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1
    
    # M3: A00 * (B01 - B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # M4: A11 * (B10 - B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # M5: (A00 + A01) * B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[1, 4] = 1
    
    # M6: (A10 - A00) * (B00 + B01)
    U1[1, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # M7: (A01 - A11) * (B10 + B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Part 2: 2x3x1 Schoolbook (Rank 6)
    # A[0:2, 0:3], B[0:3, 2] -> C[0:2, 2]
    U2 = np.zeros((6, 6))
    V2 = np.zeros((9, 6))
    W2 = np.zeros((6, 6))
    
    # A00*B02 -> C02
    U2[0, 0] = 1; V2[2, 0] = 1; W2[2, 0] = 1
    # A01*B12 -> C02
    U2[1, 1] = 1; V2[5, 1] = 1; W2[2, 1] = 1
    # A02*B22 -> C02
    U2[2, 2] = 1; V2[8, 2] = 1; W2[2, 2] = 1
    # A10*B02 -> C12
    U2[3, 3] = 1; V2[2, 3] = 1; W2[5, 3] = 1
    # A11*B12 -> C12
    U2[4, 4] = 1; V2[5, 4] = 1; W2[5, 4] = 1
    # A12*B22 -> C12
    U2[5, 5] = 1; V2[8, 5] = 1; W2[5, 5] = 1
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Part 3: 2x1x2 Schoolbook (Rank 4)
    # A[0:2, 0:2], B[2, 0:2] -> C[0:2, 0:2]
    # C[i,k] += A[i,2] * B[2,k]
    U3 = np.zeros((6, 4))
    V3 = np.zeros((9, 4))
    W3 = np.zeros((6, 4))
    
    # A02*B20 -> C00
    U3[2, 0] = 1; V3[6, 0] = 1; W3[0, 0] = 1
    # A02*B21 -> C01
    U3[2, 1] = 1; V3[7, 1] = 1; W3[1, 1] = 1
    # A12*B20 -> C10
    U3[5, 2] = 1; V3[6, 2] = 1; W3[2, 2] = 1
    # A12*B21 -> C11
    U3[5, 3] = 1; V3[7, 3] = 1; W3[3, 3] = 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    # Final verification
    if not check_decomposition(U, V, W, 17, atol=1e-6):
        # Fallback to schoolbook rank 18 if 17 fails (should not)
        best_rank = 18
        U = np.zeros((6, 18))
        V = np.zeros((9, 18))
        W = np.zeros((6, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
