import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def verify(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        # Check exactness: all entries should be exactly 1
        if not np.allclose(T_prime, 1.0, atol=1e-10):
            return False
        # Check if entries are close to integers (for half-integer support)
        if not np.allclose(T_prime, np.round(T_prime), atol=1e-10):
            return False
        return True

    def _build_strassen_rank21():
        """
        Build the rank-21 Strassen decomposition for 3x3x3 matrix multiplication.
        Returns U, V, W matrices for the 3x3x3 case (n=3, m=3, p=3).
        """
        r = 21
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*3, r), dtype=float)  # p=3 for the 3x3x3 part
        W = np.zeros((3*n, r), dtype=float)  # p=3 for the 3x3x3 part
        
        # Strassen's 7 multiplications for 2x2x2, extended to 3x3x3
        # The standard rank-21 algorithm for 3x3 matrix multiplication
        # Uses the Strassen-Winograd algorithm
        
        # Define the 21 terms
        # M1 = (A00 + A22)(B00 + B22)
        # M2 = (A20 + A22)B00
        # M3 = A00(B01 - B21)
        # M4 = A22(B21 - B01)
        # M5 = (A01 + A11)B10
        # M6 = (A10 + A20)(B02 + B12)
        # M7 = (A00 + A10)(B12 - B22)
        # M8 = (A11 + A21)(B00 + B10)
        # M9 = (A01 - A11 + A21)(B10 + B20)
        # M10 = (A00 - A10 + A11 - A21 + A22)(B01 + B11 - B21)
        # M11 = (A02 + A12 - A22)(B11 + B21)
        # M12 = (A02 - A12 + A22)(B10 + B11 + B20 + B21)
        # M13 = (A00 + A02 - A10 + A12 - A20)(B02 + B12)
        # M14 = (A01 + A02 + A11 + A12 - A20 - A21)(B01 + B02 + B11 + B12)
        # M15 = (A00 + A01 + A02 + A10 + A11 + A12 - A20 - A21 - A22)(B00 + B01 + B02 + B10 + B11 + B12 + B20 + B21 + B22)
        # M16 = (A00 + A01 - A11 + A21 - A22)(B12 - B22)
        # M17 = (A00 - A11 + A12 + A22)(B02 - B12 + B22)
        # M18 = (A01 - A11 + A12 - A22)(B02 + B12)
        # M19 = (A00 - A10 + A11 - A21)(B01 + B11 - B21)
        # M20 = (A02 + A12 - A20 - A21)(B00 + B01 + B10 + B11)
        # M21 = (A00 + A02 + A10 + A12 + A20 + A22)(B00 + B02 + B20 + B22)
        
        # This is getting complex. Let me use a known rank-21 decomposition.
        # Actually, let me use a simpler approach: use the standard 3x3x3 rank-21 algorithm.
        
        # I'll use a known set of 21 terms from the literature
        # For now, let me construct a valid decomposition using a different approach.
        
        # Alternative: Use the fact that we can combine 3x3x3 (rank 21) with 3x3x1 (rank 3)
        # to get 3x3x4 with rank at most 24, but we want lower.
        
        # Let me try a direct construction for 3x3x4 with rank 20.
        # This requires careful algebraic manipulation.
        
        # For now, return a placeholder that will be filled by the search.
        return None

    def _build_rank20_decomposition():
        """
        Attempt to build a rank-20 decomposition for 3x3x4 matrix multiplication.
        Uses algebraic structure to share terms between the 3x3x3 and 3x3x1 parts.
        """
        r = 20
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*p, r), dtype=float)
        W = np.zeros((p*n, r), dtype=float)
        
        # Strategy: Use 15 terms for a partial 3x3x3 decomposition and 5 shared terms
        # that cover both the remaining 3x3x3 part and the 3x3x1 part.
        
        # This is a heuristic construction. We'll use random initialization + local search.
        # Initialize with random small integers
        U[:] = np.random.randint(-2, 3, size=(n*m, r))
        V[:] = np.random.randint(-2, 3, size=(m*p, r))
        W[:] = np.random.randint(-2, 3, size=(p*n, r))
        
        # Local search to minimize error
        U_real = U.astype(float)
        V_real = V.astype(float)
        W_real = W.astype(float)
        
        current_error = np.inf
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Compute current error
            U_3d = U_real.reshape(n, m, r)
            V_3d = V_real.reshape(m, p, r)
            W_3d = W_real.reshape(p, n, r)
            T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            current_error = np.sum((T_prime - 1.0)**2)
            
            if current_error < 1e-10:
                # Try rounding to get exact integers
                U_int = np.round(U_real)
                V_int = np.round(V_real)
                W_int = np.round(W_real)
                if verify(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                # Also try half-integers
                U_half = np.round(U_real * 2) / 2
                V_half = np.round(V_real * 2) / 2
                W_half = np.round(W_real * 2) / 2
                if verify(U_half, V_half, W_half):
                    return U_half, V_half, W_half
            
            # Random perturbation
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randrange(n*m)
                col = random.randrange(r)
                old_val = U_real[idx, col]
                U_real[idx, col] += random.uniform(-1, 1)
            elif mat == 'V':
                idx = random.randrange(m*p)
                col = random.randrange(r)
                old_val = V_real[idx, col]
                V_real[idx, col] += random.uniform(-1, 1)
            else:
                idx = random.randrange(p*n)
                col = random.randrange(r)
                old_val = W_real[idx, col]
                W_real[idx, col] += random.uniform(-1, 1)
            
            # Compute new error
            U_3d = U_real.reshape(n, m, r)
            V_3d = V_real.reshape(m, p, r)
            W_3d = W_real.reshape(p, n, r)
            T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            new_error = np.sum((T_prime - 1.0)**2)
            
            if new_error < current_error:
                current_error = new_error
            else:
                # Revert
                if mat == 'U':
                    U_real[idx, col] = old_val
                elif mat == 'V':
                    V_real[idx, col] = old_val
                else:
                    W_real[idx, col] = old_val
        
        return None

    # Try to find a rank-20 decomposition
    result = _build_rank20_decomposition()
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
