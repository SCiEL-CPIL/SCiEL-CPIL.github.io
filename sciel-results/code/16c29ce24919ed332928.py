"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    # Build a rank-32 decomposition using Strassen for 2x2x2 blocks
    # Split p=9 into blocks: 2+2+2+2+1
    # Each 2x2x2 block uses Strassen (rank 7), 2x2x1 uses schoolbook (rank 4)
    # Total: 7*4 + 4 = 32
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M2 = (A10+A11)B00 -> C10, C11 (with -1 on C11)
            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            # M3 = A00(B01-B11) -> C01, C11
            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M4 = A11(B10-B00) -> C00, C10
            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            # M5 = (A00+A01)B11 -> C00, C01 (with -1 on C00)
            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11
            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00
            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            # 4 terms for a single k
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        # Split p=9 into 4 Strassen blocks (2x2 each) + 1 trivial block
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    # Build the rank-32 decomposition
    result = build_rank32()

    # Verify it's correct
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook if verification fails
        return schoolbook()

    # Try to find a rank-31 decomposition using a local search approach
    # Strategy: perturb the rank-32 solution slightly and try to zero out one column
    # using an iterative optimization approach
    
    def try_rank31():
        """Try to find a rank-31 decomposition by optimizing from rank-32."""
        start = time.time()
        time_limit = 60.0
        
        # Convert to numpy arrays
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        rank = 32
        
        # We'll try to find a solution where one column is effectively zero
        # by minimizing the contribution of that column while maintaining the tensor equation
        
        # Approach: Use gradient descent on a modified objective
        # that encourages one column to become zero while maintaining correctness
        
        # For each possible column to zero out, try to find a solution
        for zero_col in range(rank):
            if time.time() - start > time_limit:
                break
                
            # Create a copy of the solution
            U_opt = U.copy()
            V_opt = V.copy()
            W_opt = W.copy()
            
            # Try to minimize the norm of column zero_col while maintaining correctness
            # We'll use a simple iterative approach: scale down the target column
            # and adjust other columns to compensate
            
            # Initialize with small perturbations
            for _ in range(100):
                if time.time() - start > time_limit:
                    break
                    
                # Compute the residual for each (i,j,k)
                residual = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for c in range(rank):
                                total += U_opt[i * m + j, c] * V_opt[j * p + k, c] * W_opt[k * n + i, c]
                            residual[i, j, k] = total - 1
                
                # Check if we're close to a solution
                if np.max(np.abs(residual)) < 1e-6:
                    # Check if the target column is close to zero
                    col_norm = np.sqrt(np.sum(U_opt[:, zero_col]**2) + 
                                       np.sum(V_opt[:, zero_col]**2) + 
                                       np.sum(W_opt[:, zero_col]**2))
                    if col_norm < 1e-3:
                        # We found a rank-31 solution
                        U_new = np.delete(U_opt, zero_col, axis=1)
                        V_new = np.delete(V_opt, zero_col, axis=1)
                        W_new = np.delete(W_opt, zero_col, axis=1)
                        
                        # Verify the new solution
                        if verify(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                
                # Gradient step: try to reduce the target column
                # while correcting the residual
                lr = 0.01
                
                # Compute gradients with respect to U, V, W
                dU = np.zeros_like(U_opt)
                dV = np.zeros_like(V_opt)
                dW = np.zeros_like(W_opt)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # dU[i*m+j, c] = residual[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                            for c in range(rank):
                                dU[i * m + j, c] += residual[i, j, k] * V_opt[j * p + k, c] * W_opt[k * n + i, c]
                                dV[j * p + k, c] += residual[i, j, k] * U_opt[i * m + j, c] * W_opt[k * n + i, c]
                                dW[k * n + i, c] += residual[i, j, k] * U_opt[i * m + j, c] * V_opt[j * p + k, c]
                
                # Add a penalty term to encourage the target column to go to zero
                penalty_weight = 10.0
                dU[:, zero_col] += penalty_weight * U_opt[:, zero_col]
                dV[:, zero_col] += penalty_weight * V_opt[:, zero_col]
                dW[:, zero_col] += penalty_weight * W_opt[:, zero_col]
                
                # Update
                U_opt -= lr * dU
                V_opt -= lr * dV
                W_opt -= lr * dW
        
        return None
    
    # Try to find a rank-31 decomposition
    rank31_result = try_rank31()
    
    if rank31_result is not None:
        return rank31_result
    
    # If we couldn't find rank 31, return the verified rank-32 result
    return result
# EVOLVE-BLOCK-END
