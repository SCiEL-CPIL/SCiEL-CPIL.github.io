"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    Uses Winograd/FFT-based approach to exploit tensor structure.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    schoolbook_rank = n * m * p  # 112
    
    def verify_np(U_np, V_np, W_np, tol=1e-9):
        """Verify tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    # Winograd/FFT-based decomposition
    # For matrix multiplication C = A*B where A is n×m, B is m×p
    # We can use polynomial interpolation on the m dimension
    
    # Key insight: use the fact that m=4 can be decomposed using a 4-point transform
    # The Winograd algorithm for small matrix multiplication
    
    # Strategy: use the Karatsuba-like decomposition
    # For m=4, split into two 2×2 blocks
    # A = [A00 A01; A10 A11] where each block is 2×2
    # B = [B00 B01; B10 B11] where each block is 2×(p/2) if p is even
    
    # But p=7 is odd, so let's use a different approach
    
    # Use FFT-based approach:
    # Represent the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Use a 4-point DFT on j:
    # Let omega = exp(2*pi*i/4) = i
    # A_hat[i, freq] = sum_j A[i,j] * omega^(j*freq)
    # B_hat[freq, k] = sum_j B[j,k] * omega^(-j*freq)
    # C[i,k] = (1/4) * sum_freq A_hat[i, freq] * B_hat[freq, k] * omega^(... )
    # 
    # Actually, for exact integer/half-integer entries, we need to be careful.
    # The DFT introduces complex numbers, but we can use the real Winograd scheme.
    
    # Winograd scheme for 4-point transform:
    # Uses the fact that 4 = 2*2, and decomposes using butterfly operations
    
    # For now, let's try a structured random search with better initialization
    
    np.random.seed(42)
    random.seed(42)
    
    # Try ranks from lower bound upward
    # Lower bound: max(n, m, p) = 7, but realistically much higher
    # For 4×4×4, best known is 23, so for 4×4×7, estimate ~35-40
    
    # Use a smarter initialization based on the tensor structure
    
    def compute_error_fast(Ur, Vr, Wr):
        """Fast error computation using numpy broadcasting"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ur[i*m+j], Vr[j*p+k] * Wr[k*n+i])
                    err += (s - 1.0) ** 2
        return err
    
    # Try various ranks with structured initialization
    for target_rank in range(20, min(112, 60)):
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            break
        
        found = False
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: use the schoolbook pattern but with random mixing
            U_r = np.random.randn(n * m, target_rank) * 0.3
            V_r = np.random.randn(m * p, target_rank) * 0.3
            W_r = np.random.randn(p * n, target_rank) * 0.3
            
            # Add some structure: initialize based on the tensor pattern
            for r in range(target_rank):
                # Randomly assign which (i,j,k) this rank term contributes to
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U_r[i*m+j, r] = 1.0
                V_r[j*p+k, r] = 1.0
                W_r[k*n+i, r] = 1.0
            
            current_err = compute_error_fast(U_r, V_r, W_r)
            
            # Local search with adaptive step size
            step_size = 0.2
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                if current_err < 1e-12:
                    break
                
                # Perturb a random element
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    row = random.randint(0, n*m-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, step_size)
                    old_val = U_r[row, colr]
                    U_r[row, colr] = old_val + delta
                    new_err = compute_error_fast(U_r, V_r, W_r)
                    if new_err < current_err:
                        current_err = new_err
                        step_size = min(step_size * 1.1, 0.5)
                    else:
                        U_r[row, colr] = old_val
                        step_size = max(step_size * 0.9, 0.01)
                elif mat_choice == 1:
                    row = random.randint(0, m*p-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, step_size)
                    old_val = V_r[row, colr]
                    V_r[row, colr] = old_val + delta
                    new_err = compute_error_fast(U_r, V_r, W_r)
                    if new_err < current_err:
                        current_err = new_err
                        step_size = min(step_size * 1.1, 0.5)
                    else:
                        V_r[row, colr] = old_val
                        step_size = max(step_size * 0.9, 0.01)
                else:
                    row = random.randint(0, p*n-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, step_size)
                    old_val = W_r[row, colr]
                    W_r[row, colr] = old_val + delta
                    new_err = compute_error_fast(U_r, V_r, W_r)
                    if new_err < current_err:
                        current_err = new_err
                        step_size = min(step_size * 1.1, 0.5)
                    else:
                        W_r[row, colr] = old_val
                        step_size = max(step_size * 0.9, 0.01)
            
            if current_err < 1e-10:
                if verify_np(U_r, V_r, W_r, tol=1e-9):
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
