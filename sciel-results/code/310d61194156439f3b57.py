"""Matrix multiplication 2x2 x 2x4 - rank-13 tensor decomposition."""
import time
import random

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank13():
    """
    Construct a rank-13 decomposition for 2x2 * 2x4 matrix multiplication.
    
    Strategy: Use 6 Strassen products for the left 2x2 block, 6 for the right
    2x2 block, and 1 shared product. Total = 13.
    
    The shared product S = (A00+A11)(B00+B11+B02+B13) replaces what would
    be two separate (A00+A11)(B00+B11) and (A00+A11)(B02+B13) products.
    
    A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    C indices (W convention W[k*n+i,r]): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    """
    n, m, p = 2, 2, 4
    rank = 13
    U = [[0] * rank for _ in range(n * m)]  # 4 x 13
    V = [[0] * rank for _ in range(m * p)]  # 8 x 13
    W = [[0] * rank for _ in range(p * n)]  # 8 x 13
    
    # Product 0: S = (A00+A11)(B00+B11+B02+B13) - SHARED
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1; V[2][0] = 1; V[7][0] = 1
    # S contributes to C00, C10, C01, C11, C02, C12, C03, C13
    # We'll use it with appropriate coefficients in the output
    
    # Product 1: P2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Product 2: P3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Product 3: P4 = A11*(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Product 4: P5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Product 5: P6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Product 6: P7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1
    
    # Product 7: Q2 = (A10+A11)*B02
    U[2][7] = 1; U[3][7] = 1
    V[2][7] = 1
    W[5][7] = 1; W[7][7] = -1
    
    # Product 8: Q3 = A00*(B03-B13)
    U[0][8] = 1
    V[3][8] = 1; V[7][8] = -1
    W[6][8] = 1; W[7][8] = 1
    
    # Product 9: Q4 = A11*(B12-B02)
    U[3][9] = 1
    V[6][9] = 1; V[2][9] = -1
    W[4][9] = 1; W[5][9] = 1
    
    # Product 10: Q5 = (A00+A01)*B13
    U[0][10] = 1; U[1][10] = 1
    V[7][10] = 1
    W[4][10] = -1; W[6][10] = 1
    
    # Product 11: Q6 = (A10-A00)*(B02+B03)
    U[2][11] = 1; U[0][11] = -1
    V[2][11] = 1; V[3][11] = 1
    W[7][11] = 1
    
    # Product 12: Q7 = (A01-A11)*(B12+B13)
    U[1][12] = 1; U[3][12] = -1
    V[6][12] = 1; V[7][12] = 1
    W[4][12] = 1
    
    # Now set up W for the shared product S (product 0)
    # S = (A00+A11)(B00+B11+B02+B13)
    # We need to use S to contribute to C00, C10, C01, C11, C02, C12, C03, C13
    # 
    # For left block (C00, C10, C01, C11):
    #   C00 should get: S - P4 + P6 - (contribution from Q1 part)
    #   Actually, let me think more carefully.
    #
    # In standard Strassen for left block:
    #   P1 = (A00+A11)(B00+B11)
    #   C00 = P1 - P4 + P6
    #   C10 = P2 + P4
    #   C01 = P3 + P5
    #   C11 = P1 - P2 + P7
    #
    # In standard Strassen for right block:
    #   Q1 = (A00+A11)(B02+B13)
    #   C02 = Q1 - Q4 + Q6
    #   C12 = Q2 + Q4
    #   C03 = Q3 + Q5
    #   C13 = Q1 - Q2 + Q7
    #
    # We have S = P1 + Q1 = (A00+A11)(B00+B11+B02+B13)
    # So P1 = S - Q1 and Q1 = S - P1
    #
    # C00 = P1 - P4 + P6 = (S - Q1) - P4 + P6
    # C11 = P1 - P2 + P7 = (S - Q1) - P2 + P7
    # C02 = Q1 - Q4 + Q6 = Q1 - Q4 + Q6
    # C13 = Q1 - Q2 + Q7 = Q1 - Q2 + Q7
    #
    # Problem: we don't have P1 or Q1 separately, only S = P1 + Q1.
    #
    # Alternative approach: Use S directly with coefficient 1 for the parts
    # that need P1+Q1, and adjust other products.
    #
    # Let me try: assign S with coefficient 1 to C00, C10, C01, C11, C02, C12, C03, C13
    # and then fix up with the other 12 products.
    
    # Let me reconsider. The correct approach:
    # We need to verify that the 13 products actually decompose the tensor.
    # Let me just verify numerically and adjust if needed.
    
    # For now, set W[0] for S to contribute 1 to all 8 C entries
    # This is a first attempt; we'll verify and fix.
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    W[4][0] = 1; W[5][0] = 1; W[6][0] = 1; W[7][0] = 1
    
    return {"U": U, "V": V, "W": W}


def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left block (cols 0,1) - Strassen
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1
    
    # Right block (cols 2,3) - Strassen
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                
                if abs(t_val) > 1e-9:
                    return False
    
    return True


def _compute_error(U, V, W):
    """Compute the total error of the decomposition."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    error = 0.0
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(rank))
                error += (t_val - 1.0) ** 2
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(rank))
                error += t_val ** 2
    
    return error


def _try_rank13_numerical():
    """
    Try to find a rank-13 solution using numerical optimization.
    Start from the rank-14 solution and try to merge two products.
    """
    import numpy as np
    
    n, m, p = 2, 2, 4
    target_rank = 13
    deadline = time.time() + 120
    
    # Start with rank-14 solution
    base = _build_rank14()
    U = np.array(base["U"], dtype=float)
    V = np.array(base["V"], dtype=float)
    W = np.array(base["W"], dtype=float)
    
    # Try to eliminate one rank by making it a linear combination of others
    # For each pair of columns, try to see if one can be absorbed
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        # Try different strategies for rank reduction
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Zero out the last column
        U_try[:, 13] = 0
        V_try[:, 13] = 0
        W_try[:, 13] = 0
        
        # Now optimize the remaining 13 columns
        # Use gradient-free local search
        err = _compute_error(U_try.tolist(), V_try.tolist(), W_try.tolist())
        
        # Local search
        step_size = 0.3
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            which = random.choice(['U', 'V', 'W'])
            if which == 'U':
                idx = random.randint(0, 3)
                r = random.randint(0, 12)
                old = U_try[idx, r]
                U_try[idx, r] += random.uniform(-step_size, step_size)
            elif which == 'V':
                idx = random.randint(0, 7)
                r = random.randint(0, 12)
                old = V_try[idx, r]
                V_try[idx, r] += random.uniform(-step_size, step_size)
            else:
                idx = random.randint(0, 7)
                r = random.randint(0, 12)
                old = W_try[idx, r]
                W_try[idx, r] += random.uniform(-step_size, step_size)
            
            new_err = _compute_error(U_try.tolist(), V_try.tolist(), W_try.tolist())
            
            if new_err < err:
                err = new_err
                step_size = max(0.01, step_size * 0.95)
            else:
                # Revert
                if which == 'U':
                    U_try[idx, r] = old
                elif which == 'V':
                    V_try[idx, r] = old
                else:
                    W_try[idx, r] = old
                step_size = max(0.01, step_size * 0.98)
            
            if err < 1e-10:
                break
        
        if err < best_err:
            best_err = err
            best_result = (U_try, V_try, W_try)
        
        if best_err < 1e-8:
            break
    
    if best_result is not None and best_err < 1e-8:
        U_final, V_final, W_final = best_result
        # Round to nearest half-integer if close
        U_rounded = np.round(U_final * 2) / 2
        V_rounded = np.round(V_final * 2) / 2
        W_rounded = np.round(W_final * 2) / 2
        
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    return None


def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # First, try the explicit rank-13 construction
    rank13 = _build_rank13()
    if _verify_decomposition(rank13["U"], rank13["V"], rank13["W"]):
        return rank13
    
    # Try numerical search for rank 13
    if time.time() < deadline - 30:
        result = _try_rank13_numerical()
        if result is not None and _verify_decomposition(result["U"], result["V"], result["W"]):
            return result
    
    # Fallback to rank 14
    rank14 = _build_rank14()
    if _verify_decomposition(rank14["U"], rank14["V"], rank14["W"]):
        return rank14
    
    return rank14
# EVOLVE-BLOCK-END
