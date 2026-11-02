"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # First, return the known rank-14 solution immediately as a fallback
    rank14_result = _build_rank14()
    
    # Try to find a rank-13 decomposition using randomized search over small integers
    # The key insight: if rank 13 exists, the coefficients are likely small integers
    # We can search over coefficient values in {-2, -1, 0, 1, 2} for each entry
    
    # For rank r, we need to find U (4xr), V (8xr), W (8xr) such that
    # for all i,j,k: sum_t U[i,t]*V[j,t]*W[k,t] = 1 if (i,j,k) is a valid multiplication entry, else 0
    
    # The valid entries for 2x2 * 2x4:
    # A is 2x2: indices (i,j) where i in {0,1}, j in {0,1}
    # B is 2x4: indices (j,k) where j in {0,1}, k in {0,1,2,3}
    # C is 2x4: indices (i,k) where i in {0,1}, k in {0,1,2,3}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In tensor notation: T[i,j,k] = 1 if the entry exists, else 0
    # But actually the tensor is: T[i,j,k] = delta_{i,j,k} where the multiplication is defined
    # More precisely: the bilinear form gives C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) triples
    
    # Let's define the target tensor
    n, m, p = DIMENSIONS  # 2, 2, 4
    # U has shape (n*m, r) = (4, r)
    # V has shape (m*p, r) = (8, r)  
    # W has shape (p*n, r) = (8, r)
    
    # The tensor T[i, j, k] where i is row of A (0,1), j is col of A / row of B (0,1), k is col of B (0,1,2,3)
    # Actually let me re-read the convention:
    # U: (n*m, rank) - n=2, m=2, so 4 entries
    # V: (m*p, rank) - m=2, p=4, so 8 entries
    # W: (p*n, rank) - p=4, n=2, so 8 entries, indexed as W[k*n+i]
    
    # The decomposition: C[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, I need to be more careful about the indexing.
    
    # Standard matrix multiplication tensor:
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U has shape (n*m, rank): U[i*m+j, r]
    # V has shape (m*p, rank): V[j*p+k, r]
    # W has shape (p*n, rank): W[k*n+i, r]
    
    # So: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this sums over r only, and j is implicit? No, that's not right.
    
    # Let me reconsider. The standard tensor decomposition for matrix multiplication:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r)_{i,j} = U[i*m+j, r], (V_r)_{j,k} = V[j*p+k, r], (W_r)_{i,k} = W[k*n+i, r]
    # And T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    
    # So we need: for all i in {0,1}, j in {0,1}, k in {0,1,2,3}:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 2*2*4 = 16 equations.
    
    # For rank 13, we have 13*(4+8+8) = 13*20 = 260 unknowns.
    # This is heavily overdetermined but the equations are nonlinear (trilinear).
    
    # Strategy: Randomized search with local refinement
    # 1. Start with random integer coefficients in {-1, 0, 1}
    # 2. Compute the residual for each of the 16 equations
    # 3. Try to fix one equation at a time by adjusting one coefficient
    
    # Actually, let's try a simpler approach:
    # Use the rank-14 solution and try to eliminate one rank by merging
    # Or try random rank-13 solutions and check validity
    
    # Let's try random search with small integer coefficients
    best_score = 14  # Current best rank
    
    # Define the 16 target equations
    # Equation (i,j,k): sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    
    def check_solution(U, V, W, r):
        """Check if the solution is valid. Returns True if all 16 equations are satisfied."""
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    s = 0
                    for t in range(r):
                        s += U[i*2+j, t] * V[j*4+k, t] * W[k*2+i, t]
                    if s != 1:
                        return False
        return True
    
    def compute_residuals(U, V, W, r):
        """Compute residuals for all 16 equations."""
        residuals = {}
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    s = 0
                    for t in range(r):
                        s += U[i*2+j, t] * V[j*4+k, t] * W[k*2+i, t]
                    residuals[(i,j,k)] = s - 1
        return residuals
    
    # Try random rank-13 search
    target_rank = 13
    max_iterations = 50000
    coeff_range = [-2, -1, 0, 1, 2]
    
    found = False
    iterations_done = 0
    
    for iteration in range(max_iterations):
        iterations_done = iteration + 1
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization with small integers
        U = np.array([[random.choice(coeff_range) for _ in range(target_rank)] for _ in range(4)], dtype=int)
        V = np.array([[random.choice(coeff_range) for _ in range(target_rank)] for _ in range(8)], dtype=int)
        W = np.array([[random.choice(coeff_range) for _ in range(target_rank)] for _ in range(8)], dtype=int)
        
        # Quick check: compute all 16 sums
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    s = 0
                    for t in range(target_rank):
                        s += U[i*2+j, t] * V[j*4+k, t] * W[k*2+i, t]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            found = True
            break
    
    if found:
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no rank-13 found, return the rank-14 solution
    return rank14_result

def _build_rank14():
    """Build the known rank-14 decomposition using Strassen's algorithm."""
    n, m, p = DIMENSIONS
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0: a00, 1: a01, 2: a10, 3: a11
    # Forms:
    # 0: a00 + a11
    # 1: a10 + a11
    # 2: a00
    # 3: a11
    # 4: a00 + a01
    # 5: a10 - a00
    # 6: a01 - a11
    
    # Set U for first 7 terms (B1 part)
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # Set U for next 7 terms (B2 part) - same forms
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    # Strassen linear forms for B (2x2 blocks)
    # B1 indices in V: 0: b00, 1: b01, 4: b10, 5: b11
    # B2 indices in V: 2: b02, 3: b03, 6: b12, 7: b13
    
    # Set V for first 7 terms (B1)
    # 0: b00 + b11
    V[0][0] = 1; V[5][0] = 1
    # 1: b00
    V[0][1] = 1
    # 2: b01 - b11
    V[1][2] = 1; V[5][2] = -1
    # 3: b10 - b00
    V[4][3] = 1; V[0][3] = -1
    # 4: b11
    V[5][4] = 1
    # 5: b00 + b01
    V[0][5] = 1; V[1][5] = 1
    # 6: b10 + b11
    V[4][6] = 1; V[5][6] = 1
    
    # Set V for next 7 terms (B2)
    # 7: b02 + b13
    V[2][7] = 1; V[7][7] = 1
    # 8: b02
    V[2][8] = 1
    # 9: b03 - b13
    V[3][9] = 1; V[7][9] = -1
    # 10: b12 - b02
    V[6][10] = 1; V[2][10] = -1
    # 11: b13
    V[7][11] = 1
    # 12: b02 + b03
    V[2][12] = 1; V[3][12] = 1
    # 13: b12 + b13
    V[6][13] = 1; V[7][13] = 1
    
    # Output coefficients W
    # W indices: k*n + i
    # C1 (cols 0,1):
    # C00 (W[0]): M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C10 (W[1]): M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C01 (W[2]): M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C11 (W[3]): M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # C2 (cols 2,3):
    # C02 (W[4]): M'1 + M'4 - M'5 + M'7
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    # C12 (W[5]): M'2 + M'4
    W[5][8] = 1; W[5][10] = 1
    # C03 (W[6]): M'3 + M'5
    W[6][9] = 1; W[6][11] = 1
    # C13 (W[7]): M'1 - M'2 + M'3 + M'6
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
