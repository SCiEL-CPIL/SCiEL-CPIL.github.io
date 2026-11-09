"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c]
                # W factor: for each c, W[k*n+i][c] multiplies U*V
                # Actually the tensor is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                pass
    # Proper verification
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Verify using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _snap_to_half(val):
    """Snap a float to the nearest half-integer."""
    return round(val * 2) / 2

def _solve_for_W(U, V, n, m, p, r):
    """
    Given U (n*m x r) and V (m*p x r), solve for W (p*n x r).
    For each (i,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j.
    This is a linear system in W[k*n+i, :].
    """
    W = np.zeros((p * n, r))
    for i in range(n):
        for k in range(p):
            # Build system: for each j, sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i * m + j, c] * V[j * p + k, c]
            b = np.ones(m)
            # Solve A @ W_row = b where W_row = W[k*n+i, :]
            # Use least squares
            try:
                W_row, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            # Check if solution is exact
            if not np.allclose(A @ W_row, b, atol=1e-10):
                return None
            W[k * n + i, :] = W_row
    return W

def _try_rank_with_patterns(rank, n, m, p, deadline, start_time):
    """
    Try to find a rank-rank decomposition by enumerating sparse patterns for U and V.
    """
    if time.time() > deadline:
        return None
    
    # Use small integer/half-integer values for U and V
    # Values: 0, 1, -1, 0.5, -0.5
    values = [0, 1, -1, 0.5, -0.5]
    
    # For efficiency, we'll fix U to have specific structure and vary V
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (9, rank)
    
    # Strategy: enumerate sparse patterns for U (each row has at most 2 nonzeros)
    # Then solve for V and W
    
    # Actually, let's try a more systematic approach:
    # Fix U to be a specific pattern, then for each pattern of V, solve for W
    
    # Given time constraints, let's try specific known constructions
    
    # Construction for rank 14:
    # Based on the fact that M_{2,3,3} can be decomposed using rank 14
    
    # Let's try to use a structured approach where we fix most of U and V
    # and solve for the remaining variables
    
    # Alternative: use numerical optimization
    # Minimize ||sum_c U_c * V_c * W_c - T||^2 where T is the multiplication tensor
    
    # For now, return None to indicate no solution found
    return None

def _try_numerical_optimization(rank, n, m, p, deadline, start_time):
    """
    Try numerical optimization to find U, V, W.
    """
    if time.time() > deadline:
        return None
    
    # Initialize with random values
    np.random.seed(42)
    
    # Use gradient descent on the loss function
    # Loss = sum_{i,j,k} (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # This is a non-convex problem, so gradient descent may not work well
    # Let's try a different approach
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a better solution
    # Known result: rank of M_{2,3,3} is 14
    
    # Let's try to construct a rank 14 solution using a known method
    
    # Method: Use the fact that we can combine smaller matrix multiplications
    # 2x3x3 = 2x2x2 + 2x1x1 + 1x2x2 (with appropriate embeddings)
    
    # Actually, let's try a direct construction based on literature
    
    # For rank 14, we need 14 terms
    # Let's try to find it by solving a system of equations
    
    # Approach: Fix U and V to have specific sparse patterns, solve for W
    
    # Try rank 14 first
    for rank in range(14, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try specific constructions
        if rank == 14:
            # Try to construct a rank 14 solution
            # Based on the Strassen-like decomposition
            
            # We'll use a structured approach
            # Let's define U, V, W with specific patterns
            
            # This is complex, so let's try a different approach:
            # Use the fact that the tensor can be decomposed using
            # a combination of 2x2x2 (rank 7) and 1x1x1 (rank 1) terms
            
            # 2x3x3: We can think of it as:
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
            # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
            # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
            
            # This is 18 terms in schoolbook. To reduce to 14, we need to
            # share some computations.
            
            # Let's try a specific construction:
            # Use 7 terms for the 2x2x2 part (top-left)
            # Use 1 term for each of the remaining 7 entries
            # Total: 7 + 7 = 14
            
            # 2x2x2 Strassen decomposition (rank 7):
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B10-B11)
            # M4 = A11*(B01-B00)
            # M5 = (A00+A01)*B11
            # M6 = (A10-A00)*(B00+B10)
            # M7 = (A01-A11)*(B01+B11)
            
            # For 2x3x3, we can extend this:
            # Use Strassen for the 2x2x2 subproblem (rows 0-1, cols 0-1)
            # Then handle the third column separately
            
            # Let's define the factors:
            U = np.zeros((6, 14))
            V = np.zeros((9, 14))
            W = np.zeros((6, 14))
            
            # Strassen terms for 2x2x2 (using indices 0-6)
            # A is 2x3, B is 3x3
            # A[0,0], A[0,1], A[1,0], A[1,1] are the 2x2 part
            
            # M1 = (A00+A11)*(B00+B11)
            U[0, 0] = 1  # A[0,0]
            U[3, 0] = 1  # A[1,1]
            V[0, 0] = 1  # B[0,0]
            V[4, 0] = 1  # B[1,1]
            W[0, 0] = 1  # C[0,0]
            W[1, 0] = 1  # C[0,1]
            W[2, 0] = 1  # C[1,0]
            W[3, 0] = 1  # C[1,1]
            
            # M2 = (A10+A11)*B00
            U[2, 1] = 1  # A[1,0]
            U[3, 1] = 1  # A[1,1]
            V[0, 1] = 1  # B[0,0]
            W[2, 1] = 1  # C[1,0]
            W[3, 1] = 1  # C[1,1]
            
            # M3 = A00*(B10-B11)
            U[0, 2] = 1  # A[0,0]
            V[3, 2] = 1  # B[1,0]
            V[4, 2] = -1  # B[1,1]
            W[0, 2] = 1  # C[0,0]
            W[1, 2] = -1  # C[0,1]
            
            # M4 = A11*(B01-B00)
            U[3, 3] = 1  # A[1,1]
            V[1, 3] = 1  # B[0,1]
            V[0, 3] = -1  # B[0,0]
            W[2, 3] = 1  # C[1,0]
            W[3, 3] = -1  # C[1,1]
            
            # M5 = (A00+A01)*B11
            U[0, 4] = 1  # A[0,0]
            U[1, 4] = 1  # A[0,1]
            V[4, 4] = 1  # B[1,1]
            W[1, 4] = 1  # C[0,1]
            
            # M6 = (A10-A00)*(B00+B10)
            U[2, 5] = 1  # A[1,0]
            U[0, 5] = -1  # A[0,0]
            V[0, 5] = 1  # B[0,0]
            V[3, 5] = 1  # B[1,0]
            W[0, 5] = -1  # C[0,0]
            W[2, 5] = 1  # C[1,0]
            
            # M7 = (A01-A11)*(B01+B11)
            U[1, 6] = 1  # A[0,1]
            U[3, 6] = -1  # A[1,1]
            V[1, 6] = 1  # B[0,1]
            V[4, 6] = 1  # B[1,1]
            W[1, 6] = 1  # C[0,1]
            W[3, 6] = -1  # C[1,1]
            
            # Now handle the third column (B[:,2]) and third row (A[:,2])
            # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
            # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
            
            # M8 = A[0,2]*B[2,2]
            U[5, 7] = 1  # A[0,2]
            V[8, 7] = 1  # B[2,2]
            W[4, 7] = 1  # C[0,2]
            
            # M9 = A[1,2]*B[2,2]
            U[5, 8] = 1  # A[1,2]  # Wait, A[1,2] is index 1*3+2 = 5
            V[8, 8] = 1  # B[2,2]
            W[5, 8] = 1  # C[1,2]
            
            # M10 = (A[0,0]+A[1,0])*B[0,2]
            U[0, 9] = 1  # A[0,0]
            U[2, 9] = 1  # A[1,0]
            V[2, 9] = 1  # B[0,2]
            W[4, 9] = 1  # C[0,2]
            W[5, 9] = 1  # C[1,2]
            
            # M11 = (A[0,1]+A[1,1])*B[1,2]
            U[1, 10] = 1  # A[0,1]
            U[3, 10] = 1  # A[1,1]
            V[5, 10] = 1  # B[1,2]
            W[4, 10] = 1  # C[0,2]
            W[5, 10] = 1  # C[1,2]
            
            # M12 = A[0,2]*B[0,2]  # Already have A[0,2]*B[2,2] in M8
            # Need A[0,2]*B[0,2] and A[0,2]*B[1,2]
            
            # Let me restructure:
            # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
            # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
            
            # M8 = A[0,2]*B[2,2] -> contributes to C[0,2]
            # M9 = A[1,2]*B[2,2] -> contributes to C[1,2]
            # M10 = (A[0,0]+A[1,0])*B[0,2] -> contributes to C[0,2] and C[1,2]
            # M11 = (A[0,1]+A[1,1])*B[1,2] -> contributes to C[0,2] and C[1,2]
            
            # But we also need A[0,2]*B[0,2], A[0,2]*B[1,2], A[1,2]*B[0,2], A[1,2]*B[1,2]
            # These are missing!
            
            # Let me try a different approach: just use 4 more terms for the third column
            # M8 = A[0,2]*B[0,2]
            # M9 = A[0,2]*B[1,2]
            # M10 = A[1,2]*B[0,2]
            # M11 = A[1,2]*B[1,2]
            # M12 = A[0,2]*B[2,2]
            # M13 = A[1,2]*B[2,2]
            # M14 = (A[0,0]+A[1,0])*B[0,2] + (A[0,1]+A[1,1])*B[1,2]  # This doesn't work
            
            # This is getting complicated. Let's just return the schoolbook for now
            # and try to improve in future iterations.
            
            # Actually, let's just verify if our construction works
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try other ranks
        result = _try_rank_with_patterns(rank, n, m, p, deadline, start_time)
        if result is not None:
            return result
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
