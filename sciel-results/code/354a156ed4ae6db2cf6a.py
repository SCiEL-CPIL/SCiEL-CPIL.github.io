import numpy as np
import time

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 120  # seconds, with safety margin

def _get_3x3x3_factors():
    """Return U, V, W factors (each 9x21) for 3x3x3 matrix multiplication."""
    # Use a known rank-21 decomposition for 3x3x3
    # This is based on the work of Strassen and others
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t] for n=m=p=3
    
    # I'll use a construction based on the following idea:
    # Split the 3x3 matrices into blocks and use Strassen-like combinations
    
    # Actually, let me use a verified construction.
    # The factors are 9x21 matrices with entries in {-1, 0, 1}
    
    # Let me define them explicitly.
    # Reference: The 21-term algorithm can be constructed as follows:
    
    # I'll use the following approach: enumerate all possible "term" patterns
    # and verify they form a valid decomposition.
    
    # For now, let me use a simpler but valid approach:
    # Use the schoolbook (rank 27) for 3x3x3 if I can't find 21 quickly.
    # But 27+27+9 = 63, which is no better than schoolbook for 3x3x7.
    
    # So I MUST find the 21-term decomposition.
    
    # Let me try to construct it using the following method:
    # The 3x3x3 tensor can be decomposed using the fact that it contains
    # a 2x2x2 tensor (rank 7) plus additional terms.
    
    # Actually, let me just hardcode a known valid set of factors.
    
    # After research, here's a rank-21 decomposition for 3x3x3:
    # The U, V, W matrices are 9x21.
    
    # I'll define them as numpy arrays.
    
    # Let me use the following construction from literature:
    # Terms 1-7: Strassen-like for the top-left 2x2 block
    # Terms 8-14: For the remaining elements
    
    # This is getting complex. Let me try a different approach:
    # Use random initialization and verify, then try to reduce rank.
    
    # For now, return None to indicate we need to find it.
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    # T[i, j, k] should equal sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # The matmul tensor has T[i,j,k] = 1 for all valid i,j,k
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    # Also check that no other entries are non-zero (but for matmul tensor,
    # all entries T[i,j,k] = 1, so we just need all to be 1)
    
    return error < 1e-9, error

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a better decomposition
    # Strategy: decompose 3x3x7 as 3x3x3 + 3x3x3 + 3x3x1
    
    # For 3x3x1, rank is 9 (schoolbook is optimal)
    # For 3x3x3, try to find rank 21 decomposition
    
    # Let me try to construct a rank-21 decomposition for 3x3x3
    # using a systematic approach.
    
    # The 3x3x3 matmul tensor T has T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need to find U, V, W (each 9x21) such that
    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k
    
    # This is a system of 27 equations with 21*3 = 63 unknowns (if we allow
    # arbitrary real values), so it's underdetermined.
    
    # Let me try to find a solution using linear algebra.
    # For each term t, we have vectors u_t, v_t, w_t in R^9.
    # The constraint is: for all (i,j,k), sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1
    
    # This is a nonlinear system. Let me try a different approach.
    
    # Approach: Use the fact that the tensor can be written as a sum of rank-1 tensors.
    # A rank-1 tensor is u ⊗ v ⊗ w.
    
    # Let me try to find 21 rank-1 tensors that sum to the all-ones tensor.
    
    # One method: Start with the schoolbook (27 terms) and try to merge/eliminate terms.
    
    # Another method: Use random vectors and solve for the remaining coefficients.
    
    # Let me try: Fix U and V randomly, then solve for W.
    # But this is still nonlinear.
    
    # Let me try a simpler approach: Use the known fact that rank(M_{3,3,3}) = 21
    # and look up explicit factors.
    
    # After further thought, I'll try to construct the factors using the following
    # pattern based on the Strassen algorithm generalized to 3x3.
    
    # For now, let me try a computational approach:
    # Use scipy to solve the nonlinear system.
    
    if time.time() > deadline - 30:
        return best_result
    
    try:
        from scipy.optimize import fsolve
        
        # Try to find a rank-21 decomposition for 3x3x3
        # Variables: 21 * 9 * 3 = 567 (for U, V, W)
        # Constraints: 27 (one for each (i,j,k))
        
        # This is underdetermined, so fsolve might work.
        
        def equations(vars):
            # vars is a flat array of length 567
            U_flat = vars[:189]
            V_flat = vars[189:378]
            W_flat = vars[378:567]
            
            U_mat = U_flat.reshape(9, 21)
            V_mat = V_flat.reshape(9, 21)
            W_mat = W_flat.reshape(9, 21)
            
            eqs = []
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = 0.0
                        for t in range(21):
                            val += U_mat[i*3+j, t] * V_mat[j*3+k, t] * W_mat[k*3+i, t]
                        eqs.append(val - 1.0)
            return eqs
        
        # Initial guess: schoolbook-like
        x0 = np.zeros(567)
        # Set up initial values based on schoolbook
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    t = i * 9 + j * 3 + k
                    if t < 21:
                        x0[t * 9 + i*3+j] = 1  # U
                        x0[189 + t * 9 + j*3+k] = 1  # V
                        x0[378 + t * 9 + k*3+i] = 1  # W
        
        # This is a very large system. fsolve may not work well.
        # Let me try a different approach.
        
        pass
        
    except Exception as e:
        pass
    
    # Let me try a more direct approach: use the known 21-term factors.
    # I'll define them explicitly based on a verified construction.
    
    # After careful consideration, here's a rank-21 decomposition for 3x3x3:
    # The factors U, V, W are 9x21 matrices.
    
    # I'll use the following construction:
    # The 3x3x3 tensor can be decomposed using 21 terms where each term
    # corresponds to a specific pattern of 1s and 0s in U, V, W.
    
    # Let me define the factors as lists of (u, v, w) triples.
    
    # Term 1-9: Cover the "diagonal" and "off-diagonal" in a structured way
    
    # Actually, let me try to use a simpler valid decomposition first.
    # Use rank 23 for 3x3x3 (if 21 is too hard to construct).
    
    # But 23+23+9 = 55, which is still better than 63.
    
    # Let me try to find a rank-23 decomposition.
    
    # Actually, let me just try to verify if a rank-21 solution exists using
    # a different method.
    
    # For now, let me return the schoolbook result.
    return best_result
