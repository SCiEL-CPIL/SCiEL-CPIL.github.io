"""Matrix multiplication 2,2,8 - rank-28 exact tensor decomposition via 4x Strassen."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def _strassen_28():
    """Return rank-28 Strassen decomposition for M_{2,2,8}."""
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1},
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for a_idx, val in U_coeffs[s].items():
                U[a_idx][r] = val
            for b_local, val in V_coeffs[s].items():
                j = b_local // 2
                k_local = b_local % 2
                k = 2 * q + k_local
                V[j * p + k][r] = val
            for c_local, val in W_coeffs[s].items():
                i = c_local % 2
                k_local = c_local // 2
                k = 2 * q + k_local
                W[k * n + i][r] = val
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=2, p=8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T_{i,j,k} = sum_r U[i,r] * V[j,r] * W[k,r] should equal
    # the matrix multiplication tensor: T_{i,j,k} = 1 if i==j and k==i, else 0
    # Wait, let me think about this more carefully.
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i, j, k) where:
    #   i = row of A / row of C (0..n-1)
    #   j = col of A / row of B (0..m-1)  
    #   k = col of B / col of C (0..p-1)
    # T[i,j,k] = 1 if this term appears in the sum, i.e., always 1 for valid indices
    # Actually T[i,j,k] = 1 for all valid (i,j,k) since every term A[i,j]*B[j,k] contributes to C[i,k]
    
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the interface:
    # U: (n*m, rank) - indexed by (i,j) where i is row of A, j is col of A / row of B
    # V: (m*p, rank) - indexed by (j,k) where j is row of B, k is col of B
    # W: (p*n, rank) - indexed by (k,i) where k is col of B, i is row of C = row of A
    
    # So the reconstruction is: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should be 1 for all valid (i,j,k) since matrix multiplication sums over j.
    
    # Actually no. The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in range
    # because the product A*B has entry (i,k) = sum_j A[i,j]*B[j,k], so each term (i,j,k) contributes.
    
    # Let's verify:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for r in range(len(U[0])):
                    val += U_np[idx_u, r] * V_np[idx_v, r] * W_np[idx_w, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _search_rank_24(deadline):
    """
    Search for a rank-24 decomposition using the structure that A is shared across blocks.
    
    Key insight: U has only 4 rows (n*m = 4), so it's a 4x24 matrix.
    The constraint is that for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We can try to find a decomposition where the 4 blocks share some structure.
    Since U is only 4x24, we can enumerate possible U patterns with small integer entries.
    """
    import itertools
    import numpy as np
    
    n, m, p = 2, 2, 8
    
    # Try rank-24: U is 4x24, V is 16x24, W is 16x24
    # The key constraint: for each (i,j,k) in [0,2)x[0,2)x[0,8), 
    #   sum_r U[i*2+j, r] * V[j*8+k, r] * W[k*2+i, r] = 1
    
    # Strategy: Since U is small (4x24), try to find U with entries in {-1, 0, 1}
    # such that the remaining system for V and W is consistent.
    
    # For a given U, the constraint becomes:
    # For each (i,j,k): sum_r U[i*2+j, r] * (V[j*8+k, r] * W[k*2+i, r]) = 1
    # Let X_{j,k,r} = V[j*8+k, r] * W[k*2+i, r] ... but this depends on i.
    
    # Alternative approach: Fix U, then for each r where U has non-zero entries,
    # we need V and W to satisfy bilinear constraints.
    
    # Let's try a more structured search. Assume U has the form where each column
    # has at most 2 non-zero entries (like Strassen).
    
    # Actually, let's try a direct approach: search for U with entries in {-1, 0, 1}
    # and then solve for V and W.
    
    # The number of possible U columns with at most 2 non-zeros in {−1,0,1}^4 is:
    # 4C0 * 1 + 4C1 * 2 + 4C2 * 4 = 1 + 8 + 24 = 33 per column
    # For 24 columns, that's 33^24 which is way too many.
    
    # Better approach: Use the fact that the problem has 4-fold block structure.
    # Try U where columns are grouped: 6 columns per block, and the U pattern is the same
    # across blocks (since A is shared).
    
    # So U is 4x24 where columns 0-5 are the same pattern as 6-11, etc.
    # Let u0..u5 be the first 6 columns (each in {-1,0,1}^4 with at most 2 non-zeros).
    
    # For a given U pattern (6 columns repeated 4 times), we need to find V and W.
    # V is 16x24, W is 16x24.
    
    # The constraints: for each (i,j,k) and each block q (0..3):
    #   sum_{s=0}^{5} U[i*2+j, q*6+s] * V[j*8+(2q + (k%2)), q*6+s] * W[(2q+(k%2))*2+i, q*6+s] = 1
    
    # Wait, k ranges over 0..7, and block q handles k in {2q, 2q+1}.
    # Let k = 2q + k_local where k_local in {0,1}.
    
    # So for each (i, j, q, k_local):
    #   sum_{s=0}^{5} U[i*2+j, q*6+s] * V[j*8+2q+k_local, q*6+s] * W[(2q+k_local)*2+i, q*6+s] = 1
    
    # If U is the same across blocks (U[i*2+j, q*6+s] = U0[i*2+j, s]), then:
    #   sum_{s=0}^{5} U0[i*2+j, s] * V[j*8+2q+k_local, q*6+s] * W[(2q+k_local)*2+i, q*6+s] = 1
    
    # Let V_q[j, k_local, s] = V[j*8+2q+k_local, q*6+s] and W_q[k_local, i, s] = W[(2q+k_local)*2+i, q*6+s]
    # Then: sum_s U0[i*2+j, s] * V_q[j, k_local, s] * W_q[k_local, i, s] = 1
    
    # For each q, this is the same constraint. So we just need to find U0 (4x6), 
    # V_q (2x2x6 for each j, k_local), W_q (2x2x6 for each k_local, i) such that
    # the constraint holds. And V, W can be the same for all q!
    
    # So: find U0 (4x6), V (2x2x6), W (2x2x6) with entries in {-1, 0, 1, 1/2, ...}
    # such that for all i,j,k_local: sum_s U0[i*2+j, s] * V[j, k_local, s] * W[k_local, i, s] = 1
    
    # This is a system of 2*2*2 = 8 equations with 6*(4 + 4 + 4) = 72 unknowns.
    # Let's search for solutions with entries in {-1, 0, 1}.
    
    # Number of possible U0 columns: 33 (as before)
    # 33^6 = ~1.3 billion - too many to enumerate fully.
    
    # Let's try a smarter approach: fix U0 to have specific patterns and solve.
    
    # Pattern 1: U0 = Strassen-like but with 6 columns instead of 7.
    # Try: U0 = [[1,1,1,0,0,0],
    #              [0,1,0,1,0,0],
    #              [0,0,1,0,1,0],
    #              [1,0,0,1,0,1]]
    # (each column has 2 non-zeros)
    
    candidate_U0s = []
    # Generate all 4x6 matrices with entries in {-1,0,1} where each column has exactly 2 non-zeros
    # That's C(4,2) * 2^2 = 6 * 4 = 24 per column, 24^6 = 191 million - still too many.
    
    # Let's just try a few structured candidates.
    
    # Candidate: based on Strassen but merged
    candidates = [
        # Each is a list of 6 columns, each column is a tuple of 4 values
        [(1,0,0,1), (0,1,0,0), (0,0,1,0), (1,0,1,0), (0,1,0,1), (0,0,1,1)],
        [(1,1,0,0), (0,0,1,1), (1,0,1,0), (0,1,0,1), (1,0,0,1), (0,1,1,0)],
        [(1,0,1,0), (0,1,0,1), (1,1,0,0), (0,0,1,1), (1,0,0,1), (0,1,1,0)],
        [(1,1,0,0), (0,0,1,1), (1,0,0,1), (0,1,1,0), (1,1,0,0), (0,0,1,1)],
        [(1,0,1,0), (0,1,0,1), (1,0,0,1), (0,1,1,0), (1,1,0,0), (0,0,1,1)],
    ]
    
    for U0_cols in candidates:
        U0 = [[U0_cols[s][a] for s in range(6)] for a in range(4)]
        
        # For this U0, try to find V and W with entries in {-1, 0, 1}
        # V[j, k_local, s] for j in {0,1}, k_local in {0,1}, s in {0..5}
        # W[k_local, i, s] for k_local in {0,1}, i in {0,1}, s in {0..5}
        
        # Constraints: for all i in {0,1}, j in {0,1}, k_local in {0,1}:
        #   sum_s U0[i*2+j, s] * V[j, k_local, s] * W[k_local, i, s] = 1
        
        # Let's enumerate V and W. Each has 4*6 = 24 entries in {-1,0,1} -> 3^24 possibilities each.
        # Too many. Let's fix V and solve for W, or use a different approach.
        
        # Alternative: For each (j, k_local), the constraint is:
        #   For all i: sum_s U0[i*2+j, s] * V[j, k_local, s] * W[k_local, i, s] = 1
        # This is a linear system in W[k_local, :, s] for fixed (j, k_local).
        
        # Let's fix V and see if W can be solved.
        # V has 2*2*6 = 24 entries. Too many to enumerate.
        
        # Let's try a different approach: assume V and W have a specific structure.
        # For example, V[j, k_local, s] depends only on s and j, not on k_local.
        # And W[k_local, i, s] depends only on s and i, not on k_local.
        
        # Then the constraint becomes: for all i, j, k_local:
        #   sum_s U0[i*2+j, s] * V[j, s] * W[i, s] = 1
        # This must hold for all k_local, so it's just: for all i, j:
        #   sum_s U0[i*2+j, s] * V[j, s] * W[i, s] = 1
        
        # This is 4 equations with 12 unknowns (V[0,0..5], V[1,0..5], W[0,0..5], W[1,0..5] but that's 24... wait)
        # V[j, s] for j in {0,1}, s in {0..5} -> 12 unknowns
        # W[i, s] for i in {0,1}, s in {0..5} -> 12 unknowns
        # Total 24 unknowns, 4 equations. Underdetermined.
        
        # Let's just try small values. For each U0 candidate, try V and W with entries in {-1, 0, 1}
        # but with a restricted search.
        
        # Actually, let's just verify if the current Strassen works and return it.
        # The search for rank-24 is complex; let's add a timeout-guarded search.
        
        if time.time() > deadline:
            break
    
    # If no rank-24 found, return rank-28
    return None

def construct():
    deadline = _start_time + 140.0  # Safety margin before 150s kill
    
    # First, try to find a rank-24 decomposition
    result_24 = _search_rank_24(deadline)
    if result_24 is not None:
        return result_24
    
    # Fall back to rank-28 Strassen
    return _strassen_28()
# EVOLVE-BLOCK-END
