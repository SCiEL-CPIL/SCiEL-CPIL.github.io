"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank < 35 decomposition
    # Start with Strassen-based rank 35, then try to reduce
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # First, construct the known rank-35 solution
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        col_base = t * 7
        # Term 1: P1 = (a+d)(e+h)
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        # Term 2: P2 = (c+d)e
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        # Term 3: P3 = a(f-h)
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        # Term 4: P4 = d(g-e)
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        # Term 5: P5 = (a+b)h
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        # Term 6: P6 = (c-a)(e+f)
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        # Term 7: P7 = (b-d)(g+h)
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Verify the decomposition is correct
    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        r = U_arr.shape[1]
        # The tensor is sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
        # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
        # U index: i*m+j, V index: j*p+k, W index: k*n+i
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k) structure
        # Actually the tensor T[i,j,k] = 1 if this is the matrix mult tensor
        # T[i,j,k] means: output C[i,k] gets contribution A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the right sense)
        # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # for the matrix multiplication tensor, which is just 1 for each (i,j,k) in the support.
        # Wait, the tensor for matrix mult A(2x2) * B(2x10) -> C(2x10):
        # T_{i,j,k} = 1 for all i in {0,1}, j in {0,1}, k in {0,...,9}
        # because C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is all-ones on its support.
        # Check: for each (i,j,k), sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(r):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to trivial rank-40
        rank = 40
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try to reduce rank by checking if any column can be removed or if there's a rank-34 solution
    # Strategy: try to find a rank-34 decomposition by random perturbation + projection
    # Or: try specific structural improvements
    
    # Try: can we combine two Strassen blocks to save a term?
    # For 2x2 * 2x4, trivial rank is 16, Strassen gives 14. Can we do 13?
    # Let's try a computational search for small rank reductions
    
    # Approach: try to find if the rank-35 solution has a linear dependency that allows reduction
    # Build the tensor and check its rank via SVD-like approach (but tensor rank is hard)
    
    # Alternative: try specific known algorithms or random search for rank 34
    # Use random initialization and iterative refinement
    
    if time.time() < deadline - 30:
        # Try random search for rank 34
        np.random.seed(42)
        found = False
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            # Random rank-34 decomposition
            r = 34
            U_rand = np.random.randn(n*m, r)
            V_rand = np.random.randn(m*p, r)
            W_rand = np.random.randn(p*n, r)
            
            # Check if this can be adjusted to match the tensor
            # The tensor T is all-ones on support {i,j,k}
            # We need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
            # This is a system of 2*2*10 = 40 equations with 34*3 = 102 unknowns (but nonlinear)
            # Use iterative method: fix two, solve for third
            
            # Simpler: try to see if we can express one Strassen term as combination of others
            # by checking linear dependencies in the rank-35 solution
            
            # Check if any column of the "flattened" tensor is in the span of others
            # This is complex; skip for now and try a different approach
            
            # Try: modify the Strassen decomposition to share one term
            # Specifically, check if P7 from block 0 can be combined with something from block 1
            pass
        
        # Try a more structured approach: use the fact that we have 10 columns
        # What if we use a 3+3+4 split?
        # 2x2 * 2x3: need to find min rank
        # 2x2 * 2x4: need to find min rank
        
        # For 2x2 * 2x3: trivial is 12. Let's see if we can do 11.
        # For 2x2 * 2x4: trivial is 16. Strassen gives 14. Can we do 13?
        
        # Let's try to verify if rank 13 works for 2x2*2x4
        # by attempting a random search
        if time.time() < deadline - 20:
            # Try to find rank-13 for 2x2*2x4
            # and rank-11 for 2x2*2x3
            # Then combine: 13 + 11 + 11 = 35 (no improvement)
            # or 13 + 13 + 13 = 39 (worse)
            # or 11 + 11 + 13 = 35 (same)
            # We need: sum of ranks for partition < 35
            # e.g., 11+11+11=33 if 2x2*2x3 can be done in 11
            # or 12+12+11=35 (no improvement)
            
            # Let's check if 2x2*2x3 can be done in rank 11
            # The tensor has 2*2*3=12 entries, all = 1
            # Rank 11 would mean one linear dependency
            # Try random search
            found_11 = False
            for trial in range(500):
                if time.time() > deadline - 20:
                    break
                r_test = 11
                U_t = np.random.randn(4, r_test)
                V_t = np.random.randn(6, r_test)  # 2*3=6
                W_t = np.random.randn(6, r_test)  # 3*2=6
                
                # Check: sum_r U[i*2+j,r]*V[j*3+k,r]*W[k*2+i,r] == 1 for i in 0,1; j in 0,1; k in 0,1,2
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            s = np.sum(U_t[i*2+j, :] * V_t[j*3+k, :] * W_t[k*2+i, :])
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    found_11 = True
                    break
            
            if found_11:
                # Great! Use 3 blocks of 2x2*2x3 with rank 11 each = 33 total
                # But wait, 10 columns = 3+3+4, not 3+3+3
                # 3+3+4: need rank for 2x2*2x3 (11) and 2x2*2x4 (14 or 13?)
                # 11+11+14 = 36 (worse) or 11+11+13=35 (same)
                # Need 2x2*2x4 in rank 12: 11+11+12=34!
                # Let's check if 2x2*2x4 can be done in rank 12
                pass
    
    # For now, return the best found (rank 35)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
