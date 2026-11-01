"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Trivial rank-50 construction as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][column] = 1
                V_trivial[j * p + k][column] = 1
                W_trivial[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    
    # ============================================================
    # Strategy: Use the block structure of 2x5x5 multiplication.
    # C = A * B where A is 2x5, B is 5x5.
    # Partition B into 5 column-blocks of width 1: B = [b1, b2, b3, b4, b5]
    # Each bi is a 5x1 column.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Alternative: Think of this as computing 2 outputs, each being
    # a dot product of a 5-dim row of A with a 5-dim column of B.
    # 
    # Key insight: We can use the rank of 5x5x5 matrix multiplication
    # (known to be 19) as a building block. For each of the 2 rows of A,
    # we compute a 5-dim vector (the result of that row times B).
    # 
    # Actually, let's think differently. The tensor T_{2,5,5} has
    # T[i,j,k] = 1 for all valid (i,j,k). The rank is at most 50.
    # 
    # Better approach: Use Strassen-like decomposition on sub-blocks.
    # Partition A into 2x2 blocks (but A is 2x5, so we can't do 2x2 nicely).
    # 
    # Let's try a different approach: use the fact that for fixed i,
    # the row A[i,:] dotted with each column of B gives C[i,:].
    # For i=0: C[0,k] = sum_j A[0,j]*B[j,k]
    # For i=1: C[1,k] = sum_j A[1,j]*B[j,k]
    # 
    # We can combine these. Let's define:
    # For each pair (j,k), the contribution to both outputs is:
    #   C[0,k] += A[0,j]*B[j,k]
    #   C[1,k] += A[1,j]*B[j,k]
    # 
    # So for each (j,k) pair, we have a rank-1 contribution to both outputs.
    # That's 25 pairs, each contributing to 2 outputs = 50 total.
    # 
    # To reduce rank, we need to find linear combinations.
    # 
    # Let's use a constructive approach with Strassen on 2x2 sub-blocks.
    # Partition the 5x5 B matrix and 2x5 A matrix into blocks.
    # 
    # Actually, the simplest constructive improvement: use the known
    # rank-19 decomposition of 5x5x5 matrix multiplication.
    # 
    # For 5x5x5, there exists a rank-19 decomposition (or close to it).
    # We can use this as a subroutine. For each row i of A, we compute
    # v_i = A[i,:] * B (a 5-dim vector). This is a 5x1x5 matrix mult.
    # The rank of 5x1x5 is at most 5 (trivially).
    # 
    # Total rank: 2 * 5 = 10? No, that's not how it works.
    # 
    # Let me think about this more carefully using the tensor structure.
    # The multiplication tensor T_{2,5,5} has entries T[i,j,k] = delta_{ik} * 1
    # Wait no. For matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,2), j in [0,5), k in [0,5).
    # 
    # The rank of T_{2,5,5} is at most 50 (trivial).
    # 
    # Let's try to find a low-rank decomposition using linear algebra.
    # We want to find rank r such that:
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # 
    # For small r, this is a system of polynomial equations.
    # 
    # Let's try a constructive approach: use the fact that we can
    # factor through intermediate variables.
    # 
    # Approach: Use a rank-25 decomposition first (improvement over 50),
    # then try to reduce further.
    # 
    # Actually, let's use the following constructive decomposition:
    # For each j (input index), define a "column" of the tensor.
    # T[i,j,k] = 1 for all i,k.
    # 
    # We can write: T[i,j,k] = sum_{t=1}^{m} delta_{j,t} * 1
    # But that's just the trivial decomposition.
    # 
    # Let's use a different approach: exploit the 2-dimensional output.
    # 
    # Key idea: Use Strassen's algorithm on 2x2x2 blocks within the larger structure.
    # 
    # Partition the 5x5 B matrix into blocks. Since 5 = 2+2+1, we can
    # partition B into 2x2, 2x2, 2x1, 1x2, 1x2, 1x1 blocks.
    # 
    # This is getting complex. Let's use a numerical search approach
    # but with a smarter initialization and verification.
    # ============================================================
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition with integer/half-integer entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for t in range(rank):
                        actual += U[i*m + j][t] * V[j*p + k][t] * W[k*n + i][t]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def try_als(r, n_restarts=10, max_iter=300):
        """Try to find a rank-r decomposition using ALS."""
        nonlocal best_rank, best_U, best_W, best_V
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            np.random.seed(restart * 7919 + r * 13 + 42)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            for iteration in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Update U
                for idx in range(n*m):
                    i = idx // m
                    j = idx % m
                    M = np.zeros((p, r))
                    for k in range(p):
                        for t in range(r):
                            M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    try:
                        sol, residuals, rank_M, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U[idx, :] = sol
                    except:
                        pass
                
                # Update V
                for idx in range(m*p):
                    j = idx // p
                    k = idx % p
                    M = np.zeros((n, r))
                    for i in range(n):
                        for t in range(r):
                            M[i, t] = U[i*m+j, t] * W[k*n+i, t]
                    try:
                        sol, residuals, rank_M, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V[idx, :] = sol
                    except:
                        pass
                
                # Update W
                for idx in range(p*n):
                    k = idx // n
                    i = idx % n
                    M = np.zeros((m, r))
                    for j in range(m):
                        for t in range(r):
                            M[j, t] = U[i*m+j, t] * V[j*p+k, t]
                    try:
                        sol, residuals, rank_M, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W[idx, :] = sol
                    except:
                        pass
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(r):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            error += (val - 1.0)**2
                if error < 1e-14:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), r):
                        if r < best_rank:
                            best_rank = r
                            best_U = U_r.tolist()
                            best_V = V_r.tolist()
                            best_W = W_r.tolist()
                        return True
            # End iterations
        # End restarts
        return False
    
    # ============================================================
    # Constructive approach: Build a known low-rank decomposition.
    # 
    # For 2x5x5, we can use the following strategy:
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0..4}, k in {0..4}.
    # 
    # We can decompose this using the fact that 1 = 1*1 for all entries.
    # A rank-50 decomposition is trivial. To reduce rank, we need
    # linear dependencies.
    # 
    # Let's use the following constructive rank-25 decomposition:
    # For each j (0..4), define 5 terms (one for each k):
    #   Term (j,k): U[i*m+j, j*5+k] = 1, V[j*p+k, j*5+k] = 1, W[k*n+i, j*5+k] = 1
    # This gives rank 25, but wait, that's not right because U depends on i.
    # 
    # Let me reconsider. The trivial decomposition has rank 50:
    # For each (i,j,k): U[i*m+j, i*25+j*5+k] = 1, etc.
    # 
    # To get rank 25, we need to eliminate the i-dependence from U.
    # But U[i*m+j, t] must depend on i in general.
    # 
    # Alternative: Use a rank-10 decomposition if possible.
    # T[i,j,k] = sum_{t=0}^{9} U[i*5+j, t] * V[j*5+k, t] * W[k*2+i, t]
    # 
    # This is a system of 50 equations in 10*10 + 25*10 + 10*10 = 500 unknowns.
    # Overdetermined, but might have solutions.
    # 
    # Let's try a different constructive approach using block Strassen.
    # ============================================================
    
    # Try constructive rank-25 decomposition using the following idea:
    # Group by j. For each j, we need:
    #   sum_t U[i*5+j, t] * V[j*5+k, t] * W[k*2+i, t] = 1 for all i,k
    # 
    # If we set V[j*5+k, t] = delta_{k, j%5} (one-hot), then we need:
    #   U[i*5+j, j%5] * W[(j%5)*2+i, j%5] = 1 for all i
    # This constrains things heavily.
    # 
    # Let's just use the numerical search but with better parameters.
    
    # First, try a constructive rank-25 approach
    def try_constructive_25():
        """Try to build a rank-25 decomposition."""
        nonlocal best_rank, best_U, best_V, best_W
        
        # Use the following structure:
        # For t in 0..24, let t = j*5 + k where j in 0..4, k in 0..4
        # Set V[j*5+k, t] = 1 if t == j*5+k else 0
        # Then we need: sum_t U[i*5+j, t] * W[k*2+i, t] = 1 for all i,k
        # With V being one-hot, this becomes:
        # U[i*5+j, j*5+k] * W[k*2+i, j*5+k] = 1 for all i,j,k
        # 
        # So set U[i*5+j, j*5+k] = 1 and W[k*2+i, j*5+k] = 1 for all i,j,k.
        # But U has shape (10, 25) and W has shape (10, 25).
        # U[i*5+j, j*5+k] = 1 for all i,j,k means U is all 1s.
        # W[k*2+i, j*5+k] = 1 for all i,j,k means W is all 1s.
        # 
        # Let's verify: sum_t U[i*5+j, t] * V[j*5+k, t] * W[k*2+i, t]
        # = sum_t 1 * delta_{t, j*5+k} * 1 = 1. Yes!
        # 
        # So rank-25 works with U=all 1s, V=identity-like, W=all 1s.
        
        U = np.ones((n*m, 25))
        V = np.zeros((m*p, 25))
        W = np.ones((p*n, 25))
        
        for j in range(m):
            for k in range(p):
                V[j*p+k, j*p+k] = 1  # Wait, this doesn't match the indexing
        
        # Let me redo: t = j*5 + k, so t ranges 0..24
        # V[j*5+k, t] should be 1 when t = j*5+k
        for j in range(m):
            for k in range(p):
                t = j * p + k  # j*5+k
                V[j*p+k, t] = 1
        
        # Verify
        if verify_exact(U.tolist(), V.tolist(), W.tolist(), 25):
            if 25 < best_rank:
                best_rank = 25
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
            return True
        return False
    
    # Try the constructive rank-25
    if try_constructive_25():
        pass  # Found rank 25
    
    # Now try to go below 25 using numerical search
    for r in range(20, 26):
        if time.time() > deadline:
            break
        if try_als(r, n_restarts=5, max_iter=200):
            break
    
    # If we still have rank 50, try a few more ranks
    if best_rank >= 30:
        for r in range(26, 35):
            if time.time() > deadline:
                break
            if try_als(r, n_restarts=3, max_iter=150):
                break
    
    # Return the best found
    if best_rank < rank_trivial:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
# EVOLVE-BLOCK-END
