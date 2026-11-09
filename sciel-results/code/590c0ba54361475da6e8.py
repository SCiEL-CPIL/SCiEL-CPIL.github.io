"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # safety margin before 150s limit

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 75
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

    best_result = (U_school, V_school, W_school)
    best_rank = rank_school

    # The 3x5x5 tensor decomposes as block-diagonal of three 1x5x5 tensors.
    # For each output row i (0..2), we need a rank-r decomposition of 1x5x5:
    #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(j,k)
    # Since the three blocks are independent, total rank = 3 * r_1x5x5.
    #
    # We'll search for a low-rank decomposition of 1x5x5 (rank < 25)
    # and then replicate it for each of the 3 output rows.
    #
    # Strategy: For 1x5x5, we need:
    #   sum_c U[j,c] * V[j,k,c] * W[k,c] = delta(j,k)  for j,k in 0..4
    # where U is 5xr, V is 5xr, W is 5xr (but W index is k, not (k,i)).
    #
    # Actually for the 1x5x5 case (n=1, m=5, p=5):
    #   U has shape (5, r) indexed by j
    #   V has shape (25, r) indexed by (j,k)
    #   W has shape (5, r) indexed by k
    #   Constraint: sum_c U[j,c]*V[j,k,c]*W[k,c] = 1 if j==k else 0
    #
    # Let's search for rank r decompositions of 1x5x5.

    def find_1x5x5(r, max_attempts=500):
        """Find a rank-r decomposition of 1x5x5 matrix multiplication.
        Returns (U, V, W) each of shape (5, r), (25, r), (5, r) or None.
        U[j, c], V[j*5+k, c], W[k, c]
        Constraint: sum_c U[j,c]*V[j*5+k,c]*W[k,c] = delta(j,k)
        """
        np.random.seed(42 + r)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random U and V with small integer entries
            # U: 5 x r, V: 25 x r
            U_rand = np.random.randint(-3, 4, size=(5, r))
            V_rand = np.random.randint(-3, 4, size=(25, r))
            
            # For each k, solve for W[k, :]:
            # sum_c U[j,c] * V[j*5+k, c] * W[k, c] = delta(j,k) for all j
            # This is: (U * V_k)^T @ W[k,:] = e_k  where V_k[j,c] = V[j*5+k, c]
            # Actually: for fixed k, the constraint is:
            #   for each j: sum_c U[j,c] * V[j*5+k, c] * W[k, c] = 1 if j==k else 0
            # Let A[j, c] = U[j, c] * V[j*5+k, c]
            # Then A @ W[k,:] = e_k (the k-th standard basis vector in R^5)
            
            W_rand = np.zeros((5, r))
            valid = True
            
            for k in range(5):
                # Build A: 5 x r
                A = np.zeros((5, r))
                for j in range(5):
                    A[j, :] = U_rand[j, :] * V_rand[j*5+k, :]
                
                # Target: e_k
                b = np.zeros(5)
                b[k] = 1.0
                
                try:
                    W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.allclose(A @ W_row, b, atol=1e-10):
                        W_rand[k, :] = W_row
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Full verification of 1x5x5
                ok = True
                for j in range(5):
                    for k in range(5):
                        s = 0.0
                        for c in range(r):
                            s += U_rand[j, c] * V_rand[j*5+k, c] * W_rand[k, c]
                        expected = 1.0 if j == k else 0.0
                        if abs(s - expected) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return (U_rand, V_rand, W_rand)
        
        return None

    # Try to find a 1x5x5 decomposition at various ranks
    # Schoolbook for 1x5x5 is rank 25
    # Known: rank 19 is achievable for 1x5x5 (from literature)
    # Let's try from rank 19 down
    
    best_1x5x5 = None
    best_1x5x5_rank = 25
    
    for r in range(19, 24):  # Try 19, 20, 21, 22, 23
        if time.time() > deadline:
            break
        
        result = find_1x5x5(r, max_attempts=300)
        if result is not None:
            best_1x5x5 = result
            best_1x5x5_rank = r
            break  # Found the lowest rank, stop
    
    # If we found a 1x5x5 decomposition, construct the 3x5x5 one
    if best_1x5x5 is not None:
        U_1, V_1, W_1 = best_1x5x5
        r_1 = best_1x5x5_rank
        total_rank = 3 * r_1
        
        # Construct 3x5x5 decomposition
        # U: (15, total_rank) - indexed by (i, j)
        # V: (25, total_rank) - indexed by (j, k)
        # W: (15, total_rank) - indexed by (k, i)
        
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        for i in range(n):
            offset = i * r_1
            # U[i*m + j, offset+c] = U_1[j, c]
            for j in range(m):
                U_full[i*m + j, offset:offset+r_1] = U_1[j, :]
            # V[j*p + k, offset+c] = V_1[j*p + k, c]  (same V for all i)
            for j in range(m):
                for k in range(p):
                    V_full[j*p + k, offset:offset+r_1] = V_1[j*p + k, :]
            # W[k*n + i, offset+c] = W_1[k, c]
            for k in range(p):
                W_full[k*n + i, offset:offset+r_1] = W_1[k, :]
        
        # Verify
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            best_result = (U_list, V_list, W_list)
            best_rank = total_rank
            print(f"Found 3x5x5 decomposition with rank {total_rank} (1x5x5 rank {r_1})")
    
    # Now try to do better with a direct search on the full 3x5x5 tensor
    # Using the block structure: for each (i,k) pair, solve for W[k*n+i, :]
    # given random U and V.
    
    if time.time() < deadline - 10:
        # Direct randomized search on full 3x5x5
        # Try ranks below what we found
        search_start = min(best_rank - 1, 56)
        search_end = max(30, best_rank - 15)
        
        for r in range(search_start, search_end - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            max_attempts = 100
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Generate random U and V with small integer entries
                U_rand = np.random.randint(-2, 3, size=(n*m, r))
                V_rand = np.random.randint(-2, 3, size=(m*p, r))
                
                W_rand = np.zeros((p*n, r))
                valid = True
                
                for i in range(n):
                    for k in range(p):
                        # For fixed (i, k), solve for W[k*n+i, :]:
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                        b = np.ones(m)
                        
                        try:
                            W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            if np.allclose(A @ W_row, b, atol=1e-10):
                                W_rand[k*n+i, :] = W_row
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    if _verify(U_list, V_list, W_list, n, m, p):
                        found = True
                        best_result = (U_list, V_list, W_list)
                        best_rank = r
                        print(f"Found direct 3x5x5 decomposition with rank {r}")
                        break
            
            if found:
                continue
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
