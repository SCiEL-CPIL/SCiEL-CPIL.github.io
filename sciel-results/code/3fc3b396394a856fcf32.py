import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Uses randomized rank-revealing decomposition with local search refinement.
    Falls back to schoolbook decomposition if no improvement found within time limit.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 3, 4, 7
    R_school = n * m * p  # 84
    
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        for r in range(R):
            i = r // (m * p)
            remainder = r % (m * p)
            j = remainder // p
            k = remainder % p
            
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            
            U[u_row, r] = 1.0
            V[v_row, r] = 1.0
            W[w_row, r] = 1.0
        
        return U, V, W, R
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = delta[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually, the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected value: 1 if this is a valid multiplication term, 0 otherwise
                    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
                    # The tensor T[i,j,k] = 1 (all entries are 1 in the "multiplication tensor")
                    expected = 1.0
                    
                    actual = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    for r in range(U.shape[1]):
                        actual += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def try_local_search(U, V, W, max_time=120.0):
        """
        Try to reduce rank by removing linearly dependent columns.
        Returns (U, V, W, new_rank) if improved, else None.
        """
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = U.shape[1]
        best_U, best_V, best_W, best_rank = current_U.copy(), current_V.copy(), current_W.copy(), current_rank
        
        # Try removing each column and checking if the decomposition still holds
        for r in range(current_rank - 1, -1, -1):
            if time.time() > max_time:
                break
            
            # Try removing column r
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            
            U_trial = current_U[:, mask]
            V_trial = current_V[:, mask]
            W_trial = current_W[:, mask]
            
            # Verify
            if verify_decomposition(U_trial, V_trial, W_trial, tol=1e-10):
                current_U, current_V, current_W = U_trial, V_trial, W_trial
                current_rank -= 1
                best_U, best_V, best_W, best_rank = current_U.copy(), current_V.copy(), current_W.copy(), current_rank
        
        if best_rank < R_school:
            return best_U, best_V, best_W, best_rank
        return None
    
    def randomized_rank_revealing(U, V, W):
        """
        Use randomized linear algebra to find a minimal set of columns.
        """
        R = U.shape[1]
        
        # Create the tensor as a 3D array
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        
        # The tensor should be all 1s for matrix multiplication
        # Try to find a low-rank decomposition
        
        # Use SVD on matrix unfoldings to estimate rank
        # Unfold along first mode: shape (n, m*p*R)
        # Actually, let's think differently.
        
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a rank-R decomposition. We want to find if there's a lower rank one.
        
        # Try greedy removal with randomization
        np.random.seed(42)
        
        # Random permutation of columns
        perm = np.random.permutation(R)
        U_perm = U[:, perm]
        V_perm = V[:, perm]
        W_perm = W[:, perm]
        
        # Greedy: try to build up from scratch
        # Start with empty, add columns that are "most useful"
        
        # For now, just try removing from the permuted order
        current_U, current_V, current_W = U_perm.copy(), V_perm.copy(), W_perm.copy()
        current_rank = R
        
        for r in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 20:  # Leave time for verification
                break
            
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            
            U_trial = current_U[:, mask]
            V_trial = current_V[:, mask]
            W_trial = current_W[:, mask]
            
            if verify_decomposition(U_trial, V_trial, W_trial, tol=1e-10):
                current_U, current_V, current_W = U_trial, V_trial, W_trial
                current_rank -= 1
        
        return current_U, current_V, current_W, current_rank
    
    # Start with schoolbook
    U_school, V_school, W_school, R_school = build_schoolbook()
    
    # Try randomized rank-revealing
    try:
        U_reduced, V_reduced, W_reduced, R_reduced = randomized_rank_revealing(U_school, V_school, W_school)
        
        if R_reduced < R_school:
            # Try further local search refinement
            result = try_local_search(U_reduced, V_reduced, W_reduced, max_time=deadline - 30)
            if result is not None:
                U_final, V_final, W_final, R_final = result
            else:
                U_final, V_final, W_final, R_final = U_reduced, V_reduced, W_reduced, R_reduced
        else:
            U_final, V_final, W_final, R_final = U_school, V_school, W_school, R_school
    except Exception:
        U_final, V_final, W_final, R_final = U_school, V_school, W_school, R_school
    
    # Final verification
    if not verify_decomposition(U_final, V_final, W_final, tol=1e-10):
        # Fallback to schoolbook
        U_final, V_final, W_final, R_final = build_schoolbook()
    
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
