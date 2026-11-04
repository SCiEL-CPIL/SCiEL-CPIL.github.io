"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 48).
    2. Use local search to try to reduce the rank by finding
       dependencies in the tensor or using known lower-bound constructions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W, rank
    
    def _verify(U, V, W):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor T[i,j,k,l] = sum_r U[i,j,r] * V[j,k,r] * W[k,l,r]
        # should equal 1 if i==l and j==k, else 0
        # Actually for matrix mult: C[i,l] = sum_j A[i,j] B[j,l]
        # Tensor form: T[i,j,k,l] where output index is (i,l), input indices are (j,k)
        # W convention: W[k*n+i, r] so W index is k*n+i
        
        # Let's verify by checking the reconstruction
        # For each r, we have outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # The sum over r should give the identity tensor
        
        # Check: for all i in [0,n), j in [0,m), k in [0,p), l in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = 1 if i==l and j==k else 0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > 1e-10:
                            return False
        return True
    
    def _try_reduce_rank(U, V, W, current_rank, max_time=5.0):
        """
        Try to reduce the rank by finding linear dependencies.
        Returns (U, V, W, new_rank) if successful, else original.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        # Try to find columns that can be eliminated
        # A column r can be eliminated if it's a linear combination of others
        # in all three matrices simultaneously
        
        for attempt in range(50):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            # Pick a random column to try to eliminate
            r_elim = random.randint(0, best_rank - 1)
            
            # Try to express column r_elim as combination of others
            # in U, V, W simultaneously
            
            other_cols = [c for c in range(best_rank) if c != r_elim]
            if len(other_cols) < 1:
                break
            
            # For U: find coefficients a such that U[:, r_elim] ≈ sum a_c * U[:, c]
            # For V: find coefficients b such that V[:, r_elim] ≈ sum b_c * V[:, c]  
            # For W: find coefficients c such that W[:, r_elim] ≈ sum c_c * W[:, c]
            
            # Use least squares to find the coefficients
            U_other = best_U[:, other_cols]
            V_other = best_V[:, other_cols]
            W_other = best_W[:, other_cols]
            
            U_target = best_U[:, r_elim]
            V_target = best_V[:, r_elim]
            W_target = best_W[:, r_elim]
            
            # Solve for coefficients
            try:
                a_coeff, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                b_coeff, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                c_coeff, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
            except:
                continue
            
            # Check if the reconstruction is good
            U_recon = U_other @ a_coeff
            V_recon = V_other @ b_coeff
            W_recon = W_other @ c_coeff
            
            U_err = np.linalg.norm(U_recon - U_target) / (np.linalg.norm(U_target) + 1e-10)
            V_err = np.linalg.norm(V_recon - V_target) / (np.linalg.norm(V_target) + 1e-10)
            W_err = np.linalg.norm(W_recon - W_target) / (np.linalg.norm(W_target) + 1e-10)
            
            if U_err < 1e-8 and V_err < 1e-8 and W_err < 1e-8:
                # Great! We can eliminate this column
                new_U = np.delete(best_U, r_elim, axis=1)
                new_V = np.delete(best_V, r_elim, axis=1)
                new_W = np.delete(best_W, r_elim, axis=1)
                
                # Verify
                if _verify(new_U, new_V, new_W):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank -= 1
        
        return best_U, best_V, best_W, best_rank
    
    def _local_search(U, V, W, current_rank, max_time=10.0):
        """
        Local search: try small perturbations to find better decompositions.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        # Try removing columns one at a time with least squares repair
        for attempt in range(100):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            # Pick a random column to try to eliminate
            if best_rank <= 1:
                break
            r_elim = random.randint(0, best_rank - 1)
            
            other_cols = [c for c in range(best_rank) if c != r_elim]
            
            U_other = best_U[:, other_cols]
            V_other = best_V[:, other_cols]
            W_other = best_W[:, other_cols]
            
            U_target = best_U[:, r_elim]
            V_target = best_V[:, r_elim]
            W_target = best_W[:, r_elim]
            
            try:
                a_coeff, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                b_coeff, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                c_coeff, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
            except:
                continue
            
            U_recon = U_other @ a_coeff
            V_recon = V_other @ b_coeff
            W_recon = W_other @ c_coeff
            
            U_err = np.linalg.norm(U_recon - U_target)
            V_err = np.linalg.norm(V_recon - V_target)
            W_err = np.linalg.norm(W_recon - W_target)
            
            total_err = U_err + V_err + W_err
            
            if total_err < 1e-6:
                new_U = np.delete(best_U, r_elim, axis=1)
                new_V = np.delete(best_V, r_elim, axis=1)
                new_W = np.delete(best_W, r_elim, axis=1)
                
                if _verify(new_U, new_V, new_W):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank -= 1
        
        return best_U, best_V, best_W, best_rank
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, rank, max_time=20.0)
    
    # Try local search for further reduction
    U, V, W, rank = _local_search(U, V, W, rank, max_time=30.0)
    
    # Convert to list format with integer/half-integer entries if possible
    def _to_clean_list(arr):
        """Convert array to list, rounding to nearest half-integer if close."""
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                # Check if close to integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
# EVOLVE-BLOCK-END
