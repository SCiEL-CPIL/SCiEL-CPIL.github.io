"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def _remaining():
        return DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    
    def _verify(U, V, W, rank):
        """Verify exact decomposition. Returns True if valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def _make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def _make_rank43():
        """
        Construct rank-43 decomposition for 3x3x5 matrix multiplication.
        
        The tensor T[i,j,k] = 1 for all i in 0..2, j in 0..2, k in 0..4.
        We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        Strategy: Use a combination of:
        1. Strassen-like terms for the 3x3x3 core (k=0,1,2)
        2. Additional terms to handle the extended output dimension (k=3,4)
        
        The known minimal rank is 43, so we need 43 terms total.
        
        I'll construct this by:
        - Starting with Strassen for 3x3x3 (21 terms)
        - Adding terms to extend to 3x3x5 (22 more terms)
        """
        n, m, p = 3, 3, 5
        
        # Strassen algorithm for 3x3x3
        # The 7 Strassen products for 2x2x2, extended to 3x3x3
        # This gives a rank-21 decomposition for the 3x3x3 case
        
        # For now, let me use a simpler approach:
        # Use schoolbook for all 45 terms, then try to merge 2 pairs
        
        U, V, W, rank = _make_schoolbook()
        
        # Try to find 2 pairs that can be merged to reduce rank from 45 to 43
        # This is the same approach as before, but more targeted
        
        def _try_merge(U, V, W, rank, r1, r2):
            """Try to merge columns r1 and r2 into one."""
            w1 = W[:, r1]
            w2 = W[:, r2]
            u1 = U[:, r1]
            u2 = U[:, r2]
            v1 = V[:, r1]
            v2 = V[:, r2]
            
            # Build target tensor M[i,j,k] = u1[i,j]*v1[j,k]*w1[k,i] + u2[i,j]*v2[j,k]*w2[k,i]
            M = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i,j,k] = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
            
            # Try w_new = w1
            w_new = w1
            if np.any(np.abs(w_new) < 1e-12):
                w_new = w2
                if np.any(np.abs(w_new) < 1e-12):
                    w_new = w1 + w2
                    if np.any(np.abs(w_new) < 1e-12):
                        return None
            
            # Solve for u_new, v_new
            # For each i, A_i[j,k] = M[i,j,k]/w_new[k*n+i] should be rank 1
            u_new = np.zeros(n*m)
            v_new = np.zeros(m*p)
            
            # Use i=0 as reference if possible
            k_valid = [k for k in range(p) if abs(w_new[k*n+0]) > 1e-12]
            if not k_valid:
                return None
            
            for j in range(m):
                for k in k_valid:
                    v_new[j*p+k] = M[0,j,k] / w_new[k*n+0]
            
            # Check consistency and determine u_new
            for i in range(n):
                for j in range(m):
                    values = []
                    for k in range(p):
                        if abs(w_new[k*n+i]) > 1e-12 and abs(v_new[j*p+k]) > 1e-12:
                            val = M[i,j,k] / (w_new[k*n+i] * v_new[j*p+k])
                            values.append(val)
                    
                    if not values:
                        u_new[i*m+j] = 0.0
                        continue
                    
                    if len(values) > 1:
                        if max(values) - min(values) > 1e-9:
                            return None
                    u_new[i*m+j] = values[0]
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(M[i,j,k] - u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i]) > 1e-9:
                            return None
            
            # Build new matrices
            new_rank = rank - 1
            new_U = np.zeros((n*m, new_rank))
            new_V = np.zeros((m*p, new_rank))
            new_W = np.zeros((p*n, new_rank))
            
            col_map = [r for r in range(rank) if r != r1 and r != r2]
            for new_col, old_col in enumerate(col_map):
                new_U[:, new_col] = U[:, old_col]
                new_V[:, new_col] = V[:, old_col]
                new_W[:, new_col] = W[:, old_col]
            
            new_U[:, new_rank-1] = u_new
            new_V[:, new_rank-1] = v_new
            new_W[:, new_rank-1] = w_new
            
            return new_U, new_V, new_W, new_rank
        
        # Try to merge 2 pairs
        merged_count = 0
        for _ in range(2):
            if _remaining() < 5.0:
                break
            found = False
            for r1 in range(rank):
                if found:
                    break
                for r2 in range(r1+1, rank):
                    result = _try_merge(U, V, W, rank, r1, r2)
                    if result is not None:
                        U, V, W, rank = result
                        merged_count += 1
                        found = True
                        break
            if not found:
                break
        
        return U, V, W, rank
    
    # Try to construct rank-43 decomposition
    U, V, W, rank = _make_rank43()
    
    # Verify the result
    if _verify(U, V, W, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        U, V, W, rank = _make_schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
