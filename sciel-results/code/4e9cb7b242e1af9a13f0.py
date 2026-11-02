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
    
    def _try_merge_pair(U, V, W, rank, pair_indices):
        """
        Try to merge two terms (columns) into one, reducing rank by 1.
        Returns (new_U, new_V, new_W, new_rank) or None if merge fails.
        """
        r1, r2 = pair_indices
        if r1 == r2:
            return None
        
        # We want to find a new column (u_new, v_new, w_new) such that
        # for all (i,j,k): u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i]
        #   = U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1]
        #     + U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]
        
        # Strategy: fix w_new = W[:,r1] (or some simple choice),
        # then solve for u_new and v_new.
        # Actually, let's try: set w_new[k*n+i] = W[k*n+i,r1] for all (k,i).
        # Then we need: u_new[i*m+j] * v_new[j*p+k] = 
        #   (U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]) / W[k*n+i,r1]
        
        # This only works if W[k*n+i,r1] != 0. Let's check.
        w1 = W[:, r1]
        w2 = W[:, r2]
        u1 = U[:, r1]
        u2 = U[:, r2]
        v1 = V[:, r1]
        v2 = V[:, r2]
        
        # Build the target matrix M[i,j,k] = u1[i,j]*v1[j,k]*w1[k,i] + u2[i,j]*v2[j,k]*w2[k,i]
        # We want M = u_new[i,j] * v_new[j,k] * w_new[k,i]
        
        # Try w_new = w1. Then we need:
        # u_new[i*m+j] * v_new[j*p+k] = M[i,j,k] / w1[k*n+i]  (when w1[k*n+i] != 0)
        
        # Check if w1 has zeros
        if np.any(np.abs(w1) < 1e-12):
            # Try w_new = w2 instead
            if np.any(np.abs(w2) < 1e-12):
                # Try w_new = w1 + w2
                w_new = w1 + w2
                if np.any(np.abs(w_new) < 1e-12):
                    # Try w_new = w1 - w2
                    w_new = w1 - w2
                    if np.any(np.abs(w_new) < 1e-12):
                        return None
            else:
                w_new = w2
        else:
            w_new = w1
        
        # Now solve for u_new and v_new
        # For each (i,j,k) where w_new[k*n+i] != 0:
        # u_new[i*m+j] * v_new[j*p+k] = M[i,j,k] / w_new[k*n+i]
        
        M = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i,j,k] = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
        
        # We need to find u_new (n*m entries) and v_new (m*p entries) such that
        # u_new[i*m+j] * v_new[j*p+k] = M[i,j,k] / w_new[k*n+i] for all i,j,k
        
        # This is a rank-1 factorization problem for a 3D tensor.
        # Let's set up a linear system. Fix u_new[i*m+0] = 1 for one i (gauge freedom).
        # Actually, let's try: fix u_new[0] = 1, then v_new[j*p+k] = M[0,j,k]/w_new[k*n+0]
        # Then check consistency for other i values.
        
        # Better approach: use the constraint structure.
        # For fixed i, the matrix A_i[j,k] = M[i,j,k]/w_new[k*n+i] should be rank 1
        # and equal to u_new[i*m+j] * v_new[j*p+k].
        
        # Let's try to solve this. Fix u_new[0] = 1.
        # Then for i=0: v_new[j*p+k] = M[0,j,k] / w_new[k*n+0] (for w_new[k*n+0] != 0)
        
        # Check if w_new[k*n+0] is nonzero for all k
        if np.any(np.abs(w_new[0::n]) < 1e-12):
            # Can't use this gauge. Try different approach.
            # Use SVD-like approach: for each i, A_i should be u_new[i*m+j]*v_new[j*p+k]
            # Let's set up a system of equations.
            
            # Unknowns: u_new (9 values) and v_new (15 values) = 24 unknowns
            # Equations: for each (i,j,k) with w_new[k*n+i] != 0:
            #   u_new[i*m+j] * v_new[j*p+k] = M[i,j,k] / w_new[k*n+i]
            
            # This is still nonlinear. Let's try a different strategy.
            
            # Strategy: assume v_new has a specific sparse structure.
            # Or: try to find u_new, v_new by iterating.
            
            # For now, return None for this case.
            return None
        
        # Gauge: u_new[0] = 1
        # v_new[j*p+k] = M[0,j,k] / w_new[k*n+0]
        v_new = np.zeros(m*p)
        for j in range(m):
            for k in range(p):
                v_new[j*p+k] = M[0,j,k] / w_new[k*n+0]
        
        # Now check consistency for i=1,2:
        # u_new[i*m+j] should equal M[i,j,k] / (w_new[k*n+i] * v_new[j*p+k])
        # for all j,k where v_new[j*p+k] != 0 and w_new[k*n+i] != 0
        
        u_new = np.zeros(n*m)
        u_new[0] = 1.0  # gauge
        
        for i in range(1, n):
            for j in range(m):
                # Find consistent value for u_new[i*m+j]
                values = []
                for k in range(p):
                    if abs(w_new[k*n+i]) > 1e-12 and abs(v_new[j*p+k]) > 1e-12:
                        val = M[i,j,k] / (w_new[k*n+i] * v_new[j*p+k])
                        values.append(val)
                
                if not values:
                    # No constraints, set to 0
                    u_new[i*m+j] = 0.0
                    continue
                
                # Check consistency
                if len(values) > 1:
                    if max(values) - min(values) > 1e-9:
                        return None  # Inconsistent
                    u_new[i*m+j] = values[0]
                else:
                    u_new[i*m+j] = values[0]
        
        # Verify the solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = M[i,j,k]
                    actual = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
                    if abs(expected - actual) > 1e-9:
                        return None
        
        # Build new matrices with merged column
        new_rank = rank - 1
        new_U = np.zeros((n*m, new_rank))
        new_V = np.zeros((m*p, new_rank))
        new_W = np.zeros((p*n, new_rank))
        
        # Copy all columns except r1 and r2
        col_map = []
        for r in range(rank):
            if r != r1 and r != r2:
                col_map.append(r)
        
        for new_col, old_col in enumerate(col_map):
            new_U[:, new_col] = U[:, old_col]
            new_V[:, new_col] = V[:, old_col]
            new_W[:, new_col] = W[:, old_col]
        
        # Add merged column at the end
        merged_col = new_rank - 1
        new_U[:, merged_col] = u_new
        new_V[:, merged_col] = v_new
        new_W[:, merged_col] = w_new
        
        return new_U, new_V, new_W, new_rank
    
    def _try_merge_pair_general(U, V, W, rank, r1, r2):
        """
        More general merge: try different choices for w_new.
        Returns (new_U, new_V, new_W, new_rank) or None.
        """
        w1 = W[:, r1]
        w2 = W[:, r2]
        u1 = U[:, r1]
        u2 = U[:, r2]
        v1 = V[:, r1]
        v2 = V[:, r2]
        
        M = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i,j,k] = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
        
        # Try different w_new choices
        w_new_candidates = [w1, w2, w1+w2, w1-w2, w2-w1]
        
        for w_new in w_new_candidates:
            # Skip if w_new has too many zeros
            if np.sum(np.abs(w_new) > 1e-12) < 3:
                continue
            
            # Try to solve for u_new, v_new
            # Use the approach: for each i, the matrix A_i[j,k] = M[i,j,k]/w_new[k*n+i]
            # should be rank 1: A_i[j,k] = u_new[i*m+j] * v_new[j*p+k]
            
            # For this to work, w_new[k*n+i] must be nonzero for the (k) values we use.
            # Let's check for each i if there are enough k values.
            
            valid = True
            u_new = np.zeros(n*m)
            v_new = np.zeros(m*p)
            
            # Try to determine v_new from i=0 (if w_new[k*n+0] is nonzero for some k)
            k_valid_0 = [k for k in range(p) if abs(w_new[k*n+0]) > 1e-12]
            if len(k_valid_0) == 0:
                # Try i=1
                k_valid_0 = [k for k in range(p) if abs(w_new[k*n+1]) > 1e-12]
                if len(k_valid_0) == 0:
                    # Try i=2
                    k_valid_0 = [k for k in range(p) if abs(w_new[k*n+2]) > 1e-12]
                    if len(k_valid_0) == 0:
                        valid = False
                else:
                    # Use i=2 as reference
                    ref_i = 2
                    for j in range(m):
                        for k in k_valid_0:
                            v_new[j*p+k] = M[ref_i,j,k] / w_new[k*n+ref_i]
            elif len(k_valid_0) > 0:
                ref_i = 0
                for j in range(m):
                    for k in k_valid_0:
                        v_new[j*p+k] = M[ref_i,j,k] / w_new[k*n+ref_i]
            else:
                valid = False
            
            if not valid:
                continue
            
            # Now determine u_new for each i
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
                            valid = False
                            break
                        u_new[i*m+j] = values[0]
                    else:
                        u_new[i*m+j] = values[0]
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = M[i,j,k]
                        actual = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
                        if abs(expected - actual) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Build new matrices
                new_rank = rank - 1
                new_U = np.zeros((n*m, new_rank))
                new_V = np.zeros((m*p, new_rank))
                new_W = np.zeros((p*n, new_rank))
                
                col_map = []
                for r in range(rank):
                    if r != r1 and r != r2:
                        col_map.append(r)
                
                for new_col, old_col in enumerate(col_map):
                    new_U[:, new_col] = U[:, old_col]
                    new_V[:, new_col] = V[:, old_col]
                    new_W[:, new_col] = W[:, old_col]
                
                merged_col = new_rank - 1
                new_U[:, merged_col] = u_new
                new_V[:, merged_col] = v_new
                new_W[:, merged_col] = w_new
                
                return new_U, new_V, new_W, new_rank
        
        return None
    
    # Start with schoolbook
    U, V, W, rank = _make_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank by merging pairs
    # We need to reduce from 45 to at most 43 (known minimal is 43)
    # So we need at least 2 successful merges
    
    max_merges = 2  # Try to get rank 43
    for merge_round in range(max_merges):
        if _remaining() < 5.0:
            break
        
        merged = False
        # Try all pairs of columns
        for r1 in range(rank):
            if _remaining() < 5.0:
                break
            for r2 in range(r1+1, rank):
                if _remaining() < 5.0:
                    break
                
                result = _try_merge_pair_general(U, V, W, rank, r1, r2)
                if result is not None:
                    U, V, W, rank = result
                    merged = True
                    # Update best if improved
                    if rank < best_rank:
                        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
                    break
            if merged:
                break
        
        if not merged:
            break
    
    # Final verification
    if _verify(best_U, best_V, best_W, best_rank):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
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
