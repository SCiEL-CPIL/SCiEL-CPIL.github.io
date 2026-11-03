"""Rank-71 decomposition with optimized verification and rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def build_rank71():
        """Build the rank-71 decomposition using 4 Strassen blocks + schoolbook."""
        total_rank = 71
        U = np.zeros((n * m, total_rank), dtype=np.float64)
        V = np.zeros((m * p, total_rank), dtype=np.float64)
        W = np.zeros((p * n, total_rank), dtype=np.float64)
        
        col_idx = 0
        blocks = [
            (0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)
        ]
        
        for (i0, j0, k0) in blocks:
            # Strassen 2x2x2 block: 7 terms instead of 8
            # Term 1: (A00+A11)(B00+B11)
            U[i0*m + j0, col_idx] += 1
            U[(i0+1)*m + (j0+1), col_idx] += 1
            V[j0*p + k0, col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += 1
            W[k0*n + i0, col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 2: (A10+A11)B00
            U[(i0+1)*m + j0, col_idx] += 1
            U[(i0+1)*m + (j0+1), col_idx] += 1
            V[j0*p + k0, col_idx] += 1
            W[k0*n + (i0+1), col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += -1
            col_idx += 1
            # Term 3: A00(B01-B11)
            U[i0*m + j0, col_idx] += 1
            V[j0*p + (k0+1), col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += -1
            W[(k0+1)*n + i0, col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 4: A11(B10-B00)
            U[(i0+1)*m + (j0+1), col_idx] += 1
            V[(j0+1)*p + k0, col_idx] += 1
            V[j0*p + k0, col_idx] += -1
            W[k0*n + i0, col_idx] += 1
            W[k0*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 5: (A00+A01)B11
            U[i0*m + j0, col_idx] += 1
            U[i0*m + (j0+1), col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += 1
            W[k0*n + i0, col_idx] += -1
            W[(k0+1)*n + i0, col_idx] += 1
            col_idx += 1
            # Term 6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col_idx] += 1
            U[i0*m + j0, col_idx] += -1
            V[j0*p + k0, col_idx] += 1
            V[j0*p + (k0+1), col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col_idx] += 1
            U[(i0+1)*m + (j0+1), col_idx] += -1
            V[(j0+1)*p + k0, col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += 1
            W[k0*n + i0, col_idx] += 1
            col_idx += 1
        
        # Track which (i,j,k) triples are covered by Strassen blocks
        covered = np.zeros((n, m, p), dtype=bool)
        for (i0, j0, k0) in blocks:
            for ii in range(i0, i0+2):
                for jj in range(j0, j0+2):
                    for kk in range(k0, k0+2):
                        covered[ii, jj, kk] = True
        
        # Schoolbook for remaining terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i, j, k]:
                        U[i*m + j, col_idx] += 1
                        V[j*p + k, col_idx] += 1
                        W[k*n + i, col_idx] += 1
                        col_idx += 1
        
        return U, V, W
    
    def verify_fast(U, V, W):
        """Vectorized verification of tensor decomposition."""
        rank = U.shape[1]
        # Compute all 75 tensor entries: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Use broadcasting for efficiency
        errors = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]  # shape (n*m,)
            v_col = V[:, r]  # shape (m*p,)
            w_col = W[:, r]  # shape (p*n,)
            # Reshape to 3D
            u_3d = u_col.reshape(n, m)  # U[i*m+j] -> [i,j]
            v_3d = v_col.reshape(m, p)  # V[j*p+k] -> [j,k]
            w_3d = w_col.reshape(p, n)  # W[k*n+i] -> [k,i]
            # T[i,j,k] += u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
            # Build the contribution: for each (i,j,k), add u_3d[i,j]*v_3d[j,k]*w_3d[k,i]
            # This is an outer-product-like operation
            # u_3d[i,j] * v_3d[j,k] gives (n,p) matrix for fixed j... 
            # Actually let's do it differently:
            # contribution[i,j,k] = u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
            # We can compute this as:
            # temp[i,j,k] = u_3d[i,j] * v_3d[j,k]  -> need to align j
            # u_3d has shape (n,m), v_3d has shape (m,p)
            # u_3d[:, :, None] * v_3d[None, :, :] won't work directly
            # Let's use: for each j, u_3d[:,j] * v_3d[j,:] gives (n,p)
            # Then multiply by w_3d[k,i] which is (p,n)
            # So: sum_j (u_3d[:,j] * v_3d[j,:]) * w_3d.T
            # Wait: contribution[i,j,k] = u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
            # Let's just loop over j since m=5
            for j in range(m):
                # u_3d[:, j] has shape (n,), v_3d[j, :] has shape (p,)
                # outer product: (n, p)
                up = np.outer(u_3d[:, j], v_3d[j, :])  # up[i,k]
                # Now multiply by w_3d[k, i] for each (i,k)
                # w_3d has shape (p, n), so w_3d[k, i]
                # up[i,k] * w_3d[k,i] = up[i,k] * w_3d.T[i,k]
                errors[:, j, :] += up * w_3d.T
        return np.all(np.abs(errors - 1.0) < 1e-9)
    
    def verify_single(U, V, W):
        """Simple verification - all 75 entries should be 1."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def can_remove_column(U, V, W, col_to_remove):
        """Check if removing a specific column still gives valid decomposition."""
        rank = U.shape[1]
        new_rank = rank - 1
        U_test = np.delete(U, col_to_remove, axis=1)
        V_test = np.delete(V, col_to_remove, axis=1)
        W_test = np.delete(W, col_to_remove, axis=1)
        
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_test[i*m+j, :], V_test[j*p+k, :] * W_test[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False, None
        return True, (U_test, V_test, W_test)
    
    # Build the known rank-71 solution
    U, V, W = build_rank71()
    
    # Verify correctness
    if not verify_fast(U, V, W):
        raise ValueError("Rank-71 decomposition failed verification")
    
    best_U, best_V, best_W = U, V, W
    best_rank = 71
    
    # Try to reduce rank by finding removable columns
    # Check each column - if removing it still gives valid decomposition, we can reduce
    for r in range(71):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        valid, result = can_remove_column(U, V, W, r)
        if valid:
            best_U, best_V, best_W = result
            best_rank = 70
            U, V, W = result
            break  # Found a reduction
    
    # If we found rank 70, try to reduce further
    if best_rank == 70 and time.time() - START_TIME < TIME_LIMIT - 5:
        for r in range(70):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            valid, result = can_remove_column(U, V, W, r)
            if valid:
                best_U, best_V, best_W = result
                best_rank = 69
                U, V, W = result
                break
    
    # If we found rank 69, try one more level
    if best_rank == 69 and time.time() - START_TIME < TIME_LIMIT - 3:
        for r in range(69):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            valid, result = can_remove_column(U, V, W, r)
            if valid:
                best_U, best_V, best_W = result
                best_rank = 68
                U, V, W = result
                break
    
    # Try alternative block configurations that might yield better rank
    # Try different Strassen block placements
    if time.time() - START_TIME < TIME_LIMIT - 10 and best_rank >= 71:
        # Try blocks that cover row 2 as well
        alt_blocks = [
            (1, 0, 0), (1, 2, 0), (1, 0, 2), (1, 2, 2)
        ]
        
        def build_with_blocks(blocks):
            total_rank = 71
            U_t = np.zeros((n * m, total_rank), dtype=np.float64)
            V_t = np.zeros((m * p, total_rank), dtype=np.float64)
            W_t = np.zeros((p * n, total_rank), dtype=np.float64)
            
            col_idx = 0
            for (i0, j0, k0) in blocks:
                # Same Strassen pattern
                U_t[i0*m + j0, col_idx] += 1
                U_t[(i0+1)*m + (j0+1), col_idx] += 1
                V_t[j0*p + k0, col_idx] += 1
                V_t[(j0+1)*p + (k0+1), col_idx] += 1
                W_t[k0*n + i0, col_idx] += 1
                W_t[(k0+1)*n + (i0+1), col_idx] += 1
                col_idx += 1
                U_t[(i0+1)*m + j0, col_idx] += 1
                U_t[(i0+1)*m + (j0+1), col_idx] += 1
                V_t[j0*p + k0, col_idx] += 1
                W_t[k0*n + (i0+1), col_idx] += 1
                W_t[(k0+1)*n + (i0+1), col_idx] += -1
                col_idx += 1
                U_t[i0*m + j0, col_idx] += 1
                V_t[j0*p + (k0+1), col_idx] += 1
                V_t[(j0+1)*p + (k0+1), col_idx] += -1
                W_t[(k0+1)*n + i0, col_idx] += 1
                W_t[(k0+1)*n + (i0+1), col_idx] += 1
                col_idx += 1
                U_t[(i0+1)*m + (j0+1), col_idx] += 1
                V_t[(j0+1)*p + k0, col_idx] += 1
                V_t[j0*p + k0, col_idx] += -1
                W_t[k0*n + i0, col_idx] += 1
                W_t[k0*n + (i0+1), col_idx] += 1
                col_idx += 1
                U_t[i0*m + j0, col_idx] += 1
                U_t[i0*m + (j0+1), col_idx] += 1
                V_t[(j0+1)*p + (k0+1), col_idx] += 1
                W_t[k0*n + i0, col_idx] += -1
                W_t[(k0+1)*n + i0, col_idx] += 1
                col_idx += 1
                U_t[(i0+1)*m + j0, col_idx] += 1
                U_t[i0*m + j0, col_idx] += -1
                V_t[j0*p + k0, col_idx] += 1
                V_t[j0*p + (k0+1), col_idx] += 1
                W_t[(k0+1)*n + (i0+1), col_idx] += 1
                col_idx += 1
                U_t[i0*m + (j0+1), col_idx] += 1
                U_t[(i0+1)*m + (j0+1), col_idx] += -1
                V_t[(j0+1)*p + k0, col_idx] += 1
                V_t[(j0+1)*p + (k0+1), col_idx] += 1
                W_t[k0*n + i0, col_idx] += 1
                col_idx += 1
            
            covered = np.zeros((n, m, p), dtype=bool)
            for (i0, j0, k0) in blocks:
                for ii in range(i0, min(i0+2, n)):
                    for jj in range(j0, min(j0+2, m)):
                        for kk in range(k0, min(k0+2, p)):
                            covered[ii, jj, kk] = True
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not covered[i, j, k]:
                            if col_idx >= total_rank:
                                return None
                            U_t[i*m + j, col_idx] += 1
                            V_t[j*p + k, col_idx] += 1
                            W_t[k*n + i, col_idx] += 1
                            col_idx += 1
            
            return U_t, V_t, W_t
        
        # Try alternative block configurations
        alt_configs = [
            [(0,0,0), (0,2,0), (0,0,2), (0,2,2)],  # original
            [(1,0,0), (1,2,0), (1,0,2), (1,2,2)],  # shifted down
            [(0,0,0), (0,2,0), (1,0,2), (1,2,2)],  # mixed
            [(0,0,0), (1,2,0), (0,0,2), (1,2,2)],  # mixed 2
        ]
        
        for config in alt_configs:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            result = build_with_blocks(config)
            if result is None:
                continue
            U_alt, V_alt, W_alt = result
            if verify_fast(U_alt, V_alt, W_alt):
                # Try rank reduction on this alternative
                U_c, V_c, W_c = U_alt, V_alt, W_alt
                for r in range(71):
                    if time.time() - START_TIME > TIME_LIMIT:
                        break
                    valid, res = can_remove_column(U_c, V_c, W_c, r)
                    if valid:
                        if best_rank > 70:
                            best_U, best_V, best_W = res
                            best_rank = 70
                        break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
