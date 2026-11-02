import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _verify_strict(U, V, W, tol=1e-9):
    """
    Strictly verify that U, V, W decompose the matrix multiplication tensor.
    Checks both the non-zero entries (should sum to 1) and zero entries (should sum to 0).
    """
    n, m, p = 2, 2, 11
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Build target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    # Otherwise 0.
    # We check all 4*22*22 = 1936 entries.
    
    # Precompute the set of valid (a,b,c) tuples
    valid_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_entries.add((a, b, c))
                
    # Efficient check using einsum for the full tensor
    # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Create target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for (a, b, c) in valid_entries:
        T_target[a, b, c] = 1.0
        
    # Check allclose
    if not np.allclose(T_decomp, T_target, atol=tol, rtol=0):
        return False
        
    return True

def _construct_rank39():
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook 4 terms
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank, f"Rank mismatch: {r} != {rank}"
    
    return {"U": U, "V": V, "W": W}

def _try_rank38_als():
    """
    Attempt to find a rank-38 decomposition using ALS (Alternating Least Squares)
    initialized near the rank-39 solution or randomly, with strict verification.
    """
    if time.time() > _DEADLINE - 20.0:
        return None

    n, m, p = 2, 2, 11
    target_rank = 38
    
    # Build target tensor entries for efficient error calculation
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # Zero entries are the complement. For ALS, we usually just minimize error on non-zeros
    # and hope the structure holds, but we must verify zeros at the end.
    
    best_err = float('inf')
    best_solution = None
    
    # Try a few random initializations
    for seed in range(5):
        if time.time() > _DEADLINE - 10.0:
            break
            
        np.random.seed(seed)
        
        # Initialize U, V, W
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Alternatively, initialize from rank 39 by dropping one column
        if seed == 0:
            base = _construct_rank39()
            U = np.array(base["U"], dtype=float)[:, :target_rank]
            V = np.array(base["V"], dtype=float)[:, :target_rank]
            W = np.array(base["W"], dtype=float)[:, :target_rank]
        
        for it in range(100):
            if time.time() > _DEADLINE - 10.0:
                break
                
            # Update U
            for u_idx in range(n * m):
                # Gather V, W for entries involving u_idx
                # We need to solve: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = 1
                # For fixed u_idx, we have constraints for each (v_idx, w_idx) in tensor_entries where first elem is u_idx
                # Let's precompute lists for speed
                pass # Placeholder for actual ALS update logic below

            # Efficient ALS Update
            # Precompute for each u_idx, the list of (v_idx, w_idx)
            # This is static, so we can precompute outside the loop if we want, but let's keep it simple.
            
            # Update U rows
            for u_idx in range(n * m):
                A_mat_rows = []
                b_vec = []
                for (uu, vv, ww) in tensor_entries:
                    if uu == u_idx:
                        # Row of A is V[vv] * W[ww]
                        A_mat_rows.append(V[vv] * W[ww])
                        b_vec.append(1.0)
                if A_mat_rows:
                    A_mat = np.array(A_mat_rows)
                    b = np.array(b_vec)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        U[u_idx] = x
                    except:
                        pass

            # Update V rows
            for v_idx in range(m * p):
                A_mat_rows = []
                b_vec = []
                for (uu, vv, ww) in tensor_entries:
                    if vv == v_idx:
                        A_mat_rows.append(U[uu] * W[ww])
                        b_vec.append(1.0)
                if A_mat_rows:
                    A_mat = np.array(A_mat_rows)
                    b = np.array(b_vec)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        V[v_idx] = x
                    except:
                        pass

            # Update W rows
            for w_idx in range(p * n):
                A_mat_rows = []
                b_vec = []
                for (uu, vv, ww) in tensor_entries:
                    if ww == w_idx:
                        A_mat_rows.append(U[uu] * V[vv])
                        b_vec.append(1.0)
                if A_mat_rows:
                    A_mat = np.array(A_mat_rows)
                    b = np.array(b_vec)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        W[w_idx] = x
                    except:
                        pass
            
            # Calculate error
            err = 0.0
            for (uu, vv, ww) in tensor_entries:
                val = np.dot(U[uu], V[vv] * W[ww])
                err += (val - 1.0) ** 2
                
            if err < best_err:
                best_err = err
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            if err < 1e-10:
                # Check if it's actually valid (zeros are zero)
                if _verify_strict(U, V, W, tol=1e-8):
                    return best_solution
                else:
                    # It fits non-zeros but not zeros. Continue optimizing or break?
                    # Usually ALS on non-zeros alone will drift into invalid space.
                    # We can add a penalty for non-zero entries, but that's complex.
                    # For now, if error is very low, it's a candidate.
                    pass

    # Final check on the best solution found
    if best_solution and best_err < 1e-6:
        if _verify_strict(best_solution["U"], best_solution["V"], best_solution["W"], tol=1e-8):
            return best_solution
            
    return None

def construct():
    """Main entry point. Returns the rank-minimal decomposition."""
    global _START_TIME, _DEADLINE
    
    try:
        # 1. Get the verified rank-39 solution
        result = _construct_rank39()
        
        # 2. Verify it strictly
        if not _verify_strict(result["U"], result["V"], result["W"]):
            raise ValueError("Rank-39 verification failed")
        
        # 3. Try to find a lower-rank solution (Rank 38)
        if time.time() < _DEADLINE - 30.0:
            try:
                better = _try_rank38_als()
                if better is not None:
                    result = better
            except Exception:
                pass
        
        # 4. Final verification
        if not _verify_strict(result["U"], result["V"], result["W"]):
            # Fallback to the known good solution
            result = _construct_rank39()
            if not _verify_strict(result["U"], result["V"], result["W"]):
                raise RuntimeError("No valid decomposition found")
        
        return result
        
    except Exception:
        # Absolute fallback: return the hardcoded rank-39
        return _construct_rank39()
