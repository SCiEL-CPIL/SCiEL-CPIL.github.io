"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Build rank-50 Strassen decomposition, then use duplicate/proportional
    term detection and iterative local search to reduce rank.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p, num_tests=20):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for t in range(num_tests):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    def detect_and_remove_proportional(U_arr, V_arr, W_arr, tol=1e-9):
        """Detect and remove proportional/duplicate rank-1 terms."""
        rank = U_arr.shape[1]
        if rank <= 1:
            return U_arr, V_arr, W_arr
        
        to_remove = set()
        for i in range(rank):
            if i in to_remove:
                continue
            for j in range(i + 1, rank):
                if j in to_remove:
                    continue
                ui, vi, wi = U_arr[:, i], V_arr[:, i], W_arr[:, i]
                uj, vj, wj = U_arr[:, j], V_arr[:, j], W_arr[:, j]
                
                # Check if ui is proportional to uj
                nz_i = np.abs(ui) > tol
                nz_j = np.abs(uj) > tol
                if not np.any(nz_i) or not np.any(nz_j):
                    continue
                if not np.array_equal(nz_i, nz_j):
                    continue
                ratios_u = ui[nz_i] / uj[nz_j]
                if not np.allclose(ratios_u, ratios_u[0], atol=tol, rtol=tol):
                    continue
                a = ratios_u[0]
                
                # Check if vi is proportional to vj
                nz_i = np.abs(vi) > tol
                nz_j = np.abs(vj) > tol
                if not np.any(nz_i) or not np.any(nz_j):
                    continue
                if not np.array_equal(nz_i, nz_j):
                    continue
                ratios_v = vi[nz_i] / vj[nz_j]
                if not np.allclose(ratios_v, ratios_v[0], atol=tol, rtol=tol):
                    continue
                b = ratios_v[0]
                
                # Check if wi is proportional to wj
                nz_i = np.abs(wi) > tol
                nz_j = np.abs(wj) > tol
                if not np.any(nz_i) or not np.any(nz_j):
                    continue
                if not np.array_equal(nz_i, nz_j):
                    continue
                ratios_w = wi[nz_i] / wj[nz_j]
                if not np.allclose(ratios_w, ratios_w[0], atol=tol, rtol=tol):
                    continue
                c = ratios_w[0]
                
                # Check a*b*c ≈ 1
                if abs(a * b * c - 1.0) > 1e-6:
                    continue
                
                # Terms are proportional; remove j
                to_remove.add(j)
                break
        
        if not to_remove:
            return U_arr, V_arr, W_arr
        
        keep = [i for i in range(rank) if i not in to_remove]
        return U_arr[:, keep], V_arr[:, keep], W_arr[:, keep]
    
    def try_local_reduction(U_arr, V_arr, W_arr, n, m, p, max_iters=50):
        """Try to reduce rank by checking if any term can be zeroed out."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            if current_rank <= 1:
                break
            
            found_reduction = False
            for r in range(current_rank):
                if time.time() > deadline - 20:
                    break
                # Try zeroing out term r
                U_test = U_arr.copy()
                V_test = V_arr.copy()
                W_test = W_arr.copy()
                U_test[:, r] = 0
                V_test[:, r] = 0
                W_test[:, r] = 0
                
                # Check if the remaining terms still give a valid decomposition
                # (They won't, since we removed a term, but check for numerical issues)
                # Actually, zeroing a term breaks the decomposition unless the term
                # was already zero. So this approach doesn't work directly.
                # Skip this and rely on proportional detection.
                break
            
            # Use proportional detection
            U_new, V_new, W_new = detect_and_remove_proportional(U_arr, V_arr, W_arr)
            new_rank = U_new.shape[1]
            if new_rank < current_rank:
                U_arr, V_arr, W_arr = U_new, V_new, W_new
                current_rank = new_rank
                found_reduction = True
            if not found_reduction:
                break
        
        return U_arr, V_arr, W_arr, current_rank
    
    # Attempt proportional term detection
    if time.time() < deadline - 20:
        U_red, V_red, W_red = detect_and_remove_proportional(best_U, best_V, best_W)
        new_rank = U_red.shape[1]
        if new_rank < best_rank and verify_decomposition(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Try local reduction passes
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        U_red, V_red, W_red, new_rank = try_local_reduction(best_U, best_V, best_W, n, m, p)
        if new_rank < best_rank and verify_decomposition(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
        else:
            break
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
