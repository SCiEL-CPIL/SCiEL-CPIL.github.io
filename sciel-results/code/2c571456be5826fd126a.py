import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all 0s
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s) > tol:
                        return False
        return True

    def verify_fast(U_mat, V_mat, W_mat, tol=1e-8):
        """Faster verification using vectorized operations"""
        r = U_mat.shape[1]
        # Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a 3-tensor. We can compute it as:
        # T = sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
        # But that's expensive. Instead, for each r, compute the outer product and accumulate.
        # Or use einsum.
        T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
        
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > tol:
                        return False
        
        # Check all 0s
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    if abs(T[a, b, c]) > tol:
                        return False
        return True

    # 1. Construct Rank 42 (Strassen blocks)
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # A indices
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        # B indices
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        # W indices
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U42[a00, offset] += 1.0; U42[a11, offset] += 1.0
        V42[b00, offset] += 1.0; V42[b11, offset] += 1.0
        W42[w00, offset] += 1.0; W42[w11, offset] += 1.0

        # Term 1: M2 = (A10+A11)B00
        U42[a10, offset+1] += 1.0; U42[a11, offset+1] += 1.0
        V42[b00, offset+1] += 1.0
        W42[w10, offset+1] += 1.0; W42[w11, offset+1] -= 1.0

        # Term 2: M3 = A00(B01-B11)
        U42[a00, offset+2] += 1.0
        V42[b01, offset+2] += 1.0; V42[b11, offset+2] -= 1.0
        W42[w01, offset+2] += 1.0; W42[w11, offset+2] += 1.0

        # Term 3: M4 = A11(B10-B00)
        U42[a11, offset+3] += 1.0
        V42[b10, offset+3] += 1.0; V42[b00, offset+3] -= 1.0
        W42[w00, offset+3] += 1.0; W42[w10, offset+3] += 1.0

        # Term 4: M5 = (A00+A01)B11
        U42[a00, offset+4] += 1.0; U42[a01, offset+4] += 1.0
        V42[b11, offset+4] += 1.0
        W42[w00, offset+4] -= 1.0; W42[w01, offset+4] += 1.0

        # Term 5: M6 = (A10-A00)(B00+B01)
        U42[a10, offset+5] += 1.0; U42[a00, offset+5] -= 1.0
        V42[b00, offset+5] += 1.0; V42[b01, offset+5] += 1.0
        W42[w11, offset+5] += 1.0

        # Term 6: M7 = (A01-A11)(B10+B11)
        U42[a01, offset+6] += 1.0; U42[a11, offset+6] -= 1.0
        V42[b10, offset+6] += 1.0; V42[b11, offset+6] += 1.0
        W42[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    if not verify_fast(U42, V42, W42):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank from 42 using merging
    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42
    
    def try_merge(U_mat, V_mat, W_mat):
        """Try to find and apply one merge. Returns (success, new_U, new_V, new_W, new_rank)"""
        current_rank = U_mat.shape[1]
        if current_rank <= 20:
            return False, U_mat, V_mat, W_mat, current_rank
        
        # Find pairs (c, d) with c < d such that U[:,c] ~ U[:,d] and V[:,c] ~ V[:,d]
        for i in range(current_rank):
            for j in range(i + 1, current_rank):
                # Check proportionality of U
                nz_u_i = np.nonzero(U_mat[:, i])[0]
                nz_u_j = np.nonzero(U_mat[:, j])[0]
                if len(nz_u_i) == 0 and len(nz_u_j) == 0:
                    continue
                if len(nz_u_i) == 0 or len(nz_u_j) == 0:
                    continue
                if not np.array_equal(nz_u_i, nz_u_j):
                    continue
                ratios_u = U_mat[nz_u_i, i] / U_mat[nz_u_i, j]
                if np.std(ratios_u) > 1e-10:
                    continue
                s = ratios_u[0]
                
                # Check proportionality of V
                nz_v_i = np.nonzero(V_mat[:, i])[0]
                nz_v_j = np.nonzero(V_mat[:, j])[0]
                if len(nz_v_i) == 0 and len(nz_v_j) == 0:
                    continue
                if len(nz_v_i) == 0 or len(nz_v_j) == 0:
                    continue
                if not np.array_equal(nz_v_i, nz_v_j):
                    continue
                ratios_v = V_mat[nz_v_i, i] / V_mat[nz_v_i, j]
                if np.std(ratios_v) > 1e-10:
                    continue
                t = ratios_v[0]
                
                # Merge: U_i = s U_j, V_i = t V_j
                # So U_j V_j = (1/s)(1/t) U_i V_i
                # Term j: U_j V_j W_j = (1/(st)) U_i V_i W_j
                # Add (1/(st)) W_j to W_i, remove column j
                scale = 1.0 / (s * t)
                new_U = U_mat.copy()
                new_V = V_mat.copy()
                new_W = W_mat.copy()
                new_W[:, i] += scale * new_W[:, j]
                
                # Remove column j
                cols = [k for k in range(current_rank) if k != j]
                new_U = new_U[:, cols]
                new_V = new_V[:, cols]
                new_W = new_W[:, cols]
                
                if verify_fast(new_U, new_V, new_W):
                    return True, new_U, new_V, new_W, current_rank - 1
        
        return False, U_mat, V_mat, W_mat, current_rank
    
    # Iteratively merge
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        success, best_U, best_V, best_W, best_rank = try_merge(best_U, best_V, best_W)
        if not success:
            break
    
    if best_rank < 42 and verify_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try random search for lower rank
    # We'll try to find a rank R decomposition by random initialization and refinement
    # This is heuristic and may not succeed
    
    def random_search(rank_target, max_iters=1000):
        """Try to find a rank rank_target decomposition using random search"""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random vectors
        U_rand = np.random.randn(nm, rank_target)
        V_rand = np.random.randn(mp, rank_target)
        W_rand = np.random.randn(pn, rank_target)
        
        # Compute current tensor
        T_current = np.einsum('ar,br,cr->abc', U_rand, V_rand, W_rand, optimize=True)
        
        # Target tensor
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        # Gradient descent to minimize ||T_current - T_target||^2
        # This is expensive, so we'll do a few iterations
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Compute gradient
            # dL/dU[a,r] = sum_{b,c} (T_current[a,b,c] - T_target[a,b,c]) * V[b,r] * W[c,r]
            # Similarly for V and W
            
            diff = T_current - T_target
            
            # Gradient for U
            grad_U = np.einsum('abc,br,cr->ar', diff, V_rand, W_rand, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', diff, U_rand, W_rand, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', diff, U_rand, V_rand, optimize=True)
            
            # Update
            U_rand -= lr * grad_U
            V_rand -= lr * grad_V
            W_rand -= lr * grad_W
            
            # Recompute tensor
            T_current = np.einsum('ar,br,cr->abc', U_rand, V_rand, W_rand, optimize=True)
            
            # Check convergence
            err = np.linalg.norm(diff)
            if err < 1e-6:
                return U_rand, V_rand, W_rand
        
        return None
    
    # Try random search with decreasing rank targets
    for target_rank in range(30, 20, -1):
        if time.time() > deadline - 15:
            break
        
        result = random_search(target_rank, max_iters=500)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_fast(U_r, V_r, W_r, tol=1e-5):
                # Polish with exact arithmetic if needed
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Return best found
    if verify_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Final fallback
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
