"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = get_schoolbook()
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def verify_decomposition(U, V, W, R, tol=1e-10):
        """Check if U, V, W exactly decompose T"""
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(T - S) < tol
    
    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap
    
    def try_als(U, V, W, R, max_iters=80, tol=1e-10, deadline=110.0):
        """Run ALS iterations, return (U, V, W, converged)"""
        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False
            
            # Update U
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                vw_mat = np.outer(V[:, r], W[:, r])
                numerator = np.einsum('ijk,jk->i', T - S, vw_mat)
                denominator = np.sum(vw_mat ** 2)
                if denominator > 1e-10:
                    U[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Update V
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                uw_mat = np.outer(U[:, r], W[:, r])
                numerator = np.einsum('ijk,ik->j', T - S, uw_mat)
                denominator = np.sum(uw_mat ** 2)
                if denominator > 1e-10:
                    V[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Update W
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                uv_mat = np.outer(U[:, r], V[:, r])
                numerator = np.einsum('ijk,ij->k', T - S, uv_mat)
                denominator = np.sum(uv_mat ** 2)
                if denominator > 1e-10:
                    W[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T - S)
            if error < tol:
                return U, V, W, True
            if abs(prev_error - error) < 1e-12:
                return U, V, W, False
            prev_error = error
        
        return U, V, W, False
    
    # Lower bound: max(nm, mp, pn)
    min_rank = max(nm, mp, pn)
    
    # Try structured initializations for various ranks
    # For matrix multiplication, there are known constructions
    # Try rank from min_rank up to schoolbook rank
    
    for R in range(min_rank, rank_school + 1):
        if time.time() > deadline:
            break
        
        # Multiple initialization strategies
        for init_idx in range(15):
            if time.time() > deadline:
                break
            
            np.random.seed(init_idx * 7919 + R * 131)
            
            # Strategy 1: Small random noise
            if init_idx < 5:
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            # Strategy 2: Random with some structure
            elif init_idx < 10:
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
                # Scale to reasonable magnitude
                U = U / np.linalg.norm(U) * np.sqrt(nm)
                V = V / np.linalg.norm(V) * np.sqrt(mp)
                W = W / np.linalg.norm(W) * np.sqrt(pn)
            # Strategy 3: Sparse random
            else:
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Pick random non-zero positions
                    u_idx = np.random.choice(nm, size=max(1, nm//3), replace=False)
                    v_idx = np.random.choice(mp, size=max(1, mp//3), replace=False)
                    w_idx = np.random.choice(pn, size=max(1, pn//3), replace=False)
                    U[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                    V[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                    W[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
            
            U_out, V_out, W_out, converged = try_als(U, V, W, R, max_iters=60, deadline=deadline)
            
            if not converged:
                continue
            
            # Try snapping to half-integers
            U_snap, V_snap, W_snap = snap_to_half_integers(U_out, V_out, W_out)
            
            if verify_decomposition(U_snap, V_snap, W_snap, R):
                best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                break
        
        if best_rank == R:
            break
    
    # Also try a few more ranks if we have time
    if best_rank > min_rank and time.time() < deadline - 5:
        # Try to squeeze out one more rank reduction
        for R in range(max(min_rank, best_rank - 3), best_rank):
            if time.time() > deadline:
                break
            for init_idx in range(5):
                if time.time() > deadline:
                    break
                np.random.seed(99999 + init_idx * 1000 + R)
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                U_out, V_out, W_out, converged = try_als(U, V, W, R, max_iters=40, deadline=deadline)
                if not converged:
                    continue
                U_snap, V_snap, W_snap = snap_to_half_integers(U_out, V_out, W_out)
                if verify_decomposition(U_snap, V_snap, W_snap, R):
                    best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                    break
            if best_rank == R:
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
