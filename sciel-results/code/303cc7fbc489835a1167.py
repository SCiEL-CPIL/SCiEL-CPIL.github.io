import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)

    def try_rank_als(rank, max_iters=100, tol=1e-10):
        """Try to find a decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + rank)
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Normalize to prevent divergence
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                scale = (1.0 / norm) ** (1.0/3.0)
                U[:, r] *= scale
                V[:, r] *= scale
                W[:, r] *= scale
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # T_u = einsum('abc,br,cr->ar', target, V, W)
            T_u = np.einsum('abc,br,cr->ar', target, V, W)
            # Gram matrix for U: G_u = (V^T V) * (W^T W) element-wise
            G_u = (V.T @ V) * (W.T @ W)
            G_u += 1e-8 * np.eye(rank)
            U_new = np.linalg.solve(G_u, T_u.T).T
            
            # Update V
            T_v = np.einsum('abc,ar,cr->br', target, U_new, W)
            G_v = (U_new.T @ U_new) * (W.T @ W)
            G_v += 1e-8 * np.eye(rank)
            V_new = np.linalg.solve(G_v, T_v.T).T
            
            # Update W
            T_w = np.einsum('abc,ar,br->cr', target, U_new, V_new)
            G_w = (U_new.T @ U_new) * (V_new.T @ V_new)
            G_w += 1e-8 * np.eye(rank)
            W_new = np.linalg.solve(G_w, T_w.T).T
            
            # Check convergence
            err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if err < tol:
                break
        
        # Snap to nearest half-integer
        u_snap = np.round(U * 2.0) / 2.0
        v_snap = np.round(V * 2.0) / 2.0
        w_snap = np.round(W * 2.0) / 2.0
        
        if verify_decomposition(u_snap, v_snap, w_snap):
            return (u_snap, v_snap, w_snap)
            
        # Try cleaning small values
        u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
        v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
        w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
        if verify_decomposition(u_clean, v_clean, w_clean):
            return (u_clean, v_clean, w_clean)
            
        # Try snapping to integers only
        u_int = np.round(U)
        v_int = np.round(V)
        w_int = np.round(W)
        if verify_decomposition(u_int, v_int, w_int):
            return (u_int, v_int, w_int)
            
        return None

    def try_block_structure():
        """
        Exploit block structure: treat 2x5 matrix as two 1x5 row vectors.
        Each 1x5 * 5x5 multiplication has a known rank-5 decomposition.
        Stack the two solutions to get total rank 10.
        
        For a 1x5 vector v and 5x5 matrix A, the product is a 1x5 vector.
        The tensor for this is rank 5 (each output element is a linear combination).
        
        Actually, for 1xM * MxP, the rank is M*P = 25 for general case,
        but if we're just doing row-vector * matrix, each of the P output
        elements depends on M inputs, so we need M*P = 25 terms.
        
        Wait, let me reconsider. The standard rank for 1xM * MxP is M*P.
        But we can do better with structured approaches.
        
        Actually, for our specific case:
        - We have a 2x5 matrix times a 5x5 matrix.
        - This gives a 2x5 output matrix.
        - Element (i,k) of output = sum_j A[i,j] * B[j,k]
        
        For each fixed i (0 or 1), we're computing a 1x5 * 5x5 product.
        The rank for each such product is 5*5 = 25 in general.
        But we can share structure between the two rows.
        
        Let's try a different approach: use the fact that the two rows
        are independent, so we can solve two separate 1x5*5x5 problems.
        
        For a 1x5 * 5x5 multiplication, the minimal rank is actually 5*5 = 25
        for the general case, but there might be structure we can exploit.
        
        Actually, I realize the block structure approach as stated may not
        immediately give rank 10. Let me think about this differently.
        
        The key insight: if we can find a rank-r decomposition for 1x5*5x5,
        then we can use rank 2r for 2x5*5x5 by stacking.
        
        But the minimal rank for 1x5*5x5 is 25 (it's just computing 5 dot products
        of length 5 each, which are independent).
        
        So this approach would give rank 50, same as schoolbook. Not helpful.
        
        Let me try a different structured approach: exploit that n=2 is small.
        """
        if time.time() > deadline:
            return None
        
        # Try to find a low-rank decomposition that exploits the n=2 structure
        # by using a shared V matrix pattern
        
        # The tensor has structure: T[i*m+j, j*p+k, k*n+i] = 1
        # For n=2: T[i*5+j, j*5+k, k*2+i] = 1
        
        # Let's try rank 25 (half of schoolbook) with structured initialization
        # where the two rows share some structure
        
        for rank in [25, 30, 35]:
            if time.time() > deadline:
                break
            
            # Structured initialization: try to make columns come in pairs
            # corresponding to the two rows
            np.random.seed(100 + rank)
            
            # Initialize with structure: for each "base" column, create two related columns
            half_rank = rank // 2
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            for r in range(half_rank):
                # Row 0 component
                U[:5, 2*r] = np.random.randn(5)
                V[:, 2*r] = np.random.randn(25)
                W[::2, 2*r] = np.random.randn(5)  # Only even indices for row 0
                
                # Row 1 component
                U[5:, 2*r+1] = np.random.randn(5)
                V[:, 2*r+1] = np.random.randn(25)
                W[1::2, 2*r+1] = np.random.randn(5)  # Only odd indices for row 1
            
            if rank % 2 == 1:
                U[:, rank-1] = np.random.randn(dim_u)
                V[:, rank-1] = np.random.randn(dim_v)
                W[:, rank-1] = np.random.randn(dim_w)
            
            # Normalize
            for r in range(rank):
                norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
                if norm > 0:
                    scale = (1.0 / norm) ** (1.0/3.0)
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Update U
                T_u = np.einsum('abc,br,cr->ar', target, V, W)
                G_u = (V.T @ V) * (W.T @ W)
                G_u += 1e-8 * np.eye(rank)
                U_new = np.linalg.solve(G_u, T_u.T).T
                
                # Update V
                T_v = np.einsum('abc,ar,cr->br', target, U_new, W)
                G_v = (U_new.T @ U_new) * (W.T @ W)
                G_v += 1e-8 * np.eye(rank)
                V_new = np.linalg.solve(G_v, T_v.T).T
                
                # Update W
                T_w = np.einsum('abc,ar,br->cr', target, U_new, V_new)
                G_w = (U_new.T @ U_new) * (V_new.T @ V_new)
                G_w += 1e-8 * np.eye(rank)
                W_new = np.linalg.solve(G_w, T_w.T).T
                
                # Check convergence
                err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if err < 1e-10:
                    break
            
            # Try to snap to half-integers
            u_snap = np.round(U * 2.0) / 2.0
            v_snap = np.round(V * 2.0) / 2.0
            w_snap = np.round(W * 2.0) / 2.0
            
            if verify_decomposition(u_snap, v_snap, w_snap):
                return (u_snap, v_snap, w_snap)
            
            # Try cleaning
            u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
            v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
            w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
            if verify_decomposition(u_clean, v_clean, w_clean):
                return (u_clean, v_clean, w_clean)
        
        return None

    best_result = None
    
    # First try the block structure approach
    if time.time() < deadline:
        result = try_block_structure()
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If that didn't work, try standard ALS with lower ranks
    if best_result is None:
        for target_rank in [25, 30, 32, 35, 40]:
            if time.time() > deadline:
                break
            
            result = try_rank_als(target_rank, max_iters=200)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
