"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Initialize schoolbook tensor T[i, j, k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    T_remaining = T.copy()
    
    # Lists to store the discovered factors
    U_cols = []
    V_cols = []
    W_cols = []
    
    def time_left():
        return deadline - time.time()
    
    # Greedy rank-1 peeling
    # We iteratively find the best rank-1 approximation u, v, w to T_remaining
    # such that T_remaining - u*v*w is as small as possible (ideally zero if rank decreases)
    # Since we want exact integer/half-integer decompositions, we look for patterns.
    # However, a generic greedy approach on floating point might drift.
    # Let's try a structured search over small integer vectors first, or use SVD.
    # SVD of the matricized tensor gives the best rank-1 approximation in Frobenius norm.
    
    while time_left() > 10.0:
        # Matricize T_remaining to find best rank-1 approximation
        # Flatten to 3D array.
        # Mode-1 matricization: (n x m*p)
        M1 = T_remaining.reshape(n, m * p)
        # Mode-2 matricization: (m x n*p)
        M2 = T_remaining.reshape(m, n * p)
        # Mode-3 matricization: (p x n*m)
        M3 = T_remaining.reshape(p, n * m)
        
        # Get singular vectors
        try:
            U1, S1, Vt1 = np.linalg.svd(M1, full_matrices=False)
            U2, S2, Vt2 = np.linalg.svd(M2, full_matrices=False)
            U3, S3, Vt3 = np.linalg.svd(M3, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        
        u = U1[:, 0]
        v = Vt1[0, :].reshape(m, p)
        w = Vt1[0, :].reshape(m, p) # Wait, Vt1 is (m*p, m*p) or similar?
        # SVD of M1 (n x m*p): U1 (n, n), S1 (n,), Vt1 (n, m*p) if n < m*p.
        # Actually, np.linalg.svd returns U, s, Vh.
        # M1 shape (n, m*p). U1 (n, k), s (k,), Vt1 (k, m*p). k = min(n, m*p) = n.
        # So u is U1[:, 0].
        # The corresponding vector in the other modes needs to be consistent.
        # It is better to project the current best guess or use a fixed-point iteration.
        
        # Let's try a simpler heuristic:
        # If the tensor has low rank, SVD of any matricization gives the principal components.
        # u = U1[:, 0]
        # v_flat = Vt1[0, :] -> shape (m*p,)
        # w_flat = Vt1[0, :] -> shape (m*p,)
        # This doesn't directly give v and w separately.
        
        # Alternative: Use the fact that for a rank-1 tensor a*b*c,
        # any non-zero slice is proportional.
        # Let's pick a random non-zero entry in T_remaining, say T[i0, j0, k0].
        # Then u[i0] * v[j0] * w[k0] = T[i0, j0, k0].
        # We can normalize u such that u[i0]=1 (if T[i0,j0,k0]!=0).
        # Then v[j0] * w[k0] = T[i0, j0, k0].
        # And for any i, j, k: T[i, j, k] = u[i] * v[j] * w[k].
        # If we fix u[i0]=1, then v[j] = T[i0, j, k0] / w[k0] and w[k] = T[i0, j0, k] / v[j0].
        # This is circular.
        
        # Let's use the SVD vectors from all three modes to form a candidate.
        # u from M1 SVD, v from M2 SVD, w from M3 SVD.
        # Check if u*v*w approximates T_remaining.
        
        u = U1[:, 0]
        v = U2[:, 0]
        w = U3[:, 0]
        
        # Normalize to match scale
        # T_remaining[i, j, k] approx u[i] v[j] w[k]
        # Let's compute the error
        approx = np.einsum('i,j,k->ijk', u, v, w)
        err = np.linalg.norm(T_remaining - approx)
        
        # If the error is very small, we have found a rank-1 component.
        if err < 1e-9:
            # Add this component
            # We need to ensure entries are half-integers or integers.
            # SVD gives floats. We might need to snap to half-integers.
            # For now, let's just record the floats. The evaluator accepts half-integers.
            # We can scale u, v, w to make entries nicer if needed, but exactness is key.
            # If T_remaining is exactly rank-1 with rational entries, SVD should recover them (up to sign).
            
            # Snap to half-integers?
            # u_snapped = np.round(u * 2) / 2
            # v_snapped = np.round(v * 2) / 2
            # w_snapped = np.round(w * 2) / 2
            # Check if snapped version also has low error.
            # If so, use snapped.
            
            u_s = np.round(u * 2) / 2
            v_s = np.round(v * 2) / 2
            w_s = np.round(w * 2) / 2
            approx_s = np.einsum('i,j,k->ijk', u_s, v_s, w_s)
            err_s = np.linalg.norm(T_remaining - approx_s)
            
            if err_s < 1e-9:
                U_cols.append(u_s)
                V_cols.append(v_s)
                W_cols.append(w_s)
                T_remaining = T_remaining - approx_s
            else:
                # Use float vectors? The task requires integer/half-integer entries.
                # If we can't snap, we might fail.
                # Let's try to scale to make one entry 1 or 0.5.
                # For simplicity, if snapping fails, we might skip or fail.
                # Given the small dimensions, exact rational decomposition should be findable.
                pass
            
            # Check if T_remaining is zero
            if np.linalg.norm(T_remaining) < 1e-9:
                break
        else:
            # The current SVD vectors don't form a rank-1 component that exactly matches.
            # This might mean the tensor is not rank-1, or SVD is not giving the right basis.
            # For a sum of rank-1 tensors, peeling works if we can identify one.
            # If greedy peeling fails, we might be stuck.
            # Let's try a different strategy: 
            # If the tensor is small, we can try to solve for u, v, w directly from equations.
            # But that's complex.
            
            # Let's just break if we can't make progress to avoid infinite loop.
            # Or try a few random restarts?
            # For now, break.
            break
    
    # If we didn't finish, return the best we have (which might be incomplete/invalid)
    # But we must return a valid decomposition.
    # If T_remaining is not zero, our decomposition is incomplete.
    # We should fall back to a known valid decomposition (e.g., schoolbook) if we fail.
    
    if np.linalg.norm(T_remaining) > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Construct output matrices
    rank = len(U_cols)
    if rank == 0:
        # Edge case: T was zero? Not possible for mat mult.
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
        
    U_mat = np.column_stack(U_cols).reshape(n * m, rank)
    V_mat = np.column_stack(V_cols).reshape(m * p, rank)
    W_mat = np.column_stack(W_cols).reshape(p * n, rank)
    
    # Convert to lists of lists
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
