"""Systematic rank-1 peeling with HOOI and half-integer snapping for 2x3x5 matrix multiplication."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_rank = n * m * p  # 30
    rank = 0
    
    # HOOI function to find best rank-1 approximation
    def hooi(tensor, max_iter=100):
        # Initialize with random or SVD-based
        # SVD based initialization is better for determinism
        u = np.ones(tensor.shape[0]) / np.sqrt(tensor.shape[0])
        v = np.ones(tensor.shape[1]) / np.sqrt(tensor.shape[1])
        w = np.ones(tensor.shape[2]) / np.sqrt(tensor.shape[2])
        
        for _ in range(max_iter):
            # Update u
            u_new = np.einsum('ijk,j,k->i', tensor, v, w)
            # Update v
            v_new = np.einsum('ijk,i,k->j', tensor, u, w)
            # Update w
            w_new = np.einsum('ijk,i,j->k', tensor, u, v)
            
            # Normalize to prevent divergence
            if np.linalg.norm(u_new) > 1e-10: u = u_new / np.linalg.norm(u_new)
            else: break
            if np.linalg.norm(v_new) > 1e-10: v = v_new / np.linalg.norm(v_new)
            else: break
            if np.linalg.norm(w_new) > 1e-10: w = w_new / np.linalg.norm(w_new)
            else: break
            
        # Scale factors so that the product of norms is 1? 
        # Or just return the unnormalized vectors?
        # For peeling, we want to subtract u⊗v⊗w.
        # The optimal scaling is such that the inner product with the tensor is maximized?
        # Actually, if u,v,w are eigenvectors, the optimal scaling is:
        # s = <tensor, u⊗v⊗w>
        # Then the best rank-1 approx is (s / (||u|| ||v|| ||w||)) * u⊗v⊗w?
        # No, the best rank-1 approx is u⊗v⊗w scaled by the singular value.
        # Let's compute the scalar factor.
        val = np.einsum('ijk,i,j,k', tensor, u, v, w)
        # The rank-1 tensor is val * u⊗v⊗w ? No.
        # If u,v,w are unit vectors, the projection is val * u⊗v⊗w.
        # Let's return u, v, w scaled such that their product is the rank-1 tensor.
        # We can just return u, v, w and the scalar val.
        return u, v, w, val

    while rank < max_rank:
        if np.linalg.norm(residual) < 1e-10:
            break
            
        u, v, w, val = hooi(residual)
        
        # The best rank-1 approximation is val * (u⊗v⊗w) where u,v,w are unit?
        # Wait, HOOI finds unit vectors. The rank-1 tensor is val * u⊗v⊗w.
        # Let's snap val * u, val * v, w? No, we can distribute the scaling.
        # We want to snap the factors to half-integers.
        # Let the factors be U, V, W.
        # We know that U⊗V⊗W should approximate the "corner" of the residual.
        # Since the residual has integer entries (initially), and we subtract half-integers,
        # the residual will have half-integer entries.
        
        # Let's try to find U, V, W with entries in {0, ±0.5, ±1} such that
        # U⊗V⊗W is a "valid" subtraction.
        # A simple heuristic: Snap the unnormalized vectors from HOOI?
        # HOOI returns unit vectors.
        # Let's scale them to have max entry 1.
        u_scale = np.max(np.abs(u))
        v_scale = np.max(np.abs(v))
        w_scale = np.max(np.abs(w))
        
        u_raw = u / u_scale if u_scale > 0 else u
        v_raw = v / v_scale if v_scale > 0 else v
        w_raw = w / w_scale if w_scale > 0 else w
        
        # Snap to nearest 0.5
        U_snap = np.round(u_raw * 2) / 2
        V_snap = np.round(v_raw * 2) / 2
        W_snap = np.round(w_raw * 2) / 2
        
        # Check if this snap is a good approximation of the rank-1 component
        # We want to subtract U_snap⊗V_snap⊗W_snap from residual.
        # But we need to ensure the residual remains "valid".
        # Validity: The new residual should have entries that are multiples of 0.5.
        # And ideally, it should be "simpler".
        
        new_residual = residual - np.einsum('i,j,k->ijk', U_snap, V_snap, W_snap)
        
        # Check if new_residual entries are close to multiples of 0.5
        # Since residual is multiples of 0.5 and U,V,W are multiples of 0.5,
        # new_residual should be multiples of 0.5.
        # Check for numerical noise
        check = np.abs(new_residual * 2 - np.round(new_residual * 2))
        if np.max(check) > 1e-5:
            # Snapping failed, try next rank? Or fallback?
            # For now, if snapping fails, we break and fallback to schoolbook
            # to ensure we return a valid answer.
            break
        
        # Check if new_residual is "simpler" (has more zeros) or zero
        if np.linalg.norm(new_residual) < 1e-10:
            # Success
            U_list.append(U_snap)
            V_list.append(V_snap)
            W_list.append(W_snap)
            rank += 1
            residual = new_residual
            break
        
        # Heuristic: Only accept if the number of non-zero entries decreases
        # or stays the same (if we are just shuffling)
        # For exact peeling, the support should shrink.
        old_nonzero = np.sum(np.abs(residual) > 1e-10)
        new_nonzero = np.sum(np.abs(new_residual) > 1e-10)
        
        if new_nonzero < old_nonzero:
            U_list.append(U_snap)
            V_list.append(V_snap)
            W_list.append(W_snap)
            rank += 1
            residual = new_residual
        else:
            # If support doesn't shrink, this might not be a valid peeling step
            # that leads to a lower rank decomposition.
            # Break and fallback.
            break

    # Fallback to schoolbook if we didn't finish
    if rank >= max_rank or np.linalg.norm(residual) > 1e-10:
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the factor matrices
    U_mat = np.array(U_list).T  # shape (n*m, rank)
    V_mat = np.array(V_list).T  # shape (m*p, rank)
    W_mat = np.array(W_list).T  # shape (p*n, rank)
    
    # Convert to list of lists
    U_frac = U_mat.tolist()
    V_frac = V_mat.tolist()
    W_frac = W_mat.tolist()
    
    # Verify
    U_arr = np.array(U_frac)
    V_arr = np.array(V_frac)
    W_arr = np.array(W_frac)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    
    # Check non-diagonal
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 != i2 or j1 != j2 or k1 != k2:
                                val = np.sum(U_arr[i1*m+j1] * V_arr[j2*p+k2] * W_arr[k2*n+i2])
                                err = abs(val)
                                max_err = max(max_err, err)
    
    if max_err > 1e-6:
        # Fallback
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_frac, "V": V_frac, "W": W_frac}
