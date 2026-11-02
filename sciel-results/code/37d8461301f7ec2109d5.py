"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But structural constraints may require more
    # Try from 24 upward
    
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    # Schoolbook fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for rank in range(24, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts for each rank
        max_restarts = max(1, int((deadline - time.time()) / 5))
        if max_restarts < 1:
            break
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 1000 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Alternating least squares
            success = False
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each column c, U[:,c] should minimize ||T - sum U[:,c]⊗V[:,c]⊗W[:,c]||
                # This is a linear least squares problem
                # Residual R = T - sum_{c' != c} U[:,c']⊗V[:,c']⊗W[:,c']
                # Then U[:,c] = argmin ||R - U[:,c]⊗V[:,c]⊗W[:,c]||
                
                # Compute residual without column c
                # R[a,b,d] = T[a,b,d] - sum_{c'!=c} U[a,c']*V[b,c']*W[d,c']
                
                # Efficient: compute sum of all, then subtract column c
                # sum_all[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
                
                # For updating U[:,c]:
                # R[a,b,d] = T[a,b,d] - U[a,c]*V[b,c]*W[d,c] - sum_{c'!=c} U[a,c']*V[b,c']*W[d,c']
                # = T[a,b,d] - sum_{c'} U[a,c']*V[b,c']*W[d,c']
                # Let S[a,b,d] = sum_c' U[a,c']*V[b,c']*W[d,c'] (full sum)
                # Then R = T - S + U[:,c]⊗V[:,c]⊗W[:,c]
                # Minimize ||T - S + U[:,c]⊗V[:,c]⊗W[:,c]||_F over U[:,c]
                # = ||T - S + U_c ⊗ (V_c ⊗ W_c)||_F
                # Reshape: T-S is (n*m, m*p, p*n), V_c⊗W_c is (m*p, p*n)
                # So we need: U_c = argmin ||(T-S).reshape(n*m, m*p*p*n) - U_c @ (V_c⊗W_c).reshape(m*p*p*n)||
                
                # Actually, let's use a simpler approach: vectorize
                # T is (A, B, C) where A=n*m, B=m*p, C=p*n
                # For fixed V[:,c] and W[:,c], the contribution is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # Reshape to 2D: (A, B*C), the term is U[:,c] @ (V[:,c]⊗W[:,c])^T... no
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed r=c, varying U[:,c]: T[a,b,c_idx] = U[a,c]*V[b,c]*W[c_idx,c]
                # So the residual for fixing other columns:
                # R[a,b,c_idx] = T[a,b,c_idx] - U[a,c]*V[b,c]*W[c_idx,c] - sum_{r!=c} U[a,r]*V[b,r]*W[c_idx,r]
                
                # Let's compute the full contraction sum_c U[a,c]*V[b,c]*W[d,c]
                # = einsum('ac,bc,dc->abd', U, V, W)
                
                # For updating U[:,c]:
                # Minimize ||T - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||
                # = ||T - U[:,c]⊗V[:,c]⊗W[:,c] - sum_{r!=c} ...||
                # Let R = T - sum_{r!=c} U[:,r]⊗V[:,r]⊗W[:,r]
                # Minimize ||R - U[:,c]⊗V[:,c]⊗W[:,c]||
                # = ||R.reshape(A, B*C) - U[:,c] @ (V[:,c]⊗W[:,c]).reshape(B*C)||
                # Wait, that's not right either.
                # R[a,b,d] - U[a,c]*V[b,c]*W[d,c]
                # = R[a,b,d] - U[a,c] * (V[b,c]*W[d,c])
                # Let M[b,d] = V[b,c]*W[d,c], shape (B,C)
                # Then R[a,b,d] - U[a,c]*M[b,d]
                # Reshape to (A, B*C): R.reshape(A, B*C) - U[:,c] @ M.reshape(B*C)
                # This is a linear system: U[:,c] = argmin ||R_2d - U[:,c] @ M_1d||
                # = R_2d @ M_1d^T / (M_1d @ M_1d^T) ... no
                # ||R_2d - U[:,c] @ M_1d||_F where M_1d is (B*C,)
                # = ||R_2d - outer(U[:,c], M_1d)||_F
                # This is: U[:,c] = R_2d @ M_1d / (M_1d @ M_1d)
                # Wait, that gives a scalar per row... Let me think again.
                # ||R_2d - u @ m||_F^2 = ||R_2d||^2 - 2*u^T R_2d^T m + ||u||^2 ||m||^2
                # d/du = -2 R_2d^T m + 2 u ||m||^2 = 0
                # u = R_2d^T m / ||m||^2
                # But R_2d^T m is a vector of length A.
                # R_2d^T m[a] = sum_{b,c_idx} R[a,b,c_idx] * M[b,c_idx]
                # = sum_{b,d} R[a,b,d] * V[b,c] * W[d,c]
                
                # So U[a,c] = sum_{b,d} R[a,b,d] * V[b,c] * W[d,c] / sum_{b,d} V[b,c]^2 * W[d,c]^2
                
                # Similarly for V and W.
                
                # Let's implement this:
                
                # Compute full sum S = einsum('ac,bc,dc->abd', U, V, W)
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                R_full = T - S  # residual
                
                # Update U
                for c in range(rank):
                    # R[a,b,d] = T[a,b,d] - S[a,b,d] + U[a,c]*V[b,c]*W[d,c]
                    # = R_full[a,b,d] + U[a,c]*V[b,c]*W[d,c]
                    # We want: U[a,c] = sum_{b,d} (R_full[a,b,d] + U[a,c]*V[b,c]*W[d,c]) * V[b,c] * W[d,c] / denom
                    # But this has U[a,c] on both sides. Let me redo.
                    # Minimize ||R_full + U[:,c]⊗V[:,c]⊗W[:,c]|| (since S includes the c-th term)
                    # Wait, S = sum_r U[:,r]⊗V[:,r]⊗W[:,r], so R_full = T - S
                    # We want to minimize ||R_full + U[:,c]⊗V[:,c]⊗W[:,c]|| (removing old c, adding new)
                    # No wait. If we change U[:,c], S changes. Let me define:
                    # S_old = sum_r U[:,r]⊗V[:,r]⊗W[:,r]
                    # If we update U[:,c] to U_new[:,c], S_new = S_old - U[:,c]⊗V[:,c]⊗W[:,c] + U_new[:,c]⊗V[:,c]⊗W[:,c]
                    # Minimize ||T - S_new|| = ||T - S_old + U[:,c]⊗V[:,c]⊗W[:,c] - U_new[:,c]⊗V[:,c]⊗W[:,c]||
                    # = ||R_full + U[:,c]⊗V[:,c]⊗W[:,c] - U_new[:,c]⊗V[:,c]⊗W[:,c]||
                    # Let A_mat = V[:,c]⊗W[:,c] reshaped, so the term is outer(U_new[:,c], A_mat)
                    # Minimize ||B - outer(u, m)||_F where B = R_full + U[:,c]⊗V[:,c]⊗W[:,c]
                    # B[a,b,d] = R_full[a,b,d] + U[a,c]*V[b,c]*W[d,c]
                    # u = B_2d^T m / ||m||^2 where m = V[:,c]⊗W[:,c] reshaped
                    # u[a] = sum_{b,d} B[a,b,d] * V[b,c] * W[d,c] / sum_{b,d} V[b,c]^2 * W[d,c]^2
                    
                    B = R_full + np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    numerator = np.einsum('abd,b,d->a', B, V[:, c], W[:, c])
                    denom = np.sum(V[:, c]**2) * np.sum(W[:, c]**2)
                    if denom > 1e-12:
                        U[:, c] = numerator / denom
                
                # Recompute S and R_full
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                R_full = T - S
                
                # Update V
                for c in range(rank):
                    B = R_full + np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    numerator = np.einsum('abd,a,d->b', B, U[:, c], W[:, c])
                    denom = np.sum(U[:, c]**2) * np.sum(W[:, c]**2)
                    if denom > 1e-12:
                        V[:, c] = numerator / denom
                
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                R_full = T - S
                
                # Update W
                for c in range(rank):
                    B = R_full + np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    numerator = np.einsum('abd,a,b->d', B, U[:, c], V[:, c])
                    denom = np.sum(U[:, c]**2) * np.sum(V[:, c]**2)
                    if denom > 1e-12:
                        W[:, c] = numerator / denom
                
                # Check convergence
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(T - S)
                if err < 1e-8:
                    success = True
                    break
                
                if iteration % 20 == 0 and iteration > 0:
                    # Check if error is decreasing
                    pass
            
            if success:
                # Verify
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                if np.linalg.norm(T - S) < 1e-6:
                    # Try to round to integers or half-integers
                    # Check if values are close to integers or half-integers
                    all_close = True
                    for mat in [U, V, W]:
                        rounded = np.round(mat * 2) / 2
                        if np.linalg.norm(mat - rounded) > 1e-6:
                            all_close = False
                            break
                    if all_close:
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                    else:
                        # Use as-is (floats are acceptable per problem statement)
                        pass
                    
                    best_result = (U, V, W)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists
    U, V, W = best_result
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
