"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    # Build the multiplication tensor T[i,j,k] where output index is (k*n+i), input1 is (i*m+j), input2 is (j*p+k)
    # T is a 3D tensor of shape (nm, mp, pn) where T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k]
    # This is a 3-way tensor decomposition (CP decomposition)
    
    deadline = time.time() + 120
    best_result = None
    best_rank = nm * mp * pn  # schoolbook rank as fallback
    
    # Start with schoolbook
    def make_schoolbook():
        rank = nm * mp * pn
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
    
    U_sb, V_sb, W_sb, rank_sb = make_schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try to find lower rank decompositions using optimization
    # Strategy: fix rank r, initialize U,V,W randomly, optimize to minimize ||T - sum_c U_c V_c W_c||^2
    
    def tensor_residual(U, V, W):
        """Compute the residual tensor ||T - sum_c U[:,c] outer V[:,c] outer W[:,c]||^2"""
        # Efficient computation: for each column c, add U[:,c]*V[:,c]*W[:,c] to result
        residual = T.copy()
        for c in range(U.shape[1]):
            residual -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.sum(residual**2)
    
    def try_rank(r, max_iter=2000, tol=1e-12):
        """Try to find rank-r decomposition via alternating least squares / gradient descent."""
        if r >= best_rank:
            return None
        
        # Multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                return None
            
            # Initialize with random values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Normalize to avoid scaling issues
            for c in range(r):
                norm = max(np.linalg.norm(U[:,c]), np.linalg.norm(V[:,c]), np.linalg.norm(W[:,c]), 1e-10)
                U[:,c] /= norm
                V[:,c] /= norm
                W[:,c] /= norm
            
            # Alternating least squares style: fix two, solve for third
            # For fixed V,W: U[:,c] = argmin sum_{i,j,k} (T[i,j,k] - sum_c' U[i,c'] V[j,c'] W[k,c'])^2 * U[i,c]
            # This is a linear system for each c
            
            for it in range(max_iter):
                if time.time() > deadline - 30:
                    return None
                
                # Update U: for each c, solve for U[:,c]
                # The objective involves: for each c, minimize over U[:,c]:
                # sum_{i,j,k} (T[i,j,k] - sum_c' U[i,c'] V[j,c'] W[k,c'])^2
                # = sum_{i,j,k} [T[i,j,k] - U[i,c]*V[j,c]*W[k,c] - sum_{c'!=c} U[i,c']*V[j,c']*W[k,c']]^2
                # This is quadratic in U[:,c]
                
                # Precompute for efficiency
                # For fixed V,W, the residual without column c is:
                # R_c[i,j,k] = T[i,j,k] - sum_{c'!=c} U[i,c']*V[j,c']*W[k,c']
                # We minimize sum_{i,j,k} (R_c[i,j,k] - U[i,c]*V[j,c]*W[k,c])^2 over U[:,c]
                # This is a linear least squares problem
                
                for c in range(r):
                    # Compute residual without column c
                    R = T.copy()
                    for c2 in range(r):
                        if c2 != c:
                            R -= np.einsum('i,j,k->ijk', U[:,c2], V[:,c2], W[:,c2])
                    
                    # Minimize ||R - U[:,c] outer V[:,c] outer W[:,c]||^2
                    # For each i: sum_{j,k} (R[i,j,k] - U[i,c]*V[j,c]*W[k,c])^2
                    # = sum_{j,k} R[i,j,k]^2 - 2*U[i,c]*sum_{j,k} R[i,j,k]*V[j,c]*W[k,c] + U[i,c]^2 * sum_{j,k} V[j,c]^2 * W[k,c]^2
                    # Derivative w.r.t. U[i,c]: -2*sum_{j,k} R[i,j,k]*V[j,c]*W[k,c] + 2*U[i,c]*sum_{j,k} V[j,c]^2*W[k,c]^2 = 0
                    # U[i,c] = sum_{j,k} R[i,j,k]*V[j,c]*W[k,c] / sum_{j,k} V[j,c]^2*W[k,c]^2
                    
                    denom = np.sum(V[:,c]**2 * W[:,c]**2)
                    if abs(denom) < 1e-15:
                        continue
                    # numerator[i] = sum_{j,k} R[i,j,k] * V[j,c] * W[k,c]
                    # = sum_j V[j,c] * sum_k R[i,j,k] * W[k,c]
                    # = sum_j V[j,c] * (R[:,j,:] * W[:,c]) @ ... 
                    # More efficiently: numerator = (R * W[:,c]) contracted with V[:,c] over j
                    # numerator[i] = sum_j sum_k R[i,j,k] * V[j,c] * W[k,c]
                    # = sum_j V[j,c] * (R[i,:,:] dot W[:,c])
                    # Let's compute: temp[i,j] = sum_k R[i,j,k]*W[k,c], then numerator[i] = sum_j temp[i,j]*V[j,c]
                    temp = np.einsum('ijk,k->ij', R, W[:,c])  # (nm, mp)
                    numerator = temp @ V[:,c]  # (nm,)
                    U[:,c] = numerator / denom
                
                # Similarly update V and W
                for c in range(r):
                    R = T.copy()
                    for c2 in range(r):
                        if c2 != c:
                            R -= np.einsum('i,j,k->ijk', U[:,c2], V[:,c2], W[:,c2])
                    
                    denom = np.sum(U[:,c]**2 * W[:,c]**2)
                    if abs(denom) < 1e-15:
                        continue
                    temp = np.einsum('ijk,k->ik', R, W[:,c])  # (nm, pn)
                    numerator = temp @ U[:,c]  # (mp,)
                    V[:,c] = numerator / denom
                
                for c in range(r):
                    R = T.copy()
                    for c2 in range(r):
                        if c2 != c:
                            R -= np.einsum('i,j,k->ijk', U[:,c2], V[:,c2], W[:,c2])
                    
                    denom = np.sum(U[:,c]**2 * V[:,c]**2)
                    if abs(denom) < 1e-15:
                        continue
                    temp = np.einsum('ijk,j->ik', R, V[:,c])  # (nm, pn)
                    numerator = temp @ U[:,c]  # (pn,)
                    W[:,c] = numerator / denom
                
                # Check convergence
                if it % 50 == 0:
                    res = tensor_residual(U, V, W)
                    if res < tol:
                        break
            
            # Final check
            res = tensor_residual(U, V, W)
            if res < 1e-10:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                # Verify exactness after rounding
                res_r = tensor_residual(U_r, V_r, W_r)
                if res_r < 1e-10:
                    return (U_r, V_r, W_r, r)
        
        return None
    
    # Try ranks from low to high
    # Lower bound: max(nm, mp, pn) = 35
    for r in range(max(nm, mp, pn), best_rank):
        if time.time() > deadline - 30:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
