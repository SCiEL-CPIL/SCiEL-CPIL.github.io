"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to be safe within 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol
    
    def to_list(mat):
        """Convert numpy array to list of lists with clean values."""
        result = []
        for row in mat:
            result.append([])
            for x in row:
                if abs(x) < 1e-12:
                    result[-1].append(0)
                elif abs(x - 1) < 1e-12:
                    result[-1].append(1)
                elif abs(x + 1) < 1e-12:
                    result[-1].append(-1)
                elif abs(x - 0.5) < 1e-12:
                    result[-1].append(0.5)
                elif abs(x + 0.5) < 1e-12:
                    result[-1].append(-0.5)
                elif abs(x - 1.5) < 1e-12:
                    result[-1].append(1.5)
                elif abs(x + 1.5) < 1e-12:
                    result[-1].append(-1.5)
                elif abs(x - 2) < 1e-12:
                    result[-1].append(2)
                elif abs(x + 2) < 1e-12:
                    result[-1].append(-2)
                else:
                    # Try to find a simple fraction
                    from fractions import Fraction
                    try:
                        frac = Fraction(x).limit_denominator(100)
                        if abs(x - float(frac)) < 1e-10:
                            if frac.denominator == 1:
                                result[-1].append(frac.numerator)
                            else:
                                result[-1].append(float(frac))
                        else:
                            result[-1].append(float(x))
                    except:
                        result[-1].append(float(x))
        return result
    
    def als_optimize(U, V, W, max_iter=200, tol=1e-10):
        """Alternating least squares optimization for tensor decomposition."""
        nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each i, solve linear system
            # T[i,:,:] = sum_c U[i,c] * (V[:,c] ⊗ W[:,c])
            # So we need to find U[i,:] that best fits T[i,:,:]
            # Reshape: for fixed i, T[i,:,:] is (mp, pn)
            # V[:,c] ⊗ W[:,c] is (mp*pn,)
            # So T[i,:,:].flatten() = U[i,:] @ (V⊙W).T where V⊙W has shape (mp*pn, rank)
            
            # Build V⊙W matrix: (mp*pn, rank)
            VW = np.einsum('jc,kc->jk c', V, W).reshape(mp * pn, rank)
            
            # For each i, solve: T[i,:,:].flatten() ≈ U[i,:] @ VW.T
            # This is a least squares problem
            U_new = np.zeros_like(U)
            for i in range(nm):
                # Solve U[i,:] = argmin ||T[i,:,:].flatten() - U[i,:] @ VW.T||
                # Equivalent to: VW @ U[i,:].T ≈ T[i,:,:].flatten()
                try:
                    U_new[i, :] = np.linalg.lstsq(VW, T[i, :, :].flatten(), rcond=None)[0]
                except:
                    U_new[i, :] = U[i, :]
            U = U_new
            
            # Update V: for each j, solve linear system
            # T[:,j,:] = sum_c V[j,c] * (U[:,c] ⊗ W[:,c])
            UW = np.einsum('ic,kc->ik c', U, W).reshape(nm * pn, rank)
            
            V_new = np.zeros_like(V)
            for j in range(mp):
                try:
                    V_new[j, :] = np.linalg.lstsq(UW, T[:, j, :].flatten(), rcond=None)[0]
                except:
                    V_new[j, :] = V[j, :]
            V = V_new
            
            # Update W: for each k, solve linear system
            # T[:,:,k] = sum_c W[k,c] * (U[:,c] ⊗ V[:,c])
            UV = np.einsum('ic,jc->ij c', U, V).reshape(nm * mp, rank)
            
            W_new = np.zeros_like(W)
            for k in range(pn):
                try:
                    W_new[k, :] = np.linalg.lstsq(UV, T[:, :, k].flatten(), rcond=None)[0]
                except:
                    W_new[k, :] = W[k, :]
            W = W_new
            
            # Check convergence
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try to find lower rank decompositions using ALS
    # Start from rank_school - 1 and go down
    for target_rank in range(rank_school - 1, max(15, rank_school - 15), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_attempts = 10 if target_rank > 30 else 20
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with some structure
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Run ALS optimization
            U_opt, V_opt, W_opt, err = als_optimize(U, V, W, max_iter=100)
            
            if err < 1e-9:
                best_rank = target_rank
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                found = True
                break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we couldn't find at this rank, stop (ranks are non-increasing in difficulty)
            break
    
    # If we found a good decomposition, try to reduce further
    if best_rank < rank_school:
        for target_rank in range(best_rank - 1, max(15, best_rank - 5), -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            for attempt in range(15):
                if time.time() > deadline - 3:
                    break
                
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
                
                U_opt, V_opt, W_opt, err = als_optimize(U, V, W, max_iter=150)
                
                if err < 1e-9:
                    best_rank = target_rank
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
                    found = True
                    break
            
            if not found:
                break
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
