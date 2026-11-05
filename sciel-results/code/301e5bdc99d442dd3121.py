"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
from itertools import product
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Build the tensor T with entries T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # The schoolbook rank is n*m*p = 30
    schoolbook_rank = n * m * p
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition with rank = n*m*p."""
        U = [[0] * schoolbook_rank for _ in range(dim_u)]
        V = [[0] * schoolbook_rank for _ in range(dim_v)]
        W = [[0] * schoolbook_rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U_list, V_list, W_list):
        """Verify that the decomposition exactly reproduces the tensor."""
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        # Check all entries
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        
        # Check non-support entries are zero
        support = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    support.add((i*m+j, j*p+k, k*n+i))
        
        for u_idx in range(dim_u):
            for v_idx in range(dim_v):
                for w_idx in range(dim_w):
                    if (u_idx, v_idx, w_idx) not in support:
                        val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                        err = abs(val)
                        max_err = max(max_err, err)
        
        return max_err < 1e-8
    
    def try_rationalize(U_list, V_list, W_list, max_denom=100):
        """Try to find rational approximations for the factors."""
        U_frac = []
        for row in U_list:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    frac = Fraction(val).limit_denominator(max_denom)
                    frac_row.append(float(frac))
            U_frac.append(frac_row)
        
        V_frac = []
        for row in V_list:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    frac = Fraction(val).limit_denominator(max_denom)
                    frac_row.append(float(frac))
            V_frac.append(frac_row)
        
        W_frac = []
        for row in W_list:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    frac = Fraction(val).limit_denominator(max_denom)
                    frac_row.append(float(frac))
            W_frac.append(frac_row)
        
        return U_frac, V_frac, W_frac
    
    def find_rank1_peel(residual, seed=0, max_trials=200):
        """
        Find a rank-1 tensor u⊗v⊗w that best approximates the residual.
        Uses alternating least squares with multiple random initializations.
        Returns (u, v, w, score) where score = <residual, u⊗v⊗w>.
        """
        np.random.seed(seed)
        best_score = -np.inf
        best_u = None
        best_v = None
        best_w = None
        
        for trial in range(max_trials):
            u = np.random.randn(dim_u)
            v = np.random.randn(dim_v)
            w = np.random.randn(dim_w)
            
            # ALS iterations
            for _ in range(50):
                # Update u: u[i] = sum_{j,k} residual[i,j,k] * v[j] * w[k]
                u_new = np.einsum('ijk,j,k->i', residual, v, w)
                # Update v: v[j] = sum_{i,k} residual[i,j,k] * u[i] * w[k]
                v_new = np.einsum('ijk,i,k->j', residual, u, w)
                # Update w: w[k] = sum_{i,j} residual[i,j,k] * u[i] * v[j]
                w_new = np.einsum('ijk,i,j->k', residual, u, v)
                
                # Normalize to prevent blowup
                u_norm = np.linalg.norm(u_new)
                v_norm = np.linalg.norm(v_new)
                w_norm = np.linalg.norm(w_new)
                if u_norm < 1e-15 or v_norm < 1e-15 or w_norm < 1e-15:
                    break
                u = u_new / u_norm
                v = v_new / v_norm
                w = w_new / w_norm
            
            # Compute score
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        return best_u, best_v, best_w, best_score
    
    # Try to find a decomposition with rank < 30 using rank-1 peeling
    # We'll try to peel off rank-1 components one at a time
    
    start_time = time.time()
    time_limit = 100  # Leave some time for verification and fallback
    
    for target_rank in range(20, schoolbook_rank):
        if time.time() - start_time > time_limit:
            break
        
        residual = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        success = True
        for step in range(target_rank):
            if time.time() - start_time > time_limit:
                success = False
                break
            
            # Find best rank-1 approximation to residual
            u, v, w, score = find_rank1_peel(residual, seed=step * 1000)
            
            if score <= 1e-10:
                success = False
                break
            
            # Subtract the rank-1 tensor from residual
            # Scale so that the contraction with residual is 1
            # The rank-1 tensor is u⊗v⊗w, and <residual, u⊗v⊗w> = score
            # We want to subtract exactly u⊗v⊗w (not scaled) because
            # the peeling should remove the component in that direction
            residual -= np.einsum('i,j,k->ijk', u, v, w)
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        if not success:
            continue
        
        # Check if residual is close to zero
        if np.linalg.norm(residual) > 1e-8:
            continue
        
        # Try to rationalize
        U_frac, V_frac, W_frac = try_rationalize(U_list, V_list, W_list, max_denom=100)
        
        # Verify the rationalized decomposition
        if verify_decomposition(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # Try with larger denominators
        U_frac, V_frac, W_frac = try_rationalize(U_list, V_list, W_list, max_denom=1000)
        if verify_decomposition(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
