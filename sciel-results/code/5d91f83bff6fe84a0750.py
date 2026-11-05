"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 30
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    
    # Try to find a lower rank decomposition using rank-1 peeling
    # Start with the full tensor and try to peel off rank-1 components
    
    def try_decompose(residual, max_rank, start_rank=0):
        """Try to decompose residual into rank-1 components with half-integer entries."""
        # Check if residual is zero
        if np.linalg.norm(residual) < 1e-12:
            return []
        
        # If we've used up our budget, fail
        if start_rank >= max_rank:
            return None
        
        # Find a rank-1 component to peel off
        # A rank-1 tensor is u ⊗ v ⊗ w
        # We want to find u, v, w such that residual - c*(u⊗v⊗w) has "simpler" structure
        
        # Strategy: Look for entries in residual that are ±1 or ±0.5
        # and try to find a rank-1 tensor that matches a subset of these
        
        # First, let's try to find a rank-1 approximation using ALS
        best_u, best_v, best_w, best_score = None, None, None, -1e10
        
        np.random.seed(42 + start_rank)
        for trial in range(30):
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # ALS iterations
            for _ in range(30):
                u = np.einsum('ijk,j,k->i', residual, v, w)
                v = np.einsum('ijk,i,k->j', residual, u, w)
                w = np.einsum('ijk,i,j->k', residual, u, v)
                
                # Normalize to avoid overflow
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)
                norm_w = np.linalg.norm(w)
                if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                    break
                u = u / norm_u
                v = v / norm_v
                w = w / norm_w
            
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_score < 1e-8:
            return None
        
        # Now try to find a scaling factor c and check if the result has half-integer structure
        # We want residual - c*(u⊗v⊗w) to have entries that are 0, ±1, ±0.5, etc.
        
        # Try different scaling factors
        best_c, best_residual = None, None
        for c_candidate in [best_score, best_score/2, 1.0, 0.5, -0.5, -1.0]:
            new_residual = residual - c_candidate * np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            
            # Check if new_residual has a "nice" structure
            # Count how many entries are close to half-integers
            half_int_count = 0
            total_nonzero = 0
            for idx in np.ndindex(new_residual.shape):
                val = new_residual[idx]
                if abs(val) > 1e-8:
                    total_nonzero += 1
                    # Check if close to a half-integer
                    nearest = np.round(val * 2) / 2
                    if abs(val - nearest) < 1e-6:
                        half_int_count += 1
            
            if total_nonzero > 0:
                ratio = half_int_count / total_nonzero
                if ratio > 0.95:
                    best_c = c_candidate
                    best_residual = new_residual
                    break
        
        if best_c is None:
            # Try to find a better rank-1 component
            # Look for patterns in the residual
            return None
        
        # Verify the decomposition is exact
        if np.linalg.norm(best_residual) < 1e-10:
            return [(best_u, best_v, best_w, best_c)]
        
        # Recurse
        sub_decomp = try_decompose(best_residual, max_rank, start_rank + 1)
        if sub_decomp is None:
            return None
        
        return [(best_u, best_v, best_w, best_c)] + sub_decomp
    
    # Try different target ranks from low to high
    for target_rank in range(20, 31):
        decomp = try_decompose(T, target_rank)
        if decomp is not None:
            # Build the factor matrices
            U_list = []
            V_list = []
            W_list = []
            
            for u, v, w, c in decomp:
                U_list.append(u * np.sqrt(c) if c > 0 else u * np.sqrt(-c))
                V_list.append(v * np.sqrt(c) if c > 0 else v * np.sqrt(-c))
                W_list.append(w)
            
            # This approach is getting complicated. Let me try a different strategy.
            pass
    
    # Alternative approach: Use known algebraic identities for matrix multiplication
    # For 2x3x5, we can try to find a decomposition by exploiting the structure
    
    # Let's try a direct approach: enumerate possible rank-1 tensors with small integer/half-integer entries
    # and try to find a combination that sums to T
    
    # First, generate candidate rank-1 tensors
    # A rank-1 tensor u⊗v⊗w where u, v, w have entries in {-1, -0.5, 0, 0.5, 1}
    
    # This is too many possibilities. Let's use a smarter approach.
    
    # For matrix multiplication, we know that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (with appropriate indexing)
    
    # Let's try to find a decomposition by solving a linear system
    # Assume rank r, and we want to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # and 0 for all other (i1,j1,k1,i2,j2,k2) combinations
    
    # This is a system of bilinear equations, which is hard to solve directly.
    
    # Let's try a heuristic: start with schoolbook and try to merge terms
    
    # Schoolbook: 30 terms, one for each (i,j,k)
    # Can we merge any two terms into one?
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist
    # u, v, w such that:
    # u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = 1
    # u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
    # and all other entries of u⊗v⊗w are 0
    
    # This is only possible if the supports don't overlap in a conflicting way.
    
    # Let's try a different approach: use the fact that the tensor has a specific structure
    # and try to find a decomposition by hand or using known results.
    
    # For 2x3x5, the minimal rank is not known to me, but it's at most 30.
    # Let's just return the schoolbook decomposition as a safe fallback.
    
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
