import numpy as np
from scipy.optimize import minimize
import itertools
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p
    rank_schoolbook = n * m * p
    
    # We'll try to find a decomposition with rank less than schoolbook
    # For 3x4 * 4x4, the matrix multiplication tensor has dimensions:
    # A: 3x4, B: 4x4, C: 3x4
    # Tensor indices: (i,j,k) where i in [0,3), j in [0,4), k in [0,4)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's use a random search approach to find a lower rank decomposition
    # We'll try ranks from 35 up to schoolbook
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = None, None, None
    
    # For each candidate rank, try random initialization
    for rank in range(35, rank_schoolbook + 1):
        # Try multiple random restarts
        for attempt in range(50):
            random.seed(attempt + rank * 1000)
            np.random.seed(attempt + rank * 1000)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Build the tensor from the decomposition and compare with target
            # Target tensor: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
            # i in [0,n), j in [0,m), k in [0,p)
            
            # The decomposed tensor is: T_dec[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # We want T_dec[a, b, c] = T[a, b, c] for all a, b, c
            
            # Let's set up the equations:
            # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For all other (a, b, c): sum_r U[a, r] * V[b, r] * W[c, r] = 0
            
            # This is a system of polynomial equations. Let's use a simple approach:
            # Fix W and solve for U, V iteratively.
            
            # Actually, let's use a different approach: 
            # We can use the fact that the matrix multiplication tensor can be
            # decomposed using rank-1 tensors. Let's try to find such a decomposition.
            
            # For a given rank r, we need r triples (u_t, v_t, w_t) such that
            # sum_t u_t[a] * v_t[b] * w_t[c] = T[a,b,c]
            
            # Let's use a greedy approach:
            # 1. Start with the target tensor T
            # 2. At each step, find the best rank-1 approximation to the remaining tensor
            # 3. Subtract it and repeat until rank is reached or residual is small
            
            # Build target tensor
            T = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1.0
            
            # Greedy rank-1 decomposition
            residual = T.copy()
            U_rows = []
            V_rows = []
            W_rows = []
            
            for t in range(rank):
                # Find the best rank-1 approximation to residual
                # This is equivalent to finding the top singular value of the tensor
                # We can do this by alternating least squares
                
                # Initialize with random vectors
                u = np.random.randn(n * m)
                v = np.random.randn(m * p)
                w = np.random.randn(p * n)
                
                # Normalize
                u = u / np.linalg.norm(u)
                v = v / np.linalg.norm(v)
                w = w / np.linalg.norm(w)
                
                # Alternating minimization
                for _ in range(50):
                    # Update u: u ~ (V ⊗ W) * residual
                    VW = np.outer(v, w)
                    u_new = np.einsum('abc,ac->b', residual, VW)
                    if np.linalg.norm(u_new) > 1e-10:
                        u = u_new / np.linalg.norm(u_new)
                    
                    # Update v: v ~ (U ⊗ W) * residual
                    UW = np.outer(u, w)
                    v_new = np.einsum('abc,ab->c', residual, UW)
                    if np.linalg.norm(v_new) > 1e-10:
                        v = v_new / np.linalg.norm(v_new)
                    
                    # Update w: w ~ (U ⊗ V) * residual
                    UV = np.outer(u, v)
                    w_new = np.einsum('abc,ab->c', residual, UV)
                    if np.linalg.norm(w_new) > 1e-10:
                        w = w_new / np.linalg.norm(w_new)
                
                # Compute the best coefficient for this rank-1 component
                coeff = np.einsum('abc,a,b,c->', residual, u, v, w)
                
                # Subtract
                residual -= coeff * np.einsum('a,b,c->abc', u, v, w)
                
                # Store
                U_rows.append(u)
                V_rows.append(v)
                W_rows.append(w)
                
                # Check if residual is small
                if np.linalg.norm(residual) < 1e-6:
                    break
            
            # Build final matrices
            U_mat = np.column_stack(U_rows).T
            V_mat = np.column_stack(V_rows).T
            W_mat = np.column_stack(W_rows).T
            
            # Verify the decomposition
            T_dec = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
            error = np.linalg.norm(T_dec - T)
            
            if error < 1e-4:
                # Round to nearest integer or half-integer
                U_rounded = np.round(U_mat * 2) / 2
                V_rounded = np.round(V_mat * 2) / 2
                W_rounded = np.round(W_mat * 2) / 2
                
                # Re-verify with rounded values
                T_dec_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                error_rounded = np.linalg.norm(T_dec_rounded - T)
                
                if error_rounded < 1e-3:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        rank = rank_schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

