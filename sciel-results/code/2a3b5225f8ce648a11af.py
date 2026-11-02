import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim1 = n * m
    dim2 = m * p
    dim3 = p * n
    
    start_time = time.time()
    deadline = 115  # seconds
    
    # Build the target tensor T
    T = np.zeros((dim1, dim2, dim3), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((dim1, rank_schoolbook))
    V_sb = np.zeros((dim2, rank_schoolbook))
    W_sb = np.zeros((dim3, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Only try a few ranks to stay within time budget
    ranks_to_try = [59, 58, 57]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline - 10:
            break
        
        # Try a couple of random initializations
        for trial in range(2):
            if time.time() - start_time > deadline - 10:
                break
            
            np.random.seed(99999 + rank * 10 + trial)
            U = np.random.randn(dim1, rank)
            V = np.random.randn(dim2, rank)
            W = np.random.randn(dim3, rank)
            
            # Alternating least squares with limited iterations
            for iteration in range(20):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Solve for W
                A_W = np.zeros((dim1 * dim2 * dim3, dim3 * rank))
                for a in range(dim1):
                    for b in range(dim2):
                        UV_row = U[a, :] * V[b, :]
                        for c in range(dim3):
                            A_W[(a * dim2 + b) * dim3 + c, c * rank:(c + 1) * rank] = UV_row
                
                b_vec = T.reshape(-1)
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_vec, rcond=None)
                W = W_flat.reshape(dim3, rank)
                
                # Solve for U
                A_U = np.zeros((dim1 * dim2 * dim3, dim1 * rank))
                for a in range(dim1):
                    for b in range(dim2):
                        for c in range(dim3):
                            coeff = V[b, :] * W[c, :]
                            A_U[(a * dim2 + b) * dim3 + c, a * rank:(a + 1) * rank] = coeff
                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_vec, rcond=None)
                U = U_flat.reshape(dim1, rank)
                
                # Solve for V
                A_V = np.zeros((dim1 * dim2 * dim3, dim2 * rank))
                for a in range(dim1):
                    for b in range(dim2):
                        for c in range(dim3):
                            coeff = U[a, :] * W[c, :]
                            A_V[(a * dim2 + b) * dim3 + c, b * rank:(b + 1) * rank] = coeff
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_vec, rcond=None)
                V = V_flat.reshape(dim2, rank)
            
            # Check solution quality
            residual = T - np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(residual)) < 1e-8:
                # Check for rational entries
                def is_rational(val, tol=1e-6):
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2
                    return abs(val - nearest_int) < tol or abs(val - nearest_half) < tol
                
                all_rational = True
                for mat in [U, V, W]:
                    for val in mat.flatten():
                        if not is_rational(val):
                            all_rational = False
                            break
                    if not all_rational:
                        break
                
                if all_rational:
                    def round_to_rational(mat):
                        result = np.zeros_like(mat)
                        for idx in np.ndindex(mat.shape):
                            val = mat[idx]
                            nearest_int = round(val)
                            nearest_half = round(val * 2) / 2
                            if abs(val - nearest_int) < 1e-6:
                                result[idx] = nearest_int
                            else:
                                result[idx] = nearest_half
                        return result
                    
                    U_rounded = round_to_rational(U)
                    V_rounded = round_to_rational(V)
                    W_rounded = round_to_rational(W)
                    
                    residual_final = T - np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    if np.max(np.abs(residual_final)) < 1e-6:
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_rank = rank
                        break
        
        if best_rank < rank_schoolbook:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
