"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x7 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np
from scipy import linalg

DIMENSIONS = [4, 4, 7]
n, m, p = DIMENSIONS

# The multiplication tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# We seek rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
# with T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

# Flattened tensor indices:
# A index: a = i*m + j  (0..n*m-1)
# B index: b = j*p + k  (0..m*p-1)
# C index: c_idx = k*n + i  (0..p*n-1)
# T[a, b, c_idx] = 1 if and only if the j in A and B match, and the k in B and C match, and i in A and C match.

def build_tensor():
    """Build the multiplication tensor as a 3D array T[a, b, c_idx]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[a, b, c_idx] = 1.0
    return T

def schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
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
    return U, V, W

def check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    rank = U.shape[1]
    # T[a, b, c] = sum_c U[a,c]*V[b,c]*W[c,c]
    # Use einsum: T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
    T_reconstructed = np.einsum('ad,bd,cd->abc', U, V, W)
    T_true = build_tensor()
    max_err = np.max(np.abs(T_reconstructed - T_true))
    return max_err < tol, max_err

def random_initialization(rank, rng):
    """Random initialization of U, V, W."""
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    # Normalize columns to avoid scale issues
    for c in range(rank):
        norm_u = np.linalg.norm(U[:, c])
        norm_v = np.linalg.norm(V[:, c])
        norm_w = np.linalg.norm(W[:, c])
        total = norm_u * norm_v * norm_w
        if total > 1e-12:
            U[:, c] /= norm_u
            V[:, c] /= norm_v
            W[:, c] /= norm_w
    return U, V, W

def gradient_descent(U, V, W, max_iter=2000, lr=0.01, tol=1e-10, verbose=False):
    """
    Gradient descent on the loss: sum_{a,b,c} (T_true[a,b,c] - sum_d U[a,d]V[b,d]W[c,d])^2
    """
    T_true = build_tensor()
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute reconstruction
        T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
        error = T_rec - T_true
        loss = np.sum(error ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Gradients:
        # dL/dU[a,d] = 2 * sum_{b,c} error[a,b,c] * V[b,d] * W[c,d]
        # dL/dV[b,d] = 2 * sum_{a,c} error[a,b,c] * U[a,d] * W[c,d]
        # dL/dW[c,d] = 2 * sum_{a,b} error[a,b,c] * U[a,d] * V[b,d]
        
        dU = 2 * np.einsum('abc,bd,cd->ad', error, V, W)
        dV = 2 * np.einsum('abc,ad,cd->bd', error, U, W)
        dW = 2 * np.einsum('abc,ad,bd->cd', error, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Periodic renormalization to prevent blowup
        if it % 50 == 0:
            for c in range(rank):
                nu = np.linalg.norm(U[:, c])
                nv = np.linalg.norm(V[:, c])
                nw = np.linalg.norm(W[:, c])
                total = nu * nv * nw
                if total > 1e-12:
                    U[:, c] /= nu
                    V[:, c] /= nv
                    W[:, c] /= nw
    
    # Check final
    T_rec = np.einsum('ad,bd,cd->abc', best_U, best_V, best_W)
    max_err = np.max(np.abs(T_rec - T_true))
    return best_U, best_V, best_W, max_err

def try_rank(rank, num_restarts=5, max_iter=1500, lr=0.005, seed=42):
    """Try to find a valid decomposition at the given rank."""
    rng = np.random.default_rng(seed)
    best_err = np.inf
    best_UVW = None
    
    for restart in range(num_restarts):
        U, V, W = random_initialization(rank, rng)
        U, V, W, err = gradient_descent(U, V, W, max_iter=max_iter, lr=lr, verbose=False)
        if err < best_err:
            best_err = err
            best_UVW = (U, V, W)
        if err < 1e-9:
            return True, U, V, W
    
    return False, best_UVW[0], best_UVW[1], best_UVW[2] if best_UVW else (None, None, None)

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # For 4x4*4x7, the rank is likely around 28-32
    # Try ranks from a reasonable lower bound upward
    
    min_possible = max(n * m, m * p, p * n)  # 28
    
    # Try a range of ranks
    for rank in range(min_possible, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found, U, V, W = try_rank(rank, num_restarts=3, max_iter=1000, lr=0.005, seed=rank*1000+42)
        if found:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            # If we found it, we can stop searching for lower ranks
            # But we should check if we can go lower
            # Actually, we're iterating from low to high, so first found is the minimum
            break
    
    # Convert to list of lists with Fraction for exactness if possible
    # But the problem says "integer/half-integer entries", so we try to round
    # First, check if the solution is close to rational with small denominators
    
    # Try to find a rational approximation
    def to_fraction_matrix(M):
        """Convert a numpy matrix to a list of lists of Fractions, trying small denominators."""
        result = []
        for i in range(M.shape[0]):
            row = []
            for j in range(M.shape[1]):
                val = M[i, j]
                # Try to find a fraction with denominator up to 2
                # Since the problem allows half-integers, try denominator 1 and 2
                best_frac = Fraction(val).limit_denominator(2)
                # Check if it's close
                if abs(float(best_frac) - val) < 1e-6:
                    row.append(best_frac)
                else:
                    # Try denominator up to 4
                    best_frac = Fraction(val).limit_denominator(4)
                    if abs(float(best_frac) - val) < 1e-6:
                        row.append(best_frac)
                    else:
                        # Just use the float as a fraction
                        row.append(Fraction(val).limit_denominator(100))
            result.append(row)
        return result
    
    # Convert to the required format
    U_out = to_fraction_matrix(best_U)
    V_out = to_fraction_matrix(best_V)
    W_out = to_fraction_matrix(best_W)
    
    # Final validation
    U_np = np.array([[float(x) for x in row] for row in U_out])
    V_np = np.array([[float(x) for x in row] for row in V_out])
    W_np = np.array([[float(x) for x in row] for row in W_out])
    
    T_rec = np.einsum('ad,bd,cd->abc', U_np, V_np, W_np)
    T_true = build_tensor()
    max_err = np.max(np.abs(T_rec - T_true))
    
    if max_err > 1e-6:
        # Fall back to schoolbook
        U_np, V_np, W_np = schoolbook()
    
    # Convert back to list format
    U_final = U_np.tolist()
    V_final = V_np.tolist()
    W_final = W_np.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
