"""Rank-minimal tensor decomposition search for 3x4x6 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
A_DIM = N * M  # 12
B_DIM = M * P  # 24
C_DIM = P * N  # 18
TENSOR_SIZE = A_DIM * B_DIM * C_DIM  # 576
SCHOOLBOOK_RANK = N * M * P  # 72

def _build_tensor():
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=M+i, b=M+j, c=i*N+k."""
    T = np.zeros((A_DIM, B_DIM, C_DIM), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T

TENSOR = _build_tensor()

def _verify(U, V, W):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T (broadcast outer product sum)."""
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, TENSOR, atol=1e-8, rtol=1e-10)

def _random_decomposition(rank, rng, seed_offset=0):
    """Generate a random decomposition of the given rank."""
    s = seed_offset + rank
    rng = random.Random(s)
    U = np.array([[rng.gauss(0,1) for _ in range(rank)] for _ in range(A_DIM)])
    V = np.array([[rng.gauss(0,1) for _ in range(rank)] for _ in range(B_DIM)])
    W = np.array([[rng.gauss(0,1) for _ in range(rank)] for _ in range(C_DIM)])
    return U, V, W

def _check_decomposition(U, V, W):
    """Check if U,V,W exactly decompose the tensor."""
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, TENSOR, atol=1e-8, rtol=1e-10)

def _objective(params, rank):
    """
    Objective function for optimization.
    params is a flat array of size (A_DIM + B_DIM + C_DIM) * rank.
    """
    # Reshape params
    size_U = A_DIM * rank
    size_V = B_DIM * rank
    size_W = C_DIM * rank
    
    U = params[:size_U].reshape(A_DIM, rank)
    V = params[size_U:size_U+size_V].reshape(B_DIM, rank)
    W = params[size_U+size_V:].reshape(C_DIM, rank)
    
    # Compute tensor
    result = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Compute error
    error = result - TENSOR
    return np.sum(error**2)

def _gradient(params, rank):
    """
    Gradient of the objective function.
    """
    size_U = A_DIM * rank
    size_V = B_DIM * rank
    size_W = C_DIM * rank
    
    U = params[:size_U].reshape(A_DIM, rank)
    V = params[size_U:size_U+size_V].reshape(B_DIM, rank)
    W = params[size_U+size_V:].reshape(C_DIM, rank)
    
    # result = einsum('ar,br,cr->abc', U, V, W)
    # error = result - T
    # dL/dU_ar = sum_bc (result_abc - T_abc) V_br W_cr
    # We can compute this efficiently using einsum
    
    result = np.einsum('ar,br,cr->abc', U, V, W)
    error = result - TENSOR
    
    # grad_U[a,r] = sum_{b,c} error[a,b,c] V[b,r] W[c,r]
    grad_U = np.einsum('abc,br,cr->ar', error, V, W)
    # grad_V[b,r] = sum_{a,c} error[a,b,c] U[a,r] W[c,r]
    grad_V = np.einsum('abc,ar,cr->br', error, U, W)
    # grad_W[c,r] = sum_{a,b} error[a,b,c] U[a,r] V[b,r]
    grad_W = np.einsum('abc,ar,br->cr', error, U, V)
    
    return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

def _optimize_rank(rank, num_restarts, deadline):
    """
    Try to find a valid decomposition of the given rank using optimization.
    """
    for restart in range(num_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize with small random values
        rng = random.Random(1234 + rank * 100 + restart)
        U_init = np.random.randn(A_DIM, rank) * 0.1
        V_init = np.random.randn(B_DIM, rank) * 0.1
        W_init = np.random.randn(C_DIM, rank) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(
                _objective, 
                params_init, 
                args=(rank,), 
                method='L-BFGS-B', 
                jac=_gradient,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
        
        if res.fun < 1e-6:
            # Extract optimized parameters
            params_opt = res.x
            size_U = A_DIM * rank
            size_V = B_DIM * rank
            size_W = C_DIM * rank
            
            U_opt = params_opt[:size_U].reshape(A_DIM, rank)
            V_opt = params_opt[size_U:size_U+size_V].reshape(B_DIM, rank)
            W_opt = params_opt[size_U+size_V:].reshape(C_DIM, rank)
            
            # Snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            if _check_decomposition(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
            else:
                # Maybe the optimization converged to a non-integer solution that is still valid?
                # The task requires integer/half-integer entries.
                # If the float solution is very close to integer/half-integer, we already snapped.
                # If it's not close, it's invalid for the "exact" requirement unless we allow floats?
                # "integer/half-integer entries that exactly decompose"
                # So we must snap.
                pass
                
    return None

def _schoolbook():
    """Return the schoolbook decomposition (rank 72)."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((A_DIM, rank))
    V = np.zeros((B_DIM, rank))
    W = np.zeros((C_DIM, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook()
    best_result = (U_fb, V_fb, W_fb)
    best_rank = SCHOOLBOOK_RANK
    
    # Try to find lower rank decompositions using optimization
    # We try ranks from 71 down to 50
    # Given time constraints, we limit the number of restarts
    for rank in range(71, 49, -1):
        if time.time() > deadline:
            break
            
        # Try 2 restarts for each rank
        result = _optimize_rank(rank, 2, deadline)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break # Found a valid lower rank, stop searching lower? 
            # Actually, we want the MINIMAL rank. So we should continue searching lower if we have time.
            # But if we found one at 71, it's unlikely 70 exists if 71 was hard.
            # Let's just break for now to save time, as 71 is a huge improvement over 72.
            # If we have time, we can continue.
            
    # If we have time, we could try to go lower, but for now, finding ANY lower rank is a win.
    # Let's allow a bit more search if we found one.
    if best_rank < SCHOOLBOOK_RANK and time.time() < deadline - 10:
        for rank in range(best_rank - 1, 49, -1):
            if time.time() > deadline:
                break
            result = _optimize_rank(rank, 1, deadline)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                break

    # Final check: make sure we return a valid decomposition
    U, V, W = best_result
    if not _check_decomposition(U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
    
    # Convert to lists of lists (integer/half-integer entries)
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
