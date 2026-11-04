"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Schoolbook gives rank 60. Try to find lower rank via randomized ALS.
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from 59 down, but limit total attempts
    for rank in range(best_rank - 1, max(best_rank - 20, 10), -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            # If we found a lower rank, try even lower
            if best_rank <= 40:
                break

    return best_result


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using alternating least squares."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Build the matrix multiplication tensor as a 3-way tensor
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has entries T[i,j,k] = delta_{i,k} in the appropriate sense
    # Actually, the tensor for matmul is: T_{i,j,k} such that
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i,j] * V_r[j,k] * W_r[k,i]... 
    
    # Let me think more carefully. The matrix multiplication tensor M_{ijk}
    # is defined so that C_{ik} = sum_j A_{ij} B_{jk}
    # In tensor form: M is a 3-tensor with M[i,j,k] = 1 if... 
    # Actually the standard formulation: the matmul tensor has entries
    # M[i,j,k] = delta where the "output" index pairs (i,k) with input indices
    # 
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k) in a specific pattern.
    # For matmul 2x5 * 5x6:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 if this term contributes, which is always for valid indices.
    # But that's just all ones, which doesn't capture the structure.
    #
    # The correct tensor: T is a 3-way array where T[i,j,k] = 1 for all 
    # (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1}, and the "contraction" 
    # gives C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k. That's just the all-ones tensor, 
    # which has rank 1. That can't be right.
    #
    # Let me reconsider. The matrix multiplication tensor is:
    # T[i, j, k] where i indexes rows of A (and C), j indexes cols of A / rows of B,
    # k indexes cols of B (and C).
    # The tensor is defined by: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # As a 3-tensor, T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # Wait, that IS all ones. But the rank of the all-ones 2x5x6 tensor is 1.
    # That's clearly wrong for matrix multiplication.
    #
    # I think the confusion is in how the tensor is structured. Let me use the
    # standard formulation from Strassen's work:
    # The matmul tensor M has entries M[i,j,k] where:
    # - i ranges over row indices of A and C: 0..n-1
    # - j ranges over column indices of A / row indices of B: 0..m-1  
    # - k ranges over column indices of B and C: 0..p-1
    # And M[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k]
    # which is ALWAYS the case for valid indices. So M is indeed all-ones.
    #
    # But wait - that can't give rank 1. The issue is that the tensor 
    # decomposition is not of the form T = sum U_r ⊗ V_r ⊗ W_r with 
    # T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i].
    #
    # The correct formulation: C[i,k] = sum_{j,r} U_r[i,j] * V_r[j,k] * W_r[k,i]... no.
    #
    # Let me look at this differently. The standard tensor decomposition for matmul:
    # We want: C[i,k] = sum_r prod_{(a,b) in S_r} A[a,b] * prod_{(b,c) in T_r} B[b,c]
    # where each rank-1 term is a product of entries.
    #
    # For the schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term is A[i,j] * B[j,k], so U_r[i,j]=1, V_r[j,k]=1, and W_r is trivial.
    #
    # Actually, the standard formulation uses:
    # C[i,k] = sum_r sum_{(a,b) in pairs_r} U_r[a,b] A[a,b] * sum_{(b',c) in pairs'_r} V_r[b',c] B[b',c]
    # with W_r encoding which output position.
    #
    # Let me just use the flattening approach. The tensor T is a 3-way array
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We want to decompose T as sum_r u_r ⊗ v_r ⊗ w_r where
    # u_r is a vector of length n*m, v_r of length m*p, w_r of length p*n.
    # T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    #
    # For the schoolbook: T[i,j,k]=1, and we have rank n*m*p terms where
    # each term activates one (i,j,k) combination.
    #
    # So we need: 1 = sum_{r=1}^{R} U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] for all i,j,k.
    #
    # This is a system of n*m*p = 60 equations in R*(n*m + m*p + p*n) = R*(10+30+12)=52R unknowns.
    # For R < 60, we need 52R >= 60, so R >= 2. But the nonlinear constraints make it hard.
    
    # Build the linear system: for each (i,j,k), we need sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i] = 1
    # This is nonlinear. Use ALS: fix two factors, solve for the third.
    
    # Initialize randomly
    rng = random.Random(42 + rank)
    U = np.random.randn(dim_u, rank)
    V = np.random.randn(dim_v, rank)
    W = np.random.randn(dim_w, rank)
    
    # Scale to avoid overflow
    scale = 1.0 / (rank ** 0.5)
    U *= scale
    V *= scale
    W *= scale
    
    max_iter = 50
    tol = 1e-8
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            return None
        
        # Update U: for each (i,j), U[i*m+j, r] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # This is still nonlinear. Let's use a different approach.
        
        # Actually, let's use the fact that for fixed V and W, the equation is linear in U.
        # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # Let's define for each r: G_r[k] = V[j*p+k, r] * W[k*n+i, r]
        # Then sum_r U[i*m+j, r] * G_r[k] = 1 for all k
        # This is: U[i*m+j, :] @ G[:, k] = 1 for all k
        # Where G is a rank x p matrix with G[r, k] = V[j*p+k, r] * W[k*n+i, r]
        # So U[i*m+j, :] @ G = [1, 1, ..., 1] (length p)
        # This is underdetermined if rank > p, so use least squares.
        
        new_U = np.zeros((dim_u, rank))
        for i in range(n):
            for j in range(m):
                # G[r, k] = V[j*p+k, r] * W[k*n+i, r]
                G = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        G[r, k] = V[j * p + k, r] * W[k * n + i, r]
                # Solve: new_U[i*m+j, :] @ G = ones(p)
                # i.e., G.T @ new_U[i*m+j, :] = ones(p)
                try:
                    new_U[i * m + j, :] = np.linalg.lstsq(G.T, np.ones(p), rcond=None)[0]
                except:
                    new_U[i * m + j, :] = np.linalg.pinv(G.T) @ np.ones(p)
        
        U = new_U
        
        # Update V: for each (j,k), V[j*p+k, r] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
        # For fixed (j,k): sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
        # H[r, i] = U[i*m+j, r] * W[k*n+i, r]
        # V[j*p+k, :] @ H = ones(n)
        # H.T @ V[j*p+k, :] = ones(n)
        
        new_V = np.zeros((dim_v, rank))
        for j in range(m):
            for k in range(p):
                H = np.zeros((rank, n))
                for r in range(rank):
                    for i in range(n):
                        H[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    new_V[j * p + k, :] = np.linalg.lstsq(H.T, np.ones(n), rcond=None)[0]
                except:
                    new_V[j * p + k, :] = np.linalg.pinv(H.T) @ np.ones(n)
        
        V = new_V
        
        # Update W: for each (k,i), W[k*n+i, r] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        # For fixed (k,i): sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
        # K[r, j] = U[i*m+j, r] * V[j*p+k, r]
        # W[k*n+i, :] @ K = ones(m)
        # K.T @ W[k*n+i, :] = ones(m)
        
        new_W = np.zeros((dim_w, rank))
        for k in range(p):
            for i in range(n):
                K = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        K[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    new_W[k * n + i, :] = np.linalg.lstsq(K.T, np.ones(m), rcond=None)[0]
                except:
                    new_W[k * n + i, :] = np.linalg.pinv(K.T) @ np.ones(m)
        
        W = new_W
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += abs(val - 1.0)
        
        if error < tol:
            # Try to rationalize
            result = _rationalize(U, V, W, n, m, p)
            if result is not None:
                return result
            return None
        
        # Check time
        if time.time() > deadline - 5:
            return None
    
    # Final check
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += abs(val - 1.0)
    
    if error < tol:
        result = _rationalize(U, V, W, n, m, p)
        if result is not None:
            return result
    return None


def _rationalize(U, V, W, n, m, p):
    """Try to find rational/half-integer approximation to the decomposition."""
    rank = U.shape[1]
    
    # Try to find a scaling factor that makes entries close to half-integers
    # First, check if entries are already close to half-integers
    max_denom = 4  # Try denominators 1, 2, 4
    
    for denom in [1, 2, 4]:
        scale = 2.0 / denom  # We'll multiply by this to get integers
        
        U_int = np.round(U * scale)
        V_int = np.round(V * scale)
        W_int = np.round(W * scale)
        
        # Check if the integer approximation works
        # We need: sum_r (U_int[i*m+j]*V_int[j*p+k]*W_int[k*n+i]) / scale^3 = 1
        # i.e., sum_r U_int[i*m+j]*V_int[j*p+k]*W_int[k*n+i] = scale^3
        
        target = scale ** 3
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                    error += abs(val - target)
        
        if error < 1e-6:
            # Convert to list format
            U_list = [[0] * rank for _ in range(n * m)]
            V_list = [[0] * rank for _ in range(m * p)]
            W_list = [[0] * rank for _ in range(p * n)]
            
            for i in range(n * m):
                for r in range(rank):
                    U_list[i][r] = U_int[i, r] / scale
            for i in range(m * p):
                for r in range(rank):
                    V_list[i][r] = V_int[i, r] / scale
            for i in range(p * n):
                for r in range(rank):
                    W_list[i][r] = W_int[i, r] / scale
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                        if abs(val - 1.0) > 1e-9:
                            return None
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
