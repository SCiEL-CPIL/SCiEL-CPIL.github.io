import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p, tol=1e-5):
    """
    Verify if U, V, W decompose the matrix multiplication tensor M_{n,m,p}.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Flatten U, V, W to vectors? No, use einsum or loops.
    # T_{i,j,k} = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # We want T_{i,j,k} = 1 for all i,j,k.
    
    # Reshape for easier indexing?
    # U_flat shape (n, m, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute tensor T of shape (n, m, p)
    # T[i, j, k] = sum_t U_r[i, j, t] * V_r[j, k, t] * W_r[k, i, t]
    
    # Using einsum: 'ijt,jkt,kit->ijk'
    T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
    
    # Check if T is all 1s
    return np.allclose(T, 1.0, atol=tol)

def _find_decomposition(n, m, p, rank, max_iters=1000, tol=1e-8, seed=0):
    """
    Find a rank decomposition of M_{n,m,p} using ALS.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        # Update U
        # For each i, j, we want sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1
        # This is not a simple linear system for U because V and W are coupled.
        # However, we can linearize by fixing V, W.
        # For fixed V, W, the equation is linear in U.
        # But the equation is sum_k (sum_t U[i,j,t] V[j,k,t] W[k,i,t]) = 1? No.
        # The equation is for EACH i, j, k: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1.
        # This is a system of n*m*p equations.
        # Variables are U (n*m*rank), V (m*p*rank), W (p*n*rank).
        
        # ALS step:
        # Fix V, W. Solve for U.
        # The equations are:
        # For each i, j, k: sum_t U[i,j,t] (V[j,k,t] W[k,i,t]) = 1.
        # This can be written as:
        # Let M[i,j,k] = V[j,k,t] W[k,i,t] summed over t? No, it's a product of vectors.
        # Let A_{ijk, t} = V[j,k,t] W[k,i,t].
        # Then sum_t A_{ijk, t} U[i,j,t] = 1.
        # This is a linear system A U_flat = b.
        # A is (n*m*p) x (n*m*rank).
        # This is sparse.
        
        # Constructing A explicitly might be slow but n,m,p are small.
        # n=2, m=3, p=3 -> 18 equations.
        # U has 6*rank variables.
        
        # Let's build the matrix A_U.
        # Rows indexed by (i,j,k).
        # Cols indexed by (i,j,t).
        # A_U[(i,j,k), (i',j',t)] = 1 if i==i' and j==j' else 0?
        # No, the term is U[i,j,t] * V[j,k,t] * W[k,i,t].
        # So for a fixed (i,j,k), the equation is:
        # sum_t U[i,j,t] * (V[j,k,t] * W[k,i,t]) = 1.
        # The variables are U[i,j,t] for t=1..rank.
        # The coefficients for U[i,j,t] are V[j,k,t] * W[k,i,t].
        # Note that U[i',j',t] with (i',j') != (i,j) do not appear in equation (i,j,k).
        # So the system decouples for each (i,j).
        # For each i, j, we have p equations (for k=0..p-1):
        # sum_t U[i,j,t] * C_{jk, t}^{(i)} = 1, where C_{jk, t}^{(i)} = V[j,k,t] * W[k,i,t].
        # This is a system of p equations for rank variables U[i,j, :].
        # If rank > p, it's underdetermined. We can use least squares.
        
        # Solve for U[i, j, :]
        for i in range(n):
            for j in range(m):
                # Build matrix A (p x rank)
                A = np.zeros((p, rank))
                for k in range(p):
                    for t in range(rank):
                        A[k, t] = V_r[j, k, t] * W_r[k, i, t]
                b = np.ones(p)
                # Solve A x = b
                try:
                    U[i, j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V
        # For each j, k, we have n equations (for i=0..n-1):
        # sum_t V[j,k,t] * (U[i,j,t] * W[k,i,t]) = 1.
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for t in range(rank):
                        A[i, t] = U_r[i, j, t] * W_r[k, i, t]
                b = np.ones(n)
                try:
                    V[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update W
        # For each k, i, we have m equations (for j=0..m-1):
        # sum_t W[k,i,t] * (U[i,j,t] * V[j,k,t]) = 1.
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for t in range(rank):
                        A[j, t] = U_r[i, j, t] * V_r[j, k, t]
                b = np.ones(m)
                try:
                    W[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            if _verify(U, V, W, n, m, p, tol=1e-4):
                break

    if _verify(U, V, W, n, m, p, tol=1e-5):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try direct decomposition at increasing ranks
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try ranks from lower_bound to schoolbook rank
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds for this rank
        seeds_to_try = 10 if rank <= 35 else 5
        for seed in range(seeds_to_try):
            if time.time() > deadline - 30:
                break
            
            res = _find_decomposition(n, m, p, rank, max_iters=300, seed=seed)
            if res is not None:
                U, V, W = res
                
                # Verify with tight tolerance
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    # Try to polish with even tighter tolerance
                    U_pol, V_pol, W_pol = _polish_decomposition(U, V, W, n, m, p, tol=1e-10, max_iters=200)
                    if U_pol is not None and _verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-8):
                        U, V, W = U_pol, V_pol, W_pol
                    
                    # Try to round to integers or half-integers
                    for mult in [1, 2]:
                        U_r = np.round(U * mult) / mult
                        V_r = np.round(V * mult) / mult
                        W_r = np.round(W * mult) / mult
                        if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                            U, V, W = U_r, V_r, W_r
                            break
                    
                    # This is a valid decomposition at this rank
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
                    # Found a decomposition, return it (lower rank is better)
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
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

def _polish_decomposition(U, V, W, n, m, p, tol=1e-10, max_iters=200):
    """
    High-precision polish pass for a near-exact decomposition.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        # Update U with tight tolerance
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for t in range(rank):
                        A[k, t] = V_r[j, k, t] * W_r[k, i, t]
                b = np.ones(p)
                try:
                    U[i, j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for t in range(rank):
                        A[i, t] = U_r[i, j, t] * W_r[k, i, t]
                b = np.ones(n)
                try:
                    V[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for t in range(rank):
                        A[j, t] = U_r[i, j, t] * V_r[j, k, t]
                b = np.ones(m)
                try:
                    W[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence with tight tolerance
        if it % 5 == 0:
            if _verify(U, V, W, n, m, p, tol=tol):
                break
    
    if _verify(U, V, W, n, m, p, tol=tol):
        return U, V, W
    return None, None, None
