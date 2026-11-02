"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Strategy:
    - Start with schoolbook (rank 72)
    - Try numerical optimization to find lower-rank decompositions
    - Verify with exact rational arithmetic
    - Return best verified result
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    def verify_exact(U, V, W):
        """
        Verify decomposition using exact Fraction arithmetic.
        U: list of lists (n*m x rank)
        V: list of lists (m*p x rank)  
        W: list of lists (p*n x rank)
        Returns True if exact decomposition.
        """
        rank = len(U[0]) if U else 0
        
        # Convert to Fractions
        Uf = [[Fraction(str(x)) for x in row] for row in U]
        Vf = [[Fraction(str(x)) for x in row] for row in V]
        Wf = [[Fraction(str(x)) for x in row] for row in W]
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = Fraction(0)
                    for r in range(rank):
                        val += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if val != 1:
                        return False
        
        # Check zero entries (sample some to save time, but for small dims check all)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = Fraction(0)
                        for r in range(rank):
                            val += Uf[a][r] * Vf[b][r] * Wf[c][r]
                        if val != 0:
                            return False
        return True
    
    def verify_numeric(U, V, W, tol=1e-9):
        """Quick numerical verification."""
        rank = len(U[0]) if U else 0
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        
        # Check nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(Ua[a] * Va[b] * Wa[c])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check some zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = np.sum(Ua[a] * Va[b] * Wa[c])
                        if abs(val) > tol:
                            return False
        return True
    
    def schoolbook():
        """Return schoolbook decomposition with rank n*m*p = 72."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank(rank, num_restarts=50, max_iter=2000):
        """
        Try to find a rank-r decomposition using random initialization and
        coordinate descent optimization.
        """
        if not check_time():
            return None
        
        best_loss = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if not check_time():
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Normalize to help convergence
            for r in range(rank):
                norm_U = np.linalg.norm(U[:, r])
                norm_V = np.linalg.norm(V[:, r])
                norm_W = np.linalg.norm(W[:, r])
                if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                    scale = (norm_U * norm_V * norm_W) ** (1.0 / 3.0)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Coordinate descent
            lr = 0.01
            for it in range(max_iter):
                if not check_time():
                    break
                
                # Compute gradient with respect to U
                # Loss = sum over all (a,b,c) of (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # dLoss/dU[a,r] = -2 * sum_{b,c: T[a,b,c]!=0} T[a,b,c] * V[b,r] * W[c,r] + ...
                
                # For efficiency, only compute gradient for nonzero tensor entries
                # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
                
                # Update U
                dU = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Contribution to gradient
                            for r in range(rank):
                                pred = np.sum(U[a] * V[b] * W[c])
                                dU[a, r] += -2.0 * (1.0 - pred) * V[b, r] * W[c, r]
                
                U -= lr * dU
                
                # Update V
                dV = np.zeros_like(V)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            for r in range(rank):
                                pred = np.sum(U[a] * V[b] * W[c])
                                dV[b, r] += -2.0 * (1.0 - pred) * U[a, r] * W[c, r]
                
                V -= lr * dV
                
                # Update W
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            for r in range(rank):
                                pred = np.sum(U[a] * V[b] * W[c])
                                dW[c, r] += -2.0 * (1.0 - pred) * U[a, r] * V[b, r]
                
                W -= lr * dW
                
                # Compute loss every 100 iterations
                if it % 100 == 0:
                    loss = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                pred = np.sum(U[a] * V[b] * W[c])
                                loss += (1.0 - pred) ** 2
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (U.copy(), V.copy(), W.copy())
                    
                    if loss < 1e-6:
                        break
            
            # Check if this restart found a good solution
            if best_params is not None and best_loss < 1e-5:
                U_best, V_best, W_best = best_params
                # Round to nearest 0.5 and verify
                U_rounded = np.round(U_best * 2) / 2
                V_rounded = np.round(V_best * 2) / 2
                W_rounded = np.round(W_best * 2) / 2
                
                if verify_numeric(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    if verify_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Start with schoolbook
    best_result = schoolbook()
    if not verify_numeric(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Try ranks from 24 upward
    for rank in range(24, best_rank):
        if not check_time():
            break
        
        result = try_rank(rank, num_restarts=20, max_iter=1000)
        if result is not None:
            best_rank = rank
            best_result = result
            break  # Found a valid decomposition, stop
    
    return best_result
