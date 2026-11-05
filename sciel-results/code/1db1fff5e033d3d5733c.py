"""Matrix multiplication 3x3x6 rank-minimal decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
    # The multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise
    
    T = np.einsum('ak,bk,ck->abc', U_np, V_np, W_np)
    
    # Check diagonal elements (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check off-diagonal elements (should be 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i, j, k) == (i2, j2, k2):
                                continue
                            b = j2 * p + k2
                            c = k2 * n + i2
                            if abs(T[a, b, c]) > tol:
                                return False
    
    return True

def _round_to_halfint(x):
    """Round to nearest half-integer."""
    return 2.0 * np.round(x / 2.0) / 2.0

def _try_rank_with_optimization(n, m, p, target_rank, max_time=30.0, seed=None):
    """
    Try to find a rank-`target_rank` decomposition using gradient-based optimization
    with structured initialization exploiting the block structure.
    """
    if seed is not None:
        np.random.seed(seed)
    
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    # The 3x3x6 problem has a natural block structure:
    # k=0,1,2 (first block) and k=3,4,5 (second block)
    # Each block is essentially a 3x3x3 multiplication
    # We can initialize using two copies of a 3x3x3 decomposition
    
    # Known rank for 3x3x3: at most 23 (Lader), known best is 23
    # For 3x3x6, we can use 2*23 = 46 as an upper bound
    # But we want to find something better
    
    # Strategy: use the block structure to initialize
    # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
    
    # For each block, we need a 3x3x3 decomposition
    # Initialize with random values, then optimize
    
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Split rank into two halves for the two blocks
    half_rank = target_rank // 2
    rem = target_rank % 2
    
    for block in range(2):
        start = block * half_rank + min(block, rem)
        end = start + half_rank + (1 if block < rem else 0)
        block_size = end - start
        
        if block_size <= 0:
            continue
        
        # Random initialization for this block
        U_block = np.random.randn(nm, block_size) * 0.5
        V_block = np.random.randn(mp, block_size) * 0.5
        W_block = np.random.randn(pn, block_size) * 0.5
        
        # Only activate relevant indices for this block
        k_indices = [3 * block + b for b in range(3)]
        
        # Zero out V rows not in this block's k range
        V_block_zero = np.zeros_like(V_block)
        for k in k_indices:
            for j in range(m):
                V_block_zero[j * p + k, :] = V_block[j * p + k, :]
        V_block = V_block_zero
        
        # Zero out W rows not in this block's k range
        W_block_zero = np.zeros_like(W_block)
        for k in k_indices:
            for i in range(n):
                W_block_zero[k * n + i, :] = W_block[k * n + i, :]
        W_block = W_block_zero
        
        U[start:end] = U_block
        V[start:end] = V_block
        W[start:end] = W_block
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Gradient descent with adaptive learning rate
    lr = 0.1
    patience = 0
    max_patience = 50
    
    for it in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Compute loss and gradients efficiently
        # T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
        # We want T[a,b,c] = delta_{a,b,c} where delta is 1 if (a,b,c) corresponds to valid (i,j,k)
        
        # Build target tensor
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        # Current tensor
        T = np.einsum('ak,bk,ck->abc', U, V, W)
        
        # Loss
        loss = np.sum((T - target) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1
        
        if best_loss < 1e-10:
            break
        
        if patience > max_patience:
            # Try a new random restart
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            for block in range(2):
                start = block * half_rank + min(block, rem)
                end = start + half_rank + (1 if block < rem else 0)
                block_size = end - start
                if block_size <= 0:
                    continue
                U_block = np.random.randn(nm, block_size) * 0.5
                V_block = np.random.randn(mp, block_size) * 0.5
                W_block = np.random.randn(pn, block_size) * 0.5
                k_indices = [3 * block + b for b in range(3)]
                V_block_zero = np.zeros_like(V_block)
                for k in k_indices:
                    for j in range(m):
                        V_block_zero[j * p + k, :] = V_block[j * p + k, :]
                V_block = V_block_zero
                W_block_zero = np.zeros_like(W_block)
                for k in k_indices:
                    for i in range(n):
                        W_block_zero[k * n + i, :] = W_block[k * n + i, :]
                W_block = W_block_zero
                U[start:end] = U_block
                V[start:end] = V_block
                W[start:end] = W_block
            patience = 0
            lr = 0.1
            continue
        
        # Compute gradients
        # dL/dU[a,k] = 2 * sum_{b,c} (T[a,b,c] - target[a,b,c]) * V[b,k] * W[c,k]
        # dL/dV[b,k] = 2 * sum_{a,c} (T[a,b,c] - target[a,b,c]) * U[a,k] * W[c,k]
        # dL/dW[c,k] = 2 * sum_{a,b} (T[a,b,c] - target[a,b,c]) * U[a,k] * V[b,k]
        
        err = T - target
        
        grad_U = np.einsum('abc,bk,ck->ak', err, V, W) * 2
        grad_V = np.einsum('abc,ak,ck->bk', err, U, W) * 2
        grad_W = np.einsum('abc,ak,bk->ck', err, U, V) * 2
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if loss < best_loss * 0.9:
            lr = min(lr * 1.1, 1.0)
        else:
            lr = max(lr * 0.95, 0.001)
    
    # Try rounding to half-integers and check
    U_rounded = _round_to_halfint(best_U)
    V_rounded = _round_to_halfint(best_V)
    W_rounded = _round_to_halfint(best_W)
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-6):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist(), "rank": target_rank}
    
    # Also try without rounding
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-6):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": target_rank}
    
    return None

def _try_structured_strassen(n, m, p, target_rank, max_time=20.0):
    """
    Try a structured approach based on Strassen-like decomposition.
    For 3x3x6, we can view it as two 3x3x3 blocks.
    Use known 3x3x3 decompositions and combine.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # For 3x3x3, the minimal rank is 23 (Lader 1971)
    # We'll try to find a rank-23 decomposition for 3x3x3, then use it twice
    
    # Known 3x3x3 decomposition (Lader's rank-23)
    # This is complex; let's try a simpler approach first
    
    # Try rank 18 (lower bound) with structured initialization
    # The lower bound for 3x3x6 is max(9, 18, 18) = 18
    
    # Use a different initialization: based on the fact that
    # C = A * B where A is 3x3 and B is 3x6
    # We can write B as [B1, B2] where B1, B2 are 3x3
    # Then C = A*B1 + A*B2
    # Each A*B is a 3x3x3 multiplication
    
    # For each 3x3x3, use a rank-r decomposition
    # Total rank = 2r
    
    # Try r from 9 upward
    for r in range(9, min(target_rank // 2 + 1, 24)):
        if time.time() - start_time > max_time:
            break
        
        total_rank = 2 * r
        if total_rank > target_rank:
            continue
        
        # Initialize with two independent 3x3x3 decompositions
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        # Block 1: k in {0,1,2}
        # Block 2: k in {3,4,5}
        
        for block in range(2):
            col_start = block * r
            col_end = (block + 1) * r
            
            # Random initialization
            U[:, col_start:col_end] = np.random.randn(nm, r) * 0.5
            V[:, col_start:col_end] = np.random.randn(mp, r) * 0.5
            W[:, col_start:col_end] = np.random.randn(pn, r) * 0.5
            
            # Zero out irrelevant rows
            k_indices = [3 * block + b for b in range(3)]
            
            V_block_zero = np.zeros((mp, r))
            for k in k_indices:
                for j in range(m):
                    V_block_zero[j * p + k, :] = V[:, col_start:col_end][j * p + k, :]
            V[:, col_start:col_end] = V_block_zero
            
            W_block_zero = np.zeros((pn, r))
            for k in k_indices:
                for i in range(n):
                    W_block_zero[k * n + i, :] = W[:, col_start:col_end][k * n + i, :]
            W[:, col_start:col_end] = W_block_zero
        
        # Optimize
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        lr = 0.1
        
        for it in range(5000):
            if time.time() - start_time > max_time:
                break
            
            T = np.einsum('ak,bk,ck->abc', U, V, W)
            err = T - target
            loss = np.sum(err ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-10:
                break
            
            grad_U = np.einsum('abc,bk,ck->ak', err, V, W) * 2
            grad_V = np.einsum('abc,ak,ck->bk', err, U, W) * 2
            grad_W = np.einsum('abc,ak,bk->ck', err, U, V) * 2
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < best_loss * 0.9:
                lr = min(lr * 1.1, 1.0)
            else:
                lr = max(lr * 0.95, 0.001)
        
        # Check
        U_r = _round_to_halfint(best_U)
        V_r = _round_to_halfint(best_V)
        W_r = _round_to_halfint(best_W)
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist(), "rank": total_rank}
        
        if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-6):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": total_rank}
    
    return None

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    lower_bound = max(n * m, m * p, p * n)
    
    # Try structured approach first
    result = _try_structured_strassen(n, m, p, 46, max_time=40.0)
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Try gradient-based optimization with various ranks
    for target_rank in range(lower_bound, min(schoolbook_rank, 40)):
        if time.time() > DEADLINE - 15:
            break
        
        result = _try_rank_with_optimization(n, m, p, target_rank, max_time=20.0, seed=target_rank)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}
