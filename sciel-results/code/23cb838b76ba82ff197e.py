"""
Matrix multiplication 3x3x8 rank-minimal decomposition search.

Strategy:
- The tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication (3x3 by 3x8)
  has entries T[i,j,k] = 1 if the (i,k) entry of A@B equals sum_j A[i,j]*B[j,k].
  Actually T[i,j,k] = 1 for all (i,j,k) since A[i,j]*B[j,k] contributes to C[i,k].
  
  Wait, the tensor is: T[i,j,k] = 1 for all i,j,k? No.
  The matrix multiplication tensor is defined by:
  C[i,k] = sum_j A[i,j] * B[j,k]
  So the tensor T has T[i,j,k] = 1 for all (i,j,k) in [3]x[3]x[8].
  
  Actually no. The standard matrix multiplication tensor for (n,m,p) is:
  T_{i,j,k} = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k].
  Since every (i,j,k) triple has A[i,j]*B[j,k] contributing to C[i,k],
  T[i,j,k] = 1 for ALL i,j,k.
  
  Wait, that's not right either. Let me think again.
  
  The matrix multiplication tensor M_{n,m,p} is a tensor in R^{n*m} x R^{m*p} x R^{p*n}
  such that for matrices A (n x m), B (m x p), C = A @ B (n x p):
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  The tensor is defined as:
  T[a, b, c] where a indexes (i,j) in A, b indexes (j,k) in B, c indexes (i,k) in C.
  T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k.
  Otherwise T[a,b,c] = 0.
  
  So T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
  All other entries are 0.
  
  For n=3, m=3, p=8:
  - A has 9 entries (i,j) for i in 0..2, j in 0..2
  - B has 24 entries (j,k) for j in 0..2, k in 0..7
  - C has 24 entries (i,k) for i in 0..2, k in 0..7
  
  The tensor T has 9*24*24 = 5184 entries, of which 3*3*8 = 72 are 1 (the schoolbook terms).
  
  A rank-r decomposition gives U (9 x r), V (24 x r), W (24 x r) such that:
  sum_{r=0}^{r-1} U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
  
  for all a, b, c.
"""

import numpy as np
import random
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook():
    """Standard rank n*m*p = 72 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_target_tensor():
    """Build the target tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that U, V, W decompose the tensor T."""
    n, m, p = DIMENSIONS
    if T is None:
        T = _build_target_tensor()
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Compute sum_r U[:,r] outer V[:,r] outer W[:,r]
    # This is a tensor of shape (n*m, m*p, p*n)
    result = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    return np.allclose(result, T, atol=1e-6)

def _verify_fast(U, V, W):
    """Fast verification using matrix operations."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check all 72 non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(Ua[a, :], Va[b, :] * Wa[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check that some zero entries are actually zero
    # (sample a few)
    for _ in range(100):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        # Skip if this is a non-zero entry
        found = False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i * m + j == a and j * p + k == b and k * n + i == c:
                        found = True
        if not found:
            val = np.dot(Ua[a, :], Va[b, :] * Wa[c, :])
            if abs(val) > 1e-6:
                return False
    return True

def _try_rank_reduction(target_rank, max_time=100.0, seed=42):
    """
    Try to find a decomposition of the given rank using gradient-based optimization.
    
    We parameterize the problem as minimizing:
    f(U,V,W) = sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
    
    This is non-convex, so we use multiple random restarts.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    rng = np.random.RandomState(seed)
    
    best_error = float('inf')
    best_solution = None
    start_time = time.time()
    
    # Number of restarts based on time budget
    # Each restart takes roughly 1-5 seconds
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > max_time - 10:
            break
        
        # Initialize with random values
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Gradient descent
        lr = 0.01
        patience = 200
        no_improve = 0
        prev_error = float('inf')
        
        for step in range(3000):
            if time.time() - start_time > max_time - 5:
                break
            
            # Compute gradient
            # f = sum_{abc} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
            # df/dU[a,r] = 2 * sum_{b,c} (sum_r' U[a,r']V[b,r']W[c,r'] - T[a,b,c]) * V[b,r] * W[c,r]
            
            # Efficient computation:
            # Let S[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
            # Then df/dU[a,r] = 2 * sum_{b,c} (S[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            
            # Compute S using einsum
            S = np.einsum('ar,br,cr->abc', U, V, W)
            residual = S - T
            
            # Gradient for U
            # dU[a,r] = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error
            error = np.sum(residual ** 2)
            
            if error < prev_error - 1e-8:
                no_improve = 0
                prev_error = error
            else:
                no_improve += 1
            
            if no_improve > patience:
                break
            
            # Decay learning rate
            if step % 500 == 0 and step > 0:
                lr *= 0.8
        
        # Check final error
        S = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((S - T) ** 2)
        
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if best_error < 1e-4:
            break
    
    if best_solution is not None and best_error < 1e-4:
        U, V, W = best_solution
        # Round to nearest 0.5 for cleaner output
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        # Verify after rounding
        S = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(S, T, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_decomposition():
    """
    Try a structured decomposition that exploits the block structure.
    
    For 3x3x8, we can think of the 8 columns as two groups of 4.
    Or we can use the fact that the tensor has a specific sparsity pattern.
    
    Key insight: The tensor T[a,b,c] is non-zero only when:
    a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    
    This means the non-zero pattern is very structured.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Try various structured initializations
    # One approach: use the schoolbook as starting point and try to merge terms
    
    # Another approach: exploit the fact that for fixed i, the terms
    # (i,j,k) for all j,k form a rank-3 sub-tensor
    # And for fixed k, the terms (i,j,k) for all i,j form a rank-3 sub-tensor
    
    # Let's try a block decomposition:
    # Split the 8 k-values into two groups of 4
    # For each group, use a rank-12 decomposition (3x3x4 schoolbook)
    # Total rank: 24
    # But we can share some terms between groups
    
    # Actually, let's try to find a decomposition with rank < 72
    # using the structure more directly
    
    # For each i (3 values), we need to handle:
    # C[i,k] = sum_j A[i,j] * B[j,k] for k = 0..7
    # This is a 3x8 matrix product for each i
    
    # The tensor for fixed i has shape (3, 24, 8) where:
    # - First dim: j (3 values)
    # - Second dim: (j,k) pairs (24 values)  
    # - Third dim: k (8 values)
    # Non-zero when: a=j, b=j'*p+k, c=k where j=j'
    # So T[j, j'*8+k, k] = 1 if j=j', else 0
    
    # This is a 3x24x8 tensor with 3*8 = 24 non-zero entries
    # Its rank is at most 24 (schoolbook)
    
    # For all 3 i's, total schoolbook rank is 72
    
    # Can we do better? The three i-blocks are independent in the sense that
    # they involve different rows of A and different rows of C.
    # But they share the same columns of B.
    
    # One approach: use a shared factor for B
    # Let V be shared across all i-blocks
    # Then we need U_i and W_i for each i
    # This doesn't obviously help reduce rank
    
    # Let's just try the optimization approach with various target ranks
    return None

def _try_landsberg_random(target_rank, max_time=80.0, seed=0):
    """
    Random search for a decomposition of the given rank.
    Uses random perturbations of the schoolbook.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Start from schoolbook and try to reduce rank
    # by setting some columns to zero and adjusting others
    
    # First, try direct random initialization
    best_error = float('inf')
    best_solution = None
    
    # Try multiple random initializations
    n_trials = 20
    
    for trial in range(n_trials):
        if time.time() - start_time > max_time - 5:
            break
        
        U = rng.randn(n * m, target_rank) * 0.3
        V = rng.randn(m * p, target_rank) * 0.3
        W = rng.randn(p * n, target_rank) * 0.3
        
        # Simple gradient descent with momentum
        lr = 0.005
        momentum = 0.9
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        
        for step in range(2000):
            if time.time() - start_time > max_time - 3:
                break
            
            S = np.einsum('ar,br,cr->abc', U, V, W)
            residual = S - T
            
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            U += vel_U
            V += vel_V
            W += vel_W
        
        S = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((S - T) ** 2)
        
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if best_error < 1e-4:
            break
    
    if best_solution is not None and best_error < 1e-4:
        U, V, W = best_solution
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        S = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(S, T, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _build_block_decomposition():
    """
    Build a decomposition by splitting the 8 columns into blocks.
    
    Split into blocks of size 3, 3, 2:
    - Block 1: k=0,1,2 -> 3x3x3 tensor, rank 27
    - Block 2: k=3,4,5 -> 3x3x3 tensor, rank 27  
    - Block 3: k=6,7 -> 3x3x2 tensor, rank 18
    Total: 72 (same as schoolbook)
    
    But we can try to share factors between blocks.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Try to find a shared structure
    # The key observation is that all blocks share the same A matrix structure
    # and the B matrix has the same row structure for each k
    
    # Let's try a decomposition where:
    # - U has 9 rows (for A)
    # - V has 24 rows (for B)
    # - W has 24 rows (for C)
    
    # Try rank 54, 48, 42, 36, 30, 24
    for target_rank in [54, 48, 42, 36, 30, 24]:
        result = _try_rank_reduction(target_rank, max_time=30.0, seed=target_rank)
        if result is not None:
            return result
    
    return None

def _try_explicit_construction():
    """
    Try to explicitly construct a low-rank decomposition.
    
    Key insight: For 3x3x8, we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's define:
    U[a, r] where a = (i,j), r indexes the rank
    V[b, r] where b = (j,k), r indexes the rank
    W[c, r] where c = (i,k), r indexes the rank
    
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    One approach: use the identity matrix structure.
    For each (i,j), we need the contribution to be 1 for all k.
    This means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k.
    
    If we set U[i*m+j, r] = 1 for r = i*3+j (one rank per (i,j) pair),
    and V[j*p+k, r] = delta_{r, i*3+j} for the right r...
    This gives the schoolbook.
    
    To reduce rank, we need to share ranks across different (i,j) pairs.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Try a structured construction
    # Use 9 ranks for the "diagonal" and additional ranks for cross terms
    
    # Actually, let's try a different approach:
    # Use the fact that 3x3 matrix multiplication has rank 20 (Landsberg)
    # And 3x3x2 has rank 18 (schoolbook, possibly reducible)
    
    # For 3x3x3, the known minimal rank is 20 (not proven to be minimal, but best known)
    # For 3x3x8, we can use:
    # - Two 3x3x3 blocks (k=0,1,2 and k=3,4,5) with rank 20 each = 40
    # - One 3x3x2 block (k=6,7) with rank 12 = 12
    # Total: 52
    # But we can share some structure between the 3x3x3 blocks
    
    # Let's try to find a rank-50 or lower decomposition
    for target_rank in [52, 50, 48, 46, 44, 42, 40]:
        result = _try_rank_reduction(target_rank, max_time=25.0, seed=target_rank + 100)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    timeout = 120.0
    safety_margin = 15.0
    
    # 1. Build schoolbook as fallback (rank 72)
    schoolbook = _build_schoolbook()
    
    # 2. Try explicit construction with various target ranks
    if time.time() - start_time < timeout - safety_margin:
        explicit = _try_explicit_construction()
        if explicit is not None:
            if _verify_fast(explicit['U'], explicit['V'], explicit['W']):
                return explicit
    
    # 3. Try block decomposition
    if time.time() - start_time < timeout - safety_margin:
        block = _build_block_decomposition()
        if block is not None:
            if _verify_fast(block['U'], block['V'], block['W']):
                return block
    
    # 4. Try random search with various target ranks
    if time.time() - start_time < timeout - safety_margin:
        for target_rank in [50, 45, 40, 36, 32, 28]:
            if time.time() - start_time > timeout - safety_margin:
                break
            result = _try_landsberg_random(target_rank, max_time=20.0, seed=target_rank)
            if result is not None:
                if _verify_fast(result['U'], result['V'], result['W']):
                    return result
    
    # 5. Final attempt with direct optimization
    if time.time() - start_time < timeout - safety_margin:
        for target_rank in [48, 44, 40]:
            if time.time() - start_time > timeout - safety_margin:
                break
            result = _try_rank_reduction(target_rank, max_time=20.0, seed=target_rank + 500)
            if result is not None:
                if _verify_fast(result['U'], result['V'], result['W']):
                    return result
    
    return schoolbook

# EVOLVE-BLOCK-END
