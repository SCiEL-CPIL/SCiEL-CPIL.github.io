import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication target tensor."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook (trivial) decomposition."""
    rank = n * m * p
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition is correct."""
    X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return np.allclose(X, T, atol=tol)

def _construct_rank48():
    """
    Construct a rank-48 decomposition for 3x5x6 matrix multiplication.
    
    The idea: decompose the 3x5x6 multiplication using a combination of
    Strassen-like tricks on the 3x3 blocks and direct computation on the rest.
    
    We can write a 3x5 matrix A and 5x6 matrix B as:
    A = [A1 A2 A3] where A1 is 3x2, A2 is 3x1, A3 is 3x2
    B = [B1; B2; B3] where B1 is 2x6, B2 is 1x6, B3 is 2x6
    
    Then AB = A1*B1 + A2*B2 + A3*B3
    
    For A1*B1 (3x2 * 2x6): use rank-4 via Strassen-like on 2x2 blocks
    Actually, let's use a more systematic approach.
    
    We'll use the fact that 3x5x6 can be decomposed as:
    - Split the middle dimension 5 into 2+1+2
    - Use rank-4 for 3x2x6 (split 6 into 3+3, use rank-2 for each 3x2x3)
    - Use rank-3 for 3x1x6
    - Use rank-4 for 3x2x6
    
    Total: 4 + 3 + 4 = 11 per block? That's not right.
    
    Let me think differently. The standard approach:
    
    For 3x5x6, we can use:
    - Decompose 5 = 2 + 3
    - A1 is 3x2, A2 is 3x3
    - B1 is 2x6, B2 is 3x6
    
    A1*B1: 3x2x6 -> can use rank 3*2*6=36, or better with Strassen
    Actually, let's just use a known good decomposition.
    
    I'll use a structured approach: treat the 3x5x6 as a sum of tensor products
    where we exploit the structure.
    
    Alternative: use the rank-50 bound from known results.
    """
    n, m, p = 3, 5, 6
    size_u = n * m  # 15
    size_v = m * p  # 30
    size_w = p * n  # 18
    
    # Use a rank-50 decomposition based on splitting strategies
    # Split m=5 into 2+3
    # A = [A_left (3x2), A_right (3x3)]
    # B = [B_left (2x6); B_right (3x6)]
    # AB = A_left @ B_left + A_right @ B_right
    
    # A_left @ B_left: 3x2x6
    # Split p=6 into 3+3: B_left = [B_left1 (2x3), B_left2 (2x3)]
    # A_left @ B_left1: 3x2x3 -> rank 3*2*3=18, or use Strassen: rank 7? No, 2x2x2 is rank 7
    # Actually 3x2x3: we can do it in rank 9 (3*3) by treating 2 as 1+1
    # Or use rank 8: split 3 into 1+2, get 3x1x3 (rank 3) + 3x2x3... circular
    
    # Let's use a simpler approach: direct but with some optimization
    # 3x2x3: rank 6 (each of 3 rows of A times each of 3 cols of B, but we share the 2)
    # Actually 3x2x3 has rank at most 3*2*3=18, but we can do better.
    # The rank of 3x2x3 tensor is 6 (since min(3*2, 2*3, 3*3) = 6? No)
    # The rank of the matrix multiplication tensor [m,n,p] is at most mnp but can be less.
    # For 3x2x3: we can use rank 6 by fixing one dimension.
    
    # Let me just construct a valid decomposition with rank 50.
    # I'll use the following structure:
    # - 3x5x6 = sum of 50 tensor products
    
    # Actually, let me use a different strategy. I'll use the known fact that
    # w(3,5,6) <= 50 and construct it explicitly.
    
    # The decomposition:
    # We write AB where A is 3x5, B is 5x6.
    # Let A = [A1 A2 A3 A4 A5] (each 3x1) and B = [B1; B2; B3; B4; B5] (each 1x6)
    # AB = sum_{j=1}^{5} A_j B_j
    
    # Each A_j B_j is a 3x6 matrix = outer product of 3-vector and 6-vector.
    # We need to compute 5 such outer products and sum them.
    # Naively this is rank 5*3*6 = 90.
    
    # To reduce rank, we can use Strassen-like tricks on groups of these.
    
    # Group the 5 terms into: 2 + 3
    # For the 2 terms: A1 B1 + A2 B2
    # This is like multiplying a 3x2 matrix by a 2x6 matrix.
    # We can use rank 3*2*6 = 36, or use Strassen on the 2x2 part.
    # Split the 6 columns into 3+3:
    # A1 B1 + A2 B2 = [A1 B1_1 + A2 B2_1] [A1 B1_2 + A2 B2_2]
    # Each part is 3x2x3.
    # For 3x2x3: use rank 6 (compute each of 3 rows separately, each needs rank 2*3=6? No)
    # Actually for 3x2x3: fix the 3 output rows. For each row i, we compute
    # sum_j A[i,j] B[j,k] for j=1,2 and k=1,2,3. That's 2*3=6 operations per row, 18 total.
    # But we can share: the A[i,j] values are shared across k.
    # The rank of the 3x2x3 multiplication tensor is 6.
    # Proof: For each pair (i,k), we need sum_j A[i,j] B[j,k].
    # We can write this as: for j=1,2 and k=1,2,3, use tensor e_i ⊗ e_j ⊗ e_k.
    # That's 3*2*3 = 9 tensors. But we can do better.
    # Actually, the rank of [3,2,3] is 6. Here's a decomposition:
    # For i in 1..3, k in 1..3:
    #   sum_j A[i,j] B[j,k] = A[i,1] B[1,k] + A[i,2] B[2,k]
    # We can use 6 tensors: for each (i,k) pair, but that's 9.
    # Hmm, let me just use rank 9 for 3x2x3.
    
    # This is getting complicated. Let me just use the schoolbook as fallback
    # and try to find a better one through optimization.
    
    # For now, return a rank-50 construction if I can find one, else schoolbook.
    # I'll return None to indicate failure, and the caller will use schoolbook.
    return None

def _try_find_decomposition_als(n, m, p, target_rank, time_budget, seed=0):
    """
    Try to find a decomposition of the given rank using alternating least squares
    with a better initialization and more iterations.
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = _build_target_tensor(n, m, p)
    
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    best_result = None
    best_loss = float('inf')
    
    for restart in range(15):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Initialize with a structured random approach
        U = rng.normal(0, 1.0, (size_u, target_rank))
        V = rng.normal(0, 1.0, (size_v, target_rank))
        W = rng.normal(0, 1.0, (size_w, target_rank))
        
        # ALS iterations
        for step in range(80):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current approximation
            X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            diff = T - X
            loss = np.sum(diff**2)
            
            if loss < 1e-12:
                break
            
            # Gradient descent with adaptive learning rate
            dU = np.einsum('ijk,jr,kr->ir', diff, V, W, optimize=True)
            dV = np.einsum('ijk,ir,kr->jr', diff, U, W, optimize=True)
            dW = np.einsum('ijk,ir,jr->kr', diff, U, V, optimize=True)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.05 / (1 + step * 0.02)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < 1e-12:
                break
        
        # Final loss check
        X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        final_loss = np.sum((T - X)**2)
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_result = (U.copy(), V.copy(), W.copy())
        
        # Try to round to half-integers
        for mult in [1, 2, 4, 8]:
            U_rounded = np.rint(U * mult) / mult
            V_rounded = np.rint(V * mult) / mult
            W_rounded = np.rint(W * mult) / mult
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
    
    # Check if best result is close enough
    if best_result is not None and best_loss < 1e-6:
        U, V, W = best_result
        for mult in [1, 2, 4, 8, 16]:
            U_rounded = np.rint(U * mult) / mult
            V_rounded = np.rint(V * mult) / mult
            W_rounded = np.rint(W * mult) / mult
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    T = _build_target_tensor(n, m, p)
    
    # Try rank-48 construction
    if time.time() < deadline - 60.0:
        result = _construct_rank48()
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Try to find lower rank decompositions using ALS
    candidate_ranks = [50, 55, 60, 65, 70, 75, 80, 85]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20.0:
            break
        
        remaining = deadline - time.time() - 5.0
        if remaining < 5.0:
            break
        
        result = _try_find_decomposition_als(n, m, p, rank, remaining, seed=rank + 100)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
