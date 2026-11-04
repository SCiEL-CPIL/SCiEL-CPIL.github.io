import numpy as np
import time

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

def _construct_rank48(n, m, p):
    """
    Construct a rank-48 decomposition for 3x5x6 matrix multiplication.
    
    Strategy: Use structured tensor products.
    - Decompose the 3x5x6 multiplication into blocks.
    - Use Strassen-like tricks on sub-blocks to reduce rank.
    
    For 3x5x6, we can think of it as:
    - 3 rows, 5 cols, 6 rows of second matrix
    - Split the 5 columns into 3+2, and 6 rows into 3+3
    - This gives us a 3x3x3 part and 3x2x3 part, etc.
    
    Actually, let's use a more systematic approach:
    The multiplication tensor M_{3,5,6} can be decomposed as:
    M_{3,5,6} = M_{3,3,6} ⊗ M_{1,2,1} + M_{3,2,6} ⊗ M_{1,1,1}
    But this doesn't directly help.
    
    Let me try a different structured approach:
    Split the 5 middle dimension into 3+2.
    Then C = A * B where A is 3x5, B is 5x6.
    A = [A1 | A2] where A1 is 3x3, A2 is 3x2
    B = [B1; B2] where B1 is 3x6, B2 is 2x6
    C = A1*B1 + A2*B2
    
    Rank of M_{3,3,6} + Rank of M_{3,2,6}
    
    For M_{3,3,6}: split 6 into 3+3, get M_{3,3,3} + M_{3,3,3}
    Rank(M_{3,3,3}) = 7 (Strassen)
    So Rank(M_{3,3,6}) <= 14
    
    For M_{3,2,6}: split 6 into 3+3, get M_{3,2,3} + M_{3,2,3}
    Rank(M_{3,2,3}): we can use M_{3,2,3} <= M_{3,2,2} + M_{3,2,1}
    Rank(M_{3,2,2}) <= 3*2*2 = 12, but can we do better?
    Actually M_{3,2,2} has rank at most 3*2*2=12, but likely less.
    
    This is getting complicated. Let me just try to find a known decomposition.
    
    Alternative: Use the fact that we can embed smaller known decompositions.
    
    Let me try a direct construction with rank 48:
    We need 48 rank-1 terms.
    
    Actually, let me try a simpler approach: just construct the schoolbook 
    and see if we can verify it works, then try to find a lower rank.
    """
    size_u = n * m  # 15
    size_v = m * p  # 30
    size_w = p * n  # 18
    
    # For now, return None to indicate this needs more work
    return None

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

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try rank-48 structured decomposition
    result48 = _construct_rank48(n, m, p)
    if result48 is not None:
        U, V, W = result48
        if _verify_decomposition(U, V, W, T):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Try to find lower rank via optimization
    candidate_ranks = [48, 50, 52, 54, 56, 58, 60, 65, 70, 75, 80, 85, 89, 90]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple restarts
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize U, V, W
            U = np.random.randn(size_u, rank) * 0.1
            V = np.random.randn(size_v, rank) * 0.1
            W = np.random.randn(size_w, rank) * 0.1
            
            # Use alternating least squares approach
            # Fix V and W, solve for U, etc.
            for step in range(50):
                if time.time() > deadline - 10.0:
                    break
                    
                # Solve for U: For each r, U[:, r] should minimize ||T - X||^2
                # This is equivalent to: U[:, r] = argmin ||T - sum_r U[:,r] outer(V[:,r], W[:,r])||
                # Linear in U[:, r] for fixed r
                # dU/dr = sum_{I,J,K} (T[I,J,K] - X[I,J,K]) * V[J,r] * W[K,r]
                # But we can solve this more efficiently
                
                # Compute current X
                X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
                diff = T - X
                
                # Update U: For each r, U[:, r] -= lr * dU[:, r]
                # dU[I, r] = sum_{J,K} diff[I, J, K] * V[J, r] * W[K, r]
                dU = np.einsum('ijk,jr,kr->ir', diff, V, W, optimize=True)
                dV = np.einsum('ijk,ir,kr->jr', diff, U, W, optimize=True)
                dW = np.einsum('ijk,ir,jr->kr', diff, U, V, optimize=True)
                
                # Adaptive learning rate
                loss = np.sum(diff**2)
                lr = 0.05 / (1 + step)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-8:
                    break
            
            # Try to round to integers or half-integers
            for mult in [1, 2]:
                U_rounded = np.rint(U * mult) / mult
                V_rounded = np.rint(V * mult) / mult
                W_rounded = np.rint(W * mult) / mult
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
