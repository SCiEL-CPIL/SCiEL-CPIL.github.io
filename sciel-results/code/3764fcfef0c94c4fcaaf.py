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

def _strassen_3x3x3():
    """
    Return the 7 rank-1 tensors for Strassen's algorithm on 3x3x3.
    Each tensor is a triple (u, v, w) of 3-vectors.
    """
    # Strassen's 7 products for 2x2x2, embedded in 3x3x3
    # For 3x3, we can tile or use a direct construction
    # Actually, let's use the known rank-7 decomposition for 3x3x3
    
    # One approach: use the fact that M_{3,3,3} has rank 7
    # Construct using basis vectors
    
    # Let's use a systematic construction based on the multiplication table
    # C_{ik} = sum_j A_{ij} B_{jk}
    
    # Strassen-style for 3x3:
    # We can decompose 3x3 into overlapping 2x2 blocks
    
    # Actually, let me use a known rank-7 decomposition:
    # The 7 tensors correspond to:
    # P1 = A11 * B11
    # P2 = A12 * B21  
    # P3 = A21 * B12
    # P4 = A22 * B22
    # P5 = (A11+A22) * (B11+B22)
    # P6 = (A12+A22) * (B21-B11)
    # P7 = (A21-A11) * (B12+B22)
    
    # But this is for 2x2. For 3x3, we need a different approach.
    
    # Let me construct it directly using the standard basis
    # and the known fact that rank(M_{3,3,3}) = 7
    
    # Use a constructive approach:
    # The multiplication tensor for 3x3x3 has 27 non-zero entries
    # We need 7 rank-1 tensors that sum to it
    
    # Let me use a different strategy: construct the decomposition
    # by solving a linear system
    
    # Actually, for simplicity and correctness, let me use the schoolbook
    # for the sub-problems and see if we can get a good total rank
    
    return None  # Will be replaced with actual construction

def _construct_rank48(n, m, p):
    """
    Construct a structured decomposition for 3x5x6 matrix multiplication.
    
    Strategy: Split middle dimension 5 = 3+2, last dimension 6 = 3+3.
    C = A1*B1 + A2*B2 where:
      A1 is 3x3, A2 is 3x2
      B1 is 3x6, B2 is 2x6
    
    Then split B1 = [B1a | B1b] and B2 = [B2a | B2b] where each is x3.
    C = A1*B1a + A1*B1b + A2*B2a + A2*B2b
    
    For each sub-problem, use the best known decomposition:
    - M_{3,3,3}: rank 7 (Strassen)
    - M_{3,2,3}: need to find good rank
    
    For M_{3,2,3}: C_{ik} = sum_{j=1}^{2} A_{ij} B_{jk}
    This is like multiplying 3x2 by 2x3.
    Schoolbook rank: 3*2*3 = 18
    Can we do better? 
    
    Actually, let me use a more direct approach: just construct the full
    decomposition using the schoolbook method but with a smarter structure.
    
    For now, let's use the schoolbook as a baseline and try to optimize.
    """
    size_u = n * m  # 15
    size_v = m * p  # 30
    size_w = p * n  # 18
    
    # Use schoolbook as starting point for optimization
    U, V, W = _construct_schoolbook(n, m, p)
    
    # Try to reduce rank by zeroing out columns
    # First, check if we can just use a subset of the schoolbook terms
    # by identifying linearly dependent rank-1 tensors
    
    # For a 3x5x6 multiplication, the schoolbook uses 90 terms.
    # Let's see if we can find a lower rank decomposition.
    
    # One approach: use the fact that many of the rank-1 tensors
    # may be linearly dependent in a way that allows cancellation.
    
    # Let me try a different structured approach:
    # Split the computation into blocks and use Strassen on each block.
    
    # For 3x5x6:
    # Think of A as [A_1 | A_2] where A_1 is 3x3, A_2 is 3x2
    # Think of B as [B_1; B_2] where B_1 is 3x6, B_2 is 2x6
    # C = A_1 * B_1 + A_2 * B_2
    
    # For A_1 * B_1 (3x3 times 3x6):
    # Split B_1 = [B_1a | B_1b] where each is 3x3
    # A_1 * B_1 = A_1 * B_1a + A_1 * B_1b
    # Each A_1 * B_1x is 3x3x3, rank 7 with Strassen
    # Total: 14
    
    # For A_2 * B_2 (3x2 times 2x6):
    # Split B_2 = [B_2a | B_2b] where each is 2x3
    # A_2 * B_2 = A_2 * B_2a + A_2 * B_2b
    # Each A_2 * B_2x is 3x2x3
    
    # For 3x2x3: C_{ik} = A_{i1}*B_{1k} + A_{i2}*B_{2k}
    # This is a sum of two rank-1 tensors (for each k, but coupled across i)
    # Actually, for each output element C_{ik} = A_{i1}*B_{1k} + A_{i2}*B_{2k}
    # The tensor is sum over j=1,2 of (e_i * A_{ij}) ⊗ (B_{jk} * e_k)
    # This has rank at most 2*3 = 6 if we consider it as a 3x6 matrix
    # But as a 3x2x3 tensor, the rank is at most 3*2 = 6 (since for each j, 
    # we have a 3x3 matrix which can be decomposed into at most 3 rank-1 terms)
    # Actually, the rank of the 3x2x3 multiplication tensor is 6.
    # Proof: For each j in {1,2} and each i in {1,2,3}, we have term A_{ij}*B_{jk}
    # but these are coupled. The minimal rank is 6.
    
    # So total rank: 14 + 6 + 6 = 26
    
    # Let me construct this!
    
    U_new = []
    V_new = []
    W_new = []
    
    # Helper to add a rank-1 term with given indices
    def add_term(i, j, k, u_val=1.0, v_val=1.0, w_val=1.0):
        """Add a rank-1 term for C[i,k] += A[i,j]*B[j,k]"""
        I = i * m + j
        J = j * p + k
        K = k * n + i
        U_new.append((I, u_val))
        V_new.append((J, v_val))
        W_new.append((K, w_val))
    
    # We'll build the full decomposition using Strassen on the 3x3x3 blocks
    # and schoolbook on the 3x2x3 blocks
    
    # For the 3x3x3 blocks (A_1 * B_1a and A_1 * B_1b):
    # Use Strassen's 7-product algorithm
    
    # Strassen for 2x2x2:
    # P1 = A11 * B11
    # P2 = A12 * B21
    # P3 = A21 * B12
    # P4 = A22 * B22
    # P5 = (A11+A22) * (B11+B22)
    # P6 = (A12+A22) * (B21-B11)
    # P7 = (A21-A11) * (B12+B22)
    
    # For 3x3, we can use a tiling approach or a direct rank-7 decomposition.
    # Let me use a known rank-7 decomposition for 3x3x3.
    
    # Actually, let me just use the schoolbook for now and see if we can
    # verify it, then try to find a better decomposition.
    
    # Return the schoolbook for now
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
    # Start with schoolbook and try to reduce rank
    best_rank = n * m * p  # 90
    best_U, best_V, best_W = _construct_schoolbook(n, m, p)
    
    # Try various ranks from low to high
    for rank in [48, 50, 52, 54, 56, 58, 60, 65, 70, 75, 80, 85, 89, 90]:
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple restarts
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize U, V, W with structured initialization
            # Use schoolbook as starting point and zero out some columns
            if rank == 90:
                U = best_U.copy()
                V = best_V.copy()
                W = best_W.copy()
            else:
                # Initialize with random but scaled
                U = np.random.randn(size_u, rank) * 0.1
                V = np.random.randn(size_v, rank) * 0.1
                W = np.random.randn(size_w, rank) * 0.1
            
            # Use alternating least squares with better convergence
            for step in range(100):
                if time.time() > deadline - 10.0:
                    break
                
                # Compute current X
                X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
                diff = T - X
                
                # Loss
                loss = np.sum(diff**2)
                if loss < 1e-10:
                    break
                
                # Update U: For each r, U[:, r] should minimize ||T - X||^2
                # This is a linear least squares problem
                # Fix V and W, solve for U
                # dU[I, r] = sum_{J,K} diff[I, J, K] * V[J, r] * W[K, r]
                dU = np.einsum('ijk,jr,kr->ir', diff, V, W, optimize=True)
                dV = np.einsum('ijk,ir,kr->jr', diff, U, W, optimize=True)
                dW = np.einsum('ijk,ir,jr->kr', diff, U, V, optimize=True)
                
                # Adaptive learning rate with momentum
                lr = 0.02 / (1 + step * 0.1)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 10 == 0 and loss < 1e-6:
                    break
            
            # Check if we have a good solution
            X_final = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            final_loss = np.sum((T - X_final)**2)
            
            if final_loss < 1e-8:
                # Try to round to integers or half-integers
                for mult in [1, 2, 4]:
                    U_rounded = np.rint(U * mult) / mult
                    V_rounded = np.rint(V * mult) / mult
                    W_rounded = np.rint(W * mult) / mult
                    
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                
                # If rounding doesn't work, use the continuous solution
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
    
    # Fallback to best found or schoolbook
    if best_rank == n * m * p:
        U, V, W = _construct_schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
