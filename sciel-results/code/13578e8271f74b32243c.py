"""Rewritten program for matrix multiplication tensor decomposition (4x5 * 5x5)."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_result = None
    
    # Generate schoolbook decomposition as initial best
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    
    # Try to find lower-rank decompositions using iterative optimization
    # We'll try ranks from a lower bound up to schoolbook_rank - 1
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(20, 25, 20) = 25
    # But for matrix multiplication, the lower bound is at least max(n*p, m*p, n*m)
    # Actually for multiplying A(nxm) * B(mxp), the tensor has n*m*p entries
    # Lower bound is at least max(n*p, m*p, n*m) but can be higher
    
    # Known results: For 4x5 * 5x5, the minimum rank is not well-known
    # Let's try various ranks and see if we can find a valid decomposition
    
    def validate_decomposition(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # For each (i, j, k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other combinations, it should be 0
        # Actually, the tensor T[i,j,k] for matrix multiplication is:
        # T[i,j,k] = 1 if the bilinear form gives the (i,k) entry of C = A*B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) in the product, but structured as:
        # The (i,k) entry of C depends on sum over j of A[i,j]*B[j,k]
        # In tensor form: C[i,k] = sum_{j,r} T[i,j,k,r] * A[i,j] * B[j,k]
        # where T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need: for each (i,k), sum_j T[i,j,k,r] summed over r equals delta structure
        
        # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
        # We want: sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j V[j*p+k,r] B[j,k]) W[k*n+i,r] = C[i,k]
        # This must equal sum_j A[i,j] B[j,k] for all A, B
        
        # The condition is: for each (i,k), the bilinear form in A and B given by
        # sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k])
        # must equal sum_j A[i,j] B[j,k]
        
        # This means: sum_r W[k*n+i,r] * U[i*m+j1,r] * V[j2*p+k,r] = delta_{j1,j2} for all i,k,j1,j2
        # Wait, let me re-derive. 
        # C[i,k] = sum_r W[k*n+i,r] * (sum_{j} U[i*m+j,r] * A[i,j]) * (sum_{j'} V[j'*p+k,r] * B[j',k])
        # = sum_r sum_j sum_{j'} W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] * A[i,j] * B[j',k]
        # We need this to equal sum_j A[i,j] * B[j,k]
        # So: sum_r W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] = delta_{j,j'} for all i,k,j,j'
        
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each i, k, j, j': sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] == delta_{j,j'}
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        # Compute sum_r W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r]
                        val = np.sum(W_np[k * n + i, :] * U_np[i * m + j, :] * V_np[jp * p + k, :])
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    # Try to find decompositions of decreasing rank
    # Start from a reasonable lower bound
    # For 4x5*5x5, let's try ranks from 25 upward (since schoolbook is 100)
    
    # Use random initialization and gradient-based search
    def try_find_decomposition(target_rank, max_iters=5000, seed=42):
        """Try to find a valid decomposition of the given rank."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Define the loss function
        def compute_loss(U, V, W):
            """Compute the loss: sum of squared errors in the tensor decomposition."""
            loss = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            expected = 1.0 if j == jp else 0.0
                            val = np.sum(W[k * n + i, :] * U[i * m + j, :] * V[jp * p + k, :])
                            loss += (val - expected) ** 2
            return loss
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        best_loss = compute_loss(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            expected = 1.0 if j == jp else 0.0
                            # val = sum_r W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r]
                            # d(val)/dU[i*m+j, r] = W[k*n+i, r] * V[jp*p+k, r]
                            # d(val)/dV[jp*p+k, r] = W[k*n+i, r] * U[i*m+j, r]
                            # d(val)/dW[k*n+i, r] = U[i*m+j, r] * V[jp*p+k, r]
                            err = np.sum(W[k * n + i, :] * U[i * m + j, :] * V[jp * p + k, :]) - expected
                            dU[i * m + j, :] += 2 * err * W[k * n + i, :] * V[jp * p + k, :]
                            dV[jp * p + k, :] += 2 * err * W[k * n + i, :] * U[i * m + j, :]
                            dW[k * n + i, :] += 2 * err * U[i * m + j, :] * V[jp * p + k, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Compute new loss
            new_loss = compute_loss(U, V, W)
            
            if new_loss < best_loss:
                best_loss = new_loss
                # Check if we've found a valid solution
                if best_loss < 1e-12:
                    # Convert to integer/half-integer if possible
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    # Validate rounded
                    if validate_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), target_rank):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    # Try without rounding
                    if validate_decomposition(U.tolist(), V.tolist(), W.tolist(), target_rank):
                        return U.tolist(), V.tolist(), W.tolist()
            else:
                # Reduce learning rate
                lr *= 0.95
            
            # Early stopping if loss isn't improving
            if it > 100 and new_loss > best_loss * 1.001:
                break
        
        return None
    
    # Try various ranks
    # Start from a lower estimate and go up
    # Known lower bound for 4x5*5x5 is at least 25 (from dimension arguments)
    # Let's try from 25 to schoolbook_rank - 1
    
    for target_rank in range(25, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        # Try a few random seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 15:
                break
            
            result = try_find_decomposition(target_rank, max_iters=3000, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                # Final validation
                if validate_decomposition(U_list, V_list, W_list, target_rank):
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if best_rank < schoolbook_rank:
            # Found something better, try to go even lower
            continue
    
    return best_result
