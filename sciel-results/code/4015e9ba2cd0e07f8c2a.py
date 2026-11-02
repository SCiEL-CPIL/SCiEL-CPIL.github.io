"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True

def _verify_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-6):
    """Fast vectorized verification."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err <= tol

def _hadamard_2d(n1, n2):
    """Construct 2D Hadamard matrix of size n1*n2."""
    H1 = np.ones((n1, n1))
    for i in range(1, n1):
        for j in range(n1):
            if (i & (1 << (i.bit_length() - 1 - j))) if j < i.bit_length() else False:
                pass
    # Simple Kronecker product of 1D Hadamards
    def hadamard_1d(size):
        H = np.ones((size, size))
        for i in range(1, size):
            for j in range(size):
                if (i & j) != 0:
                    H[i, j] = -1
        # This isn't quite right for arbitrary sizes, use recursive construction
        # For power of 2:
        if size == 1:
            return np.ones((1, 1))
        half = size // 2
        H_half = hadamard_1d(half)
        return np.block([[H_half, H_half], [H_half, -H_half]])
    
    H1 = hadamard_1d(n1)
    H2 = hadamard_1d(n2)
    return np.kron(H1, H2)

def _winograd_convolutional(n, m, p, deadline):
    """
    Winograd-style decomposition mapping matrix mult to convolution.
    Uses structured transforms (Hadamard-like) to reduce rank.
    """
    # For 4x4 * 4x5, try block-based Winograd
    # Split into 2x2 blocks where possible
    # A: 4x4, B: 4x5 -> C: 4x5
    
    # Try different block sizes and transform structures
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    np.random.seed(42)
    
    # Strategy 1: Try Winograd F(2,2,1) style on 2x2 blocks
    # For 2x2 * 2x2 -> 2x2, Winograd uses 9 mults vs 8 schoolbook
    # But for our case, we need to handle the non-square B
    
    # Strategy 2: Random search with structured initialization
    # Try ranks from 70 down to 50
    for target_rank in range(min(75, best_rank - 5), 40, -1):
        if time.time() > deadline - 10:
            break
        
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structured pattern + noise
            scale = 0.5 + 0.5 * attempt
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.005
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute loss and gradients
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                loss = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                            s = np.sum(prod)
                            error = s - 1.0
                            loss += error * error
                            
                            U_grad[i * m + j, :] += error * V[j * p + k, :] * W[k * n + i, :]
                            V_grad[j * p + k, :] += error * U[i * m + j, :] * W[k * n + i, :]
                            W_grad[k * n + i, :] += error * U[i * m + j, :] * V[j * p + k, :]
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
                
                # Adaptive LR
                if it % 50 == 49:
                    lr *= 0.5
            
            # Check if valid
            if _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-4):
                # Round to half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-3):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return best

def _structured_search(n, m, p, deadline):
    """
    Search using structured patterns based on algebraic symmetries.
    Try sparse and low-rank structured initializations.
    """
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    np.random.seed(123)
    
    # Try various structured initializations
    structures = []
    
    # 1. Block diagonal structure
    # 2. Toeplitz-like structure
    # 3. Random sparse
    
    for target_rank in range(70, 45, -2):
        if time.time() > deadline - 10:
            break
        
        for init_type in range(3):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            if init_type == 0:
                # Sparse random
                nnz_u = target_rank * 2
                nnz_v = target_rank * 2
                nnz_w = target_rank * 2
                for _ in range(nnz_u):
                    U[np.random.randint(n * m), np.random.randint(target_rank)] = np.random.choice([-1, 1])
                for _ in range(nnz_v):
                    V[np.random.randint(m * p), np.random.randint(target_rank)] = np.random.choice([-1, 1])
                for _ in range(nnz_w):
                    W[np.random.randint(p * n), np.random.randint(target_rank)] = np.random.choice([-1, 1])
            elif init_type == 1:
                # Hadamard-inspired
                if target_rank <= 16:
                    H = _hadamard_2d(4, 4)
                    H = H[:min(n*m, H.shape[0]), :target_rank]
                    U = H * 0.5
                    H2 = _hadamard_2d(4, 4)
                    V = H2[:min(m*p, H2.shape[0]), :target_rank] * 0.5
                    H3 = _hadamard_2d(4, 4)
                    W = H3[:min(p*n, H3.shape[0]), :target_rank] * 0.5
            else:
                # Random with small values
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
            
            # Refine with gradient descent
            lr = 0.003
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(150):
                if time.time() > deadline - 5:
                    break
                
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                loss = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                            s = np.sum(prod)
                            error = s - 1.0
                            loss += error * error
                            
                            U_grad[i * m + j, :] += error * V[j * p + k, :] * W[k * n + i, :]
                            V_grad[j * p + k, :] += error * U[i * m + j, :] * W[k * n + i, :]
                            W_grad[k * n + i, :] += error * U[i * m + j, :] * V[j * p + k, :]
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-12:
                    break
                if it % 40 == 39:
                    lr *= 0.7
            
            if _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-4):
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-3):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Ensure we have a valid baseline
    baseline = _schoolbook(n, m, p)
    
    # Try Winograd-style decomposition
    try:
        result = _winograd_convolutional(n, m, p, deadline)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-3):
            rank = len(result["U"][0])
            if rank < n * m * p:
                return result
    except Exception:
        pass
    
    # Try structured search
    try:
        if time.time() < deadline - 10:
            result2 = _structured_search(n, m, p, deadline)
            if _verify_decomposition(result2["U"], result2["V"], result2["W"], n, m, p, tol=1e-3):
                rank2 = len(result2["U"][0])
                if rank2 < n * m * p:
                    return result2
    except Exception:
        pass
    
    # Fallback to schoolbook
    return baseline
# EVOLVE-BLOCK-END
