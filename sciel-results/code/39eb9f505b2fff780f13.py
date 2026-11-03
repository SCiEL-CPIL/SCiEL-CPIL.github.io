import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_rank10_2x3x2():
    """
    Try to find a rank-10 decomposition for 2x3x2 matrix multiplication.
    Known constructions exist for rank 10 for 2x3x2.
    """
    n, m, p = 2, 3, 2
    
    # Target tensor for 2x3x2
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try rank 10 with structured initializations
    R = 10
    for seed in range(100):
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, R) * 0.3
        V = np.random.randn(m * p, R) * 0.3
        W = np.random.randn(p * n, R) * 0.3
        
        # Use L-BFGS-like approach with multiple restarts
        lr = 0.02
        for it in range(2000):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Gradient descent with adaptive step
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-15:
                break
            
            step = lr / (1 + grad_norm * 0.1)
            U -= step * dU
            V -= step * dV
            W -= step * dW
        
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_dec, T, atol=1e-10):
            # Build full 2x3x6 decomposition using block structure
            n_full, m_full, p_full = 2, 3, 6
            R_full = 3 * R
            
            U_full = np.zeros((n_full * m_full, R_full))
            V_full = np.zeros((m_full * p_full, R_full))
            W_full = np.zeros((p_full * n_full, R_full))
            
            for block in range(3):
                offset = block * R
                U_full[:, offset:offset+R] = U
                
                for j in range(m_full):
                    for k in range(2):
                        V_full[j * p_full + (block * 2 + k), offset:offset+R] = V[j * 2 + k]
                
                for k in range(2):
                    for i in range(n_full):
                        W_full[(block * 2 + k) * n_full + i, offset:offset+R] = W[k * n_full + i]
            
            if _verify_decomposition(U_full, V_full, W_full, n_full, m_full, p_full):
                return U_full, V_full, W_full, R_full
    
    return None

def _try_structured_rank24():
    """
    Try a structured rank-24 decomposition for 2x3x6.
    Based on the idea that each output C[i,k] depends on sum over j of A[i,j]*B[j,k].
    For 2x3x6, we have 2*6=12 outputs, each requiring 3 multiplications = 36 schoolbook.
    
    Try to share computation across outputs using the structure.
    """
    n, m, p = 2, 3, 6
    R = 24
    
    # Try various structured patterns
    for pattern_seed in range(50):
        np.random.seed(pattern_seed)
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Initialize with random sparse patterns
        for r in range(R):
            # Each rank-1 term picks one (i,j) for U, one (j',k) for V, one (k',i') for W
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            
            U[i * m + j, r] = np.random.choice([-1, 1])
            V[j * p + k, r] = np.random.choice([-1, 1])
            W[k * n + i, r] = np.random.choice([-1, 1])
        
        # Optimize
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        lr = 0.01
        for it in range(1000):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-12:
                break
            
            dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_dec, T, atol=1e-10):
            return U, V, W, R
    
    return None

def _try_known_construction():
    """
    Try a known rank-24 construction for 2x3x6.
    
    One approach: use the fact that 2x3x6 can be seen as three independent 2x3x2 problems.
    If rank(2x3x2) = 8, then rank(2x3x6) <= 24.
    
    Let's try to explicitly construct a rank-8 decomposition for 2x3x2.
    """
    n, m, p = 2, 3, 2
    R = 8
    
    # Target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try systematic search for rank-8 with small integer entries
    # Each entry can be in {-1, 0, 1}
    # U is 6x8, V is 6x8, W is 4x8
    
    # This is a large search space, so use random restarts with projection
    for seed in range(200):
        np.random.seed(seed)
        
        # Random initialization with -1, 0, 1
        U = np.random.choice([-1, 0, 1], size=(n*m, R)).astype(float)
        V = np.random.choice([-1, 0, 1], size=(m*p, R)).astype(float)
        W = np.random.choice([-1, 0, 1], size=(p*n, R)).astype(float)
        
        # Local optimization
        lr = 0.05
        for it in range(500):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-12:
                break
            
            dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Round to nearest 0.5 to maintain half-integer property
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
        
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_dec, T, atol=1e-10):
            # Build full 2x3x6
            n_full, m_full, p_full = 2, 3, 6
            R_full = 3 * R
            
            U_full = np.zeros((n_full * m_full, R_full))
            V_full = np.zeros((m_full * p_full, R_full))
            W_full = np.zeros((p_full * n_full, R_full))
            
            for block in range(3):
                offset = block * R
                U_full[:, offset:offset+R] = U
                for j in range(m_full):
                    for k in range(2):
                        V_full[j * p_full + (block * 2 + k), offset:offset+R] = V[j * 2 + k]
                for k in range(2):
                    for i in range(n_full):
                        W_full[(block * 2 + k) * n_full + i, offset:offset+R] = W[k * n_full + i]
            
            if _verify_decomposition(U_full, V_full, W_full, n_full, m_full, p_full):
                return U_full, V_full, W_full, R_full
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Try known construction (rank 24 via rank-8 for 2x3x2)
    if time.time() < deadline - 60:
        result = _try_known_construction()
        if result is not None:
            U, V, W, R = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_solution = (U, V, W, R)
                best_rank = R
    
    # Try rank-10 for 2x3x2 (gives rank 30 for 2x3x6)
    if best_solution is None or best_rank > 30:
        if time.time() < deadline - 40:
            result = _try_rank10_2x3x2()
            if result is not None:
                U, V, W, R = result
                if _verify_decomposition(U, V, W, n, m, p):
                    if best_solution is None or R < best_rank:
                        best_solution = (U, V, W, R)
                        best_rank = R
    
    # Try structured rank-24
    if best_solution is None or best_rank > 24:
        if time.time() < deadline - 20:
            result = _try_structured_rank24()
            if result is not None:
                U, V, W, R = result
                if _verify_decomposition(U, V, W, n, m, p):
                    if best_solution is None or R < best_rank:
                        best_solution = (U, V, W, R)
                        best_rank = R
    
    # Fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
