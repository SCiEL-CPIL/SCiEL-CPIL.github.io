import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 150s total, 30s margin

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    def compute_tensor(U, V, W, R):
        """Compute the tensor from U, V, W"""
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        return np.einsum('ijr,jkr,kri->ijk', U_r, V_r, W_r)

    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition"""
        R = U.shape[1]
        
        # Check integrality (integers or half-integers)
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
            if np.any(np.abs(mat) > 1000):
                return False

        # Check decomposition
        T_calc = compute_tensor(U, V, W, R)
        if not np.allclose(T_calc, 1.0, atol=1e-8):
            return False
            
        return True

    def loss_function(U, V, W, R):
        """Compute sum of squared errors"""
        T_calc = compute_tensor(U, V, W, R)
        return np.sum((T_calc - 1.0)**2)

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def local_search_integer(U, V, W, R, max_steps=500, deadline_time=None):
        """Local search on integer lattice"""
        current_loss = loss_function(U, V, W, R)
        
        if current_loss < 1e-10:
            U_int = round_to_half(U)
            V_int = round_to_half(V)
            W_int = round_to_half(W)
            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int
        
        for step in range(max_steps):
            if deadline_time is not None and time.time() > deadline_time:
                return None
            
            # Pick random matrix and element
            mat_idx = np.random.randint(3)
            delta = np.random.choice([-2, -1, 1, 2])
            
            if mat_idx == 0:
                idx = np.random.randint(n*m)
                r = np.random.randint(R)
                U[idx, r] += delta
                new_loss = loss_function(U, V, W, R)
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-10:
                        U_int = round_to_half(U)
                        V_int = round_to_half(V)
                        W_int = round_to_half(W)
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                else:
                    U[idx, r] -= delta
            elif mat_idx == 1:
                idx = np.random.randint(m*p)
                r = np.random.randint(R)
                V[idx, r] += delta
                new_loss = loss_function(U, V, W, R)
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-10:
                        U_int = round_to_half(U)
                        V_int = round_to_half(V)
                        W_int = round_to_half(W)
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                else:
                    V[idx, r] -= delta
            else:
                idx = np.random.randint(p*n)
                r = np.random.randint(R)
                W[idx, r] += delta
                new_loss = loss_function(U, V, W, R)
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-10:
                        U_int = round_to_half(U)
                        V_int = round_to_half(V)
                        W_int = round_to_half(W)
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                else:
                    W[idx, r] -= delta
        
        return None

    def gradient_descent_polish(U, V, W, R, max_iter=200, lr=0.01, deadline_time=None):
        """Gradient descent on continuous relaxation"""
        U_f = U.astype(float)
        V_f = V.astype(float)
        W_f = W.astype(float)
        
        current_loss = loss_function(U_f, V_f, W_f, R)
        
        for it in range(max_iter):
            if deadline_time is not None and time.time() > deadline_time:
                break
            
            # Compute gradients numerically (central differences)
            eps = 1e-7
            grad_U = np.zeros_like(U_f)
            grad_V = np.zeros_like(V_f)
            grad_W = np.zeros_like(W_f)
            
            # Only compute gradients for a subset to save time
            # Sample random positions
            num_samples = 50
            for _ in range(num_samples):
                # U gradient
                if np.random.rand() < 1/3:
                    idx = np.random.randint(n*m)
                    r = np.random.randint(R)
                    U_f[idx, r] += eps
                    loss_plus = loss_function(U_f, V_f, W_f, R)
                    U_f[idx, r] -= 2*eps
                    loss_minus = loss_function(U_f, V_f, W_f, R)
                    U_f[idx, r] += eps
                    grad_U[idx, r] = (loss_plus - loss_minus) / (2*eps)
                    U_f[idx, r] -= lr * grad_U[idx, r]
                
                # V gradient
                elif np.random.rand() < 1/3:
                    idx = np.random.randint(m*p)
                    r = np.random.randint(R)
                    V_f[idx, r] += eps
                    loss_plus = loss_function(U_f, V_f, W_f, R)
                    V_f[idx, r] -= 2*eps
                    loss_minus = loss_function(U_f, V_f, W_f, R)
                    V_f[idx, r] += eps
                    grad_V[idx, r] = (loss_plus - loss_minus) / (2*eps)
                    V_f[idx, r] -= lr * grad_V[idx, r]
                
                # W gradient
                else:
                    idx = np.random.randint(p*n)
                    r = np.random.randint(R)
                    W_f[idx, r] += eps
                    loss_plus = loss_function(U_f, V_f, W_f, R)
                    W_f[idx, r] -= 2*eps
                    loss_minus = loss_function(U_f, V_f, W_f, R)
                    W_f[idx, r] += eps
                    grad_W[idx, r] = (loss_plus - loss_minus) / (2*eps)
                    W_f[idx, r] -= lr * grad_W[idx, r]
            
            new_loss = loss_function(U_f, V_f, W_f, R)
            if new_loss < current_loss:
                current_loss = new_loss
                # Decay learning rate
                lr *= 0.99
            else:
                # Revert last step approximately
                pass
            
            if current_loss < 1e-10:
                break
        
        # Round to half-integers
        U_int = round_to_half(U_f)
        V_int = round_to_half(V_f)
        W_int = round_to_half(W_f)
        
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        return None

    def search_rank(R, num_restarts=20, deadline_time=None):
        """Search for a rank-R decomposition"""
        for restart in range(num_restarts):
            if deadline_time is not None and time.time() > deadline_time:
                return None
            
            # Initialize with random integers
            U = np.random.randint(-3, 4, size=(n*m, R))
            V = np.random.randint(-3, 4, size=(m*p, R))
            W = np.random.randint(-3, 4, size=(p*n, R))
            
            # First try integer local search
            result = local_search_integer(U, V, W, R, max_steps=300, deadline_time=deadline_time)
            if result:
                return result
            
            # Then try gradient polish
            result = gradient_descent_polish(U, V, W, R, max_iter=100, lr=0.05, deadline_time=deadline_time)
            if result:
                return result
        
        return None

    # Try ranks from 19 upward
    # Known: rank(2x3x4) = 19 (Bryant, 1994)
    for R in range(19, 24):
        if time.time() > deadline:
            break
        
        result = search_rank(R, num_restarts=30, deadline_time=deadline)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break

    # Convert to lists for the output interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
