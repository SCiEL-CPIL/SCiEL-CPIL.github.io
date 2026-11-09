"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40

    def schoolbook():
        rank = schoolbook_rank
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

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def verify_numpy(U, V, W):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def round_to_half(vals):
        return [round(v * 2) / 2 for v in vals]

    def compute_tensor(U, V, W):
        """Compute the current tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]"""
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        return T

    def compute_gradients(U, V, W, T):
        """Compute gradients of L = sum (T-1)^2 with respect to U, V, W"""
        rank = U.shape[1]
        E = T - 1.0  # Error tensor
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    # Gradient w.r.t U[i,j,c] = 2 * sum_k E[i,j,k] * V[j,k,c] * W[k,i,c]
                    s = 0.0
                    for k in range(p):
                        s += E[i, j, k] * V[j*p+k, c] * W[k*n+i, c]
                    U_grad[i*m+j, c] = 2.0 * s
        
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    # Gradient w.r.t V[j,k,c] = 2 * sum_i E[i,j,k] * U[i,j,c] * W[k,i,c]
                    s = 0.0
                    for i in range(n):
                        s += E[i, j, k] * U[i*m+j, c] * W[k*n+i, c]
                    V_grad[j*p+k, c] = 2.0 * s
        
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    # Gradient w.r.t W[k,i,c] = 2 * sum_j E[i,j,k] * U[i,j,c] * V[j,k,c]
                    s = 0.0
                    for j in range(m):
                        s += E[i, j, k] * U[i*m+j, c] * V[j*p+k, c]
                    W_grad[k*n+i, c] = 2.0 * s
        
        return U_grad, V_grad, W_grad

    def adam_step(params, grads, opt_states, beta1, beta2, eps, lr):
        """Perform an Adam update step"""
        for idx, (param, grad) in enumerate(zip(params, grads)):
            t = opt_states[idx]['t'] + 1
            opt_states[idx]['t'] = t
            opt_states[idx]['m'] = beta1 * opt_states[idx]['m'] + (1 - beta1) * grad
            opt_states[idx]['v'] = beta2 * opt_states[idx]['v'] + (1 - beta2) * grad**2
            m_hat = opt_states[idx]['m'] / (1 - beta1**t)
            v_hat = opt_states[idx]['v'] / (1 - beta2**t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def try_rank(rank, time_budget):
        """Try to find a rank decomposition with given rank using Adam optimizer."""
        start = time.time()
        
        # Try multiple random initializations with different strategies
        for attempt in range(50):
            if time.time() > deadline or time.time() - start > time_budget:
                break
            
            # Different initialization strategies
            if attempt < 20:
                # Small random initialization
                scale = 0.1
                U_cur = np.random.randn(n*m, rank) * scale
                V_cur = np.random.randn(m*p, rank) * scale
                W_cur = np.random.randn(p*n, rank) * scale
            elif attempt < 35:
                # Moderate random initialization
                scale = 0.5
                U_cur = np.random.randn(n*m, rank) * scale
                V_cur = np.random.randn(m*p, rank) * scale
                W_cur = np.random.randn(p*n, rank) * scale
            else:
                # Larger random initialization with some structure
                U_cur = np.random.randn(n*m, rank) * 1.0
                V_cur = np.random.randn(m*p, rank) * 1.0
                W_cur = np.random.randn(p*n, rank) * 1.0
                # Add some bias to help convergence
                U_cur += 0.5
                V_cur += 0.5
                W_cur += 0.5
            
            # Adam optimizer states
            opt_states = []
            for _ in range(3):
                opt_states.append({
                    'm': np.zeros_like(U_cur),
                    'v': np.zeros_like(U_cur),
                    't': 0
                })
            
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            lr = 0.01
            best_loss = float('inf')
            best_params = None
            no_improve_count = 0
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute tensor and gradients
                T = compute_tensor(U_cur, V_cur, W_cur)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count > 200:
                    break
                
                if loss < 1e-12:
                    break
                
                U_grad, V_grad, W_grad = compute_gradients(U_cur, V_cur, W_cur, T)
                
                # Perform Adam step
                params = [U_cur, V_cur, W_cur]
                grads = [U_grad, V_grad, W_grad]
                adam_step(params, grads, opt_states, beta1, beta2, eps, lr)
                
                # Decay learning rate
                lr *= 0.9995
            
            # Use best parameters found
            if best_params is not None:
                U_cur, V_cur, W_cur = best_params
            
            # Round to half-integers and verify
            U_rounded = [[round(U_cur[r,c]*2)/2 for c in range(rank)] for r in range(n*m)]
            V_rounded = [[round(V_cur[r,c]*2)/2 for c in range(rank)] for r in range(m*p)]
            W_rounded = [[round(W_cur[r,c]*2)/2 for c in range(rank)] for r in range(p*n)]
            
            if verify(U_rounded, V_rounded, W_rounded):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            
            # Try rounding to integers as well
            U_int = [[round(U_cur[r,c]) for c in range(rank)] for r in range(n*m)]
            V_int = [[round(V_cur[r,c]) for c in range(rank)] for r in range(m*p)]
            W_int = [[round(W_cur[r,c]) for c in range(rank)] for r in range(p*n)]
            
            if verify(U_int, V_int, W_int):
                return {"U": U_int, "V": V_int, "W": W_int}
            
            # Re-seed for next attempt
            np.random.seed(42 + attempt * 7919)
        
        return None

    # Try to find lower rank decompositions
    # Known: M_{2,4,5} rank is between 34 and 40
    # Try from 34 upward
    for rank in range(34, 40):
        if time.time() > deadline:
            break
        time_budget = min(20, deadline - time.time())
        if time_budget < 5:
            break
        result = try_rank(rank, time_budget)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
