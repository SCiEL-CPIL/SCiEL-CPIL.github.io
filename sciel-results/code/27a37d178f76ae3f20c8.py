import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array or flattened
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
                
    # Function to check if a decomposition is exact (with tolerance)
    def check_exact(U, V, W, tol=1e-9):
        # Compute sum U[:, s] V[:, s] W[:, s]
        # This is a tensor contraction.
        # We can do this by iterating over s or using einsum.
        # T_approx[a, b, c] = sum_s U[a, s] V[b, s] W[c, s]
        # Using einsum: 'as,bs,cs->abc'
        T_approx = np.einsum('as,bs,cs->abc', U, V, W, optimize=True)
        # Check difference
        diff = np.abs(T_approx - T)
        return np.max(diff) < tol

    # Function to try to find a decomposition of rank r
    def find_decomposition(r, max_iter=500, lr=0.01):
        # Initialize with random or schoolbook subset
        # Use schoolbook first r columns as init?
        # Or random?
        # Let's try random initialization with small values
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Simple gradient descent on loss = || T_approx - T ||^2
        # Gradient of loss w.r.t U[a, s] is 2 * sum_{b,c} (T_approx[a,b,c] - T[a,b,c]) V[b,s] W[c,s]
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            T_approx = np.einsum('as,bs,cs->abc', U, V, W, optimize=True)
            loss = np.sum((T_approx - T)**2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_loss < 1e-12:
                break
                
            # Compute gradients
            # grad_U[a, s] = 2 * sum_{b,c} (T_approx[a,b,c] - T[a,b,c]) V[b,s] W[c,s]
            # This is equivalent to:
            # diff = T_approx - T
            # grad_U = 2 * einsum('abc,bs,cs->as', diff, V, W)
            diff = T_approx - T
            grad_U = 2.0 * np.einsum('abc,bs,cs->as', diff, V, W, optimize=True)
            grad_V = 2.0 * np.einsum('abc,as,cs->bs', diff, U, W, optimize=True)
            grad_W = 2.0 * np.einsum('abc,as,bs->cs', diff, U, V, optimize=True)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
        return best_U, best_V, best_W, best_loss

    # Try to find a decomposition with rank < 60
    # Start from 59 down to 55
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    for r in range(rank_sb - 1, max(40, rank_sb - 10), -1):
        if time.time() > deadline:
            break
            
        U, V, W, loss = find_decomposition(r, max_iter=200, lr=0.05)
        
        # Check if the solution is close to exact
        if loss < 1e-6:
            # Try to round to integers/half-integers
            # Scale? No, entries should be small integers.
            # Just round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if check_exact(U_r, V_r, W_r, tol=1e-9):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = r
                break # Found a valid one, stop searching lower? 
                # Or continue to find minimal? 
                # The task asks for rank-minimal. 
                # But finding the absolute minimum is hard. 
                # If we found 59, we can try 58. 
                # Let's continue to see if we can go lower.
                
    return best_result
