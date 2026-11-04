import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target tensor T[a, b, c]
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i for some i,j,k
    # Else 0
    nm = n * m
    mp = m * p
    pn = p * n
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = get_schoolbook()
    best_rank = n * m * p

    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Use einsum for speed
        return np.einsum('ar,br,cr->abc', U, V, W)

    def check_validity(U, V, W, tol=1e-9):
        T = compute_tensor(U, V, W)
        return np.allclose(T, target, atol=tol)

    def snap_to_half(mat):
        return np.round(mat * 2) / 2

    def search_rank(R, num_restarts=5, steps=2000):
        nonlocal best_solution, best_rank
        if R >= best_rank:
            return
        
        for r in range(num_restarts):
            if time.time() > deadline:
                return
            
            # Initialize
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            # Gradient Descent
            lr = 0.01
            for step in range(steps):
                if time.time() > deadline:
                    return
                
                T = compute_tensor(U, V, W)
                err = T - target
                loss = np.sum(err**2)
                
                if loss < 1e-12:
                    break
                
                # Gradients
                # dL/dU[a, r] = 2 * sum_{b,c} err[a, b, c] * V[b, r] * W[c, r]
                # dL/dV[b, r] = 2 * sum_{a,c} err[a, b, c] * U[a, r] * W[c, r]
                # dL/dW[c, r] = 2 * sum_{a,b} err[a, b, c] * U[a, r] * V[b, r]
                
                # Efficient computation using einsum
                # grad_U[a, r] = 2 * einsum('abc,br,cr->ar', err, V, W)
                # grad_V[b, r] = 2 * einsum('abc,ar,cr->br', err, U, W)
                # grad_W[c, r] = 2 * einsum('abc,ar,br->cr', err, U, V)
                
                grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
                grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
                grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.999

            # Check if converged
            T = compute_tensor(U, V, W)
            if np.allclose(T, target, atol=1e-9):
                # Try snapping
                U_snap = snap_to_half(U)
                V_snap = snap_to_half(V)
                W_snap = snap_to_half(W)
                if check_validity(U_snap, V_snap, W_snap):
                    best_solution = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    best_rank = R
                    return

    # Search for lower ranks
    # Start from best_rank - 1
    # We have ~110s.
    # Each search_rank call takes some time.
    # Let's try ranks 44 down to 35.
    for R in range(best_rank - 1, 34, -1):
        if time.time() > deadline:
            break
        search_rank(R, num_restarts=3, steps=1000)
        if best_rank == R:
            # Found a solution for this rank, try to go lower
            continue

    return best_solution
