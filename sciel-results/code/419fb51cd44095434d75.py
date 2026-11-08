import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 3, 4
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    # Try to find a decomposition of rank 21
    # If it fails, fallback to rank 22
    
    def try_rank(R):
        if time.time() > deadline - 10:
            return None
            
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # ALS iterations
        for _ in range(100):
            if time.time() > deadline - 10:
                return None
                
            # Update U
            # U[:, r] = (T @ V[r] @ W[r]) / (V[r] @ V[r] * W[r] @ W[r])? No.
            # T is 3D. We need to contract T with V and W.
            # U[alpha, r] = sum_beta gamma T[alpha, beta, gamma] V[beta, r] W[gamma, r] / (sum_beta V[beta, r]^2 sum_gamma W[gamma, r]^2)
            
            # Efficiently:
            # For each r, compute numerator and denominator.
            for r in range(R):
                # Numerator: T contracted with V[:, r] and W[:, r]
                # This is equivalent to (T.reshape(n*m, -1) @ (V[:, r] * W[:, r])) ? No.
                # T[alpha, beta, gamma] * V[beta, r] * W[gamma, r]
                # This is sum_beta,gamma T[alpha, beta, gamma] V[beta, r] W[gamma, r]
                
                # We can do this by reshaping T and using matrix multiplication.
                # But R is small (21), so loop is fine.
                num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
                den = np.dot(V[:, r], V[:, r]) * np.dot(W[:, r], W[:, r])
                if den > 1e-10:
                    U[:, r] = num / den
                else:
                    U[:, r] = 0
                    
            # Update V
            for r in range(R):
                num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
                den = np.dot(U[:, r], U[:, r]) * np.dot(W[:, r], W[:, r])
                if den > 1e-10:
                    V[:, r] = num / den
                else:
                    V[:, r] = 0
                    
            # Update W
            for r in range(R):
                num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
                den = np.dot(U[:, r], U[:, r]) * np.dot(V[:, r], V[:, r])
                if den > 1e-10:
                    W[:, r] = num / den
                else:
                    W[:, r] = 0
                    
        # Check error
        # Reconstruct T_approx
        # T_approx[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if err < 1e-6:
            # Try to round to half-integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            T_approx_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
            err_int = np.linalg.norm(T - T_approx_int)
            
            if err_int < 1e-9:
                return U_int, V_int, W_int
            else:
                # Maybe try rounding to integers?
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                T_approx_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                err_int = np.linalg.norm(T - T_approx_int)
                if err_int < 1e-9:
                    return U_int, V_int, W_int
        return None

    # Try Rank 21
    result = try_rank(21)
    if result:
        U, V, W = result
        # Convert to lists of lists for return
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Rank 22 (Current Program Logic)
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    for mat in [U, V, W]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = int(round(val))

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
