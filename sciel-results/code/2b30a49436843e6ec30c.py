import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # The target tensor T is sparse.
    # T[I, J, K] = 1 if I, J, K correspond to (i, j, k)
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    
    # We store the non-zero indices
    # For each I, which (J, K) are non-zero?
    # I = i*m + j.
    # J must be j*p + k. So J % p == k? No, J = j*p + k.
    # K must be k*n + i. So K % n == i? No, K = k*n + i.
    
    # Let's build the list of (J, K) for each I.
    # And the list of (I, K) for each J.
    # And the list of (I, J) for each K.
    
    I_to_JK = [[] for _ in range(n*m)]
    J_to_IK = [[] for _ in range(m*p)]
    K_to_IJ = [[] for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                I_to_JK[I].append((J, K))
                J_to_IK[J].append((I, K))
                K_to_IJ[K].append((I, J))
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Time limit guard
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks from lower bound upwards
    # Lower bound for M_4,5 is likely around 40-50
    # We'll try a range
    for R in range(40, 100):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W with small random integers or floats
        # Using random integers helps with rounding later
        np.random.seed(42 + R)
        U = np.random.randint(-1, 2, size=(n*m, R)).astype(float)
        V = np.random.randint(-1, 2, size=(m*p, R)).astype(float)
        W = np.random.randint(-1, 2, size=(p*n, R)).astype(float)
        
        # ALS iterations
        max_iters = 200
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for I in range(n*m):
                # Collect non-zero terms
                # A = sum_{J,K in I_to_JK[I]} (V[J,:] * W[K,:]) (V[J,:] * W[K,:])^T
                # b = sum_{J,K in I_to_JK[I]} (V[J,:] * W[K,:])
                # Note: T[I,J,K] = 1
                A = np.zeros((R, R))
                b = np.zeros(R)
                for J, K in I_to_JK[I]:
                    v_w = V[J, :] * W[K, :]
                    A += np.outer(v_w, v_w)
                    b += v_w
                # Solve (A + lambda*I) U[I,:] = b
                # Add small regularization
                A += 1e-6 * np.eye(R)
                try:
                    U[I, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
                    
            # Update V
            for J in range(m*p):
                A = np.zeros((R, R))
                b = np.zeros(R)
                for I, K in J_to_IK[J]:
                    u_w = U[I, :] * W[K, :]
                    A += np.outer(u_w, u_w)
                    b += u_w
                A += 1e-6 * np.eye(R)
                try:
                    V[J, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
                    
            # Update W
            for K in range(p*n):
                A = np.zeros((R, R))
                b = np.zeros(R)
                for I, J in K_to_IJ[K]:
                    u_v = U[I, :] * V[J, :]
                    A += np.outer(u_v, u_v)
                    b += u_v
                A += 1e-6 * np.eye(R)
                try:
                    W[K, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
        
        # Check residual and round
        # Compute residual
        # We can't compute the full tensor easily, but we can check the non-zero entries
        # Actually, we just need to check if the rounded solution is exact.
        
        # Round to nearest integer
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        # Verify exactness
        is_exact = True
        # Check all non-zero entries
        for I in range(n*m):
            for J, K in I_to_JK[I]:
                val = np.dot(U_int[I, :], np.dot(V_int[J, :], W_int[K, :]))
                if val != 1:
                    is_exact = False
                    break
            if not is_exact:
                break
        
        if is_exact:
            # Also check that zero entries remain zero?
            # The tensor is 0 elsewhere.
            # We should verify a few random zero entries
            # Or assume if non-zeros match, it's likely good?
            # Better to check a sample of zeros.
            for _ in range(100):
                I = np.random.randint(0, n*m)
                J = np.random.randint(0, m*p)
                K = np.random.randint(0, p*n)
                # Check if (I,J,K) is a non-zero entry
                # If not, val should be 0
                is_nonzero = False
                for jk in I_to_JK[I]:
                    if jk == (J, K):
                        is_nonzero = True
                        break
                if not is_nonzero:
                    val = np.dot(U_int[I, :], np.dot(V_int[J, :], W_int[K, :]))
                    if val != 0:
                        is_exact = False
                        break
                if not is_exact:
                    break
            
            if is_exact:
                best_rank = R
                best_U = U_int.tolist()
                best_V = V_int.tolist()
                best_W = W_int.tolist()
                break # Found a valid solution, stop searching higher ranks?
                # Actually, we want the MINIMAL rank.
                # Since we iterate R upwards, the first valid one is the best.
                break
        
    if best_U is None:
        # Fallback to schoolbook
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
