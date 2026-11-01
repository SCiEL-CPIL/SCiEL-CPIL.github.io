# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        
        # Check valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check invalid triples
        for alpha in range(n*m):
            i_a = alpha // m
            j_a = alpha % m
            for beta in range(m*p):
                j_b = beta // p
                k_b = beta % p
                for gamma in range(p*n):
                    k_c = gamma // n
                    i_c = gamma % n
                    
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    
                    val = np.sum(U[alpha] * V[beta] * W[gamma])
                    if abs(val) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    # Schoolbook solution
    rank = rank_schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    np.random.seed(42)
    
    # Precompute T_flat and indices for M construction speedup?
    # M construction is the bottleneck.
    # M[idx, gamma*r + c] = U_rand[alpha, c] * V_rand[beta, c]
    
    for r in range(23, 11, -1):
        if time.time() > deadline - 10:
            break
            
        attempts = 200
        found = False
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            U_rand = np.random.randint(-1, 2, size=(n*m, r))
            V_rand = np.random.randint(-1, 2, size=(m*p, r))
            
            M = np.zeros((n*m * m*p * p*n, p*n * r))
            T_flat = np.zeros(n*m * m*p * p*n)
            
            idx = 0
            for alpha in range(n*m):
                for beta in range(m*p):
                    for gamma in range(p*n):
                        i_a = alpha // m
                        j_a = alpha % m
                        j_b = beta // p
                        k_b = beta % p
                        k_c = gamma // n
                        i_c = gamma % n
                        
                        if i_a == i_c and j_a == j_b and k_b == k_c:
                            T_flat[idx] = 1.0
                        else:
                            T_flat[idx] = 0.0
                        
                        for c in range(r):
                            M[idx, gamma*r + c] = U_rand[alpha, c] * V_rand[beta, c]
                        idx += 1
            
            W_flat, residuals, rank_m, sv = np.linalg.lstsq(M, T_flat, rcond=None)
            W_sol = W_flat.reshape(p*n, r)
            W_snapped = snap_to_half_integers(W_sol)
            
            if verify(U_rand, V_rand, W_snapped):
                best_rank = r
                best_U, best_V, best_W = U_rand, V_rand, W_snapped
                found = True
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
