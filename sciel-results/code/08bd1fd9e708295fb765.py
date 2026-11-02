# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
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

    def verify_fast(U, V, W):
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        try:
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False
        if not np.allclose(result, 1.0, atol=1e-9):
            return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    rng = np.random.RandomState(42)
    choices = [0, 1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5]

    # Focus on ranks just below schoolbook
    # 83, 82, 81, 80
    for target_rank in range(best_rank - 1, best_rank - 5, -1):
        if time.time() > deadline:
            break
        attempts = 1000
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            U_flat = rng.choice(choices, size=(n * m, target_rank))
            V_flat = rng.choice(choices, size=(m * p, target_rank))
            W_flat = rng.choice(choices, size=(p * n, target_rank))
            
            U_list = U_flat.tolist()
            V_list = V_flat.tolist()
            W_list = W_flat.tolist()
            
            if verify_fast(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        if best_rank == target_rank:
            # Found a lower rank, can break or continue to lower
            # Since lower is better, we continue to try lower ranks
            # But we already broke out of the inner loop
            pass

    return best_result
# EVOLVE-BLOCK-END
