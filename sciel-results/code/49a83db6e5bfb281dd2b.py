def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    # Schoolbook rank
    rank = n * m * p
    u = np.zeros((n * m, rank))
    v = np.zeros((m * p, rank))
    w = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u[i * m + j, col] = 1.0
                v[j * p + k, col] = 1.0
                w[k * n + i, col] = 1.0
                col += 1
                if time.time() - start_time > deadline:
                    break
            if time.time() - start_time > deadline:
                break
        if time.time() - start_time > deadline:
            break
            
    return {"U": u.tolist(), "V": v.tolist(), "W": w.tolist()}
