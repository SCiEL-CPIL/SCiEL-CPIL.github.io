def _try_random_search_fast(n, m, p, target_rank, timeout=15):
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    values = np.array([-2, -1, 0, 1, 2]) / 2.0
    
    for attempt in range(1000): # Increased attempts due to fast verify
        if time.time() - start_time > timeout:
            break
        
        U = np.random.choice(values, size=(nm, target_rank))
        V = np.random.choice(values, size=(mp, target_rank))
        W = np.random.choice(values, size=(pn, target_rank))
        
        if _verify_fast(n, m, p, U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    return None
