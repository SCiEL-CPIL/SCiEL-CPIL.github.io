def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    T = _construct_tensor(n, m, p)
    
    # Build rank 44
    U44_list, V44_list, W44_list = _build_rank44()
    U44 = np.array(U44_list)
    V44 = np.array(V44_list)
    W44 = np.array(W44_list)
    
    best_U, best_V, best_W = U44_list, V44_list, W44_list
    best_rank = 44
    
    # Try to reduce rank
    for target_rank in [43, 42]:
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 20:
                break
            if found:
                break
                
            rng = np.random.RandomState(seed)
            # Remove (44 - target_rank) columns
            cols_to_remove = rng.choice(44, size=44-target_rank, replace=False)
            mask = [i for i in range(44) if i not in cols_to_remove]
            
            U_init = U44[:, mask]
            V_init = V44[:, mask]
            W_init = W44[:, mask]
            
            # Noise
            U_init += rng.randn(*U_init.shape) * 1e-3
            V_init += rng.randn(*V_init.shape) * 1e-3
            W_init += rng.randn(*W_init.shape) * 1e-3
            
            U_opt, V_opt, W_opt, err = _alternating_minimization(U_init, V_init, W_init, T, max_iter=100)
            
            if err < 1e-5:
                U_round = np.round(U_opt * 2) / 2.0
                V_round = np.round(V_opt * 2) / 2.0
                W_round = np.round(W_opt * 2) / 2.0
                
                if _verify_half_integer(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                    best_U, best_V, best_W = U_round.tolist(), V_round.tolist(), W_round.tolist()
                    best_rank = target_rank
                    found = True
                    break
                    
    return {'U': best_U, 'V': best_V, 'W': best_W}
