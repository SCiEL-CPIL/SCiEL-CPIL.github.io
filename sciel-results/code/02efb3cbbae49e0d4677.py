def find_rank_10_2x2x3():
    # Tensor M_{2,2,3}
    # U dim 4, V dim 6, W dim 6
    T = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T[i*2+j, j*3+k, k*2+i] = 1.0
                
    rank = 10
    # Vars: 4*10 + 6*10 + 6*10 = 160
    
    def residual(params):
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (T - T_approx).flatten()
        
    start_time = time.time()
    time_limit = 30
    
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        params0 = np.random.randn(160) * 0.5
        try:
            res = least_squares(residual, params0, method='lm', max_nfev=2000)
            if res.cost < 1e-10:
                U = res.x[0:40].reshape(4, 10)
                V = res.x[40:100].reshape(6, 10)
                W = res.x[100:160].reshape(6, 10)
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                T_approx_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T - T_approx_snap, 0, atol=1e-5):
                    return U_snap, V_snap, W_snap
        except Exception:
            pass
    return None
