    def verify_and_snap(U_flat, V_flat, W_flat):
        r = U_flat.size // (n * m)
        U_3d = U_flat.reshape(n, m, r)
        V_3d = V_flat.reshape(m, p, r)
        W_3d = W_flat.reshape(p, n, r)
        
        U_snap = np.round(U_3d * 2) / 2
        V_snap = np.round(V_3d * 2) / 2
        W_snap = np.round(W_3d * 2) / 2
        
        T_hat = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_hat[i, j, k] = np.sum(U_snap[i, j, :] * V_snap[j, k, :] * W_snap[k, i, :])
                    
        if np.linalg.norm(T - T_hat) < 1e-9:
            return {
                "U": U_snap.transpose(0, 1, 2).reshape(n * m, r).tolist(),
                "V": V_snap.transpose(0, 1, 2).reshape(m * p, r).tolist(),
                "W": W_snap.transpose(0, 1, 2).reshape(p * n, r).tolist()
            }
        return None
