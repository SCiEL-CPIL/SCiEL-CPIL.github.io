    def try_snap(U, V, W, r):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W, r):
            return U_s, V_s, W_s
        # Try other denoms? 1, 2, 4
        U_s = np.round(U)
        V_s = np.round(V)
        W_s = np.round(W)
        if verify(U_s, V_s, W_s, r):
            return U_s, V_s, W_s
        return U, V, W
