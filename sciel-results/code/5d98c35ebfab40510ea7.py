    def factors_valid(Uf, Vf, Wf, tol=2e-9):
        if Uf.shape[0] != n * m or Vf.shape[0] != m * p or Wf.shape[0] != p * n:
            return False
        if not (Uf.shape[1] == Vf.shape[1] == Wf.shape[1]):
            return False
        
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_reshaped = Uf.reshape(n, m, -1)
        V_reshaped = Vf.reshape(m, p, -1)
        W_reshaped = Wf.reshape(p, n, -1)
        
        # Compute T_ijk = sum_r U_ijk_r V_jkr W_kir
        # Note: W is indexed by (k, i) in the tensor product W_{ki}
        # So W_reshaped is (p, n, R) which is (k, i, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target tensor is all 1s
        # Check max deviation
        return np.max(np.abs(T - 1.0)) < tol
    