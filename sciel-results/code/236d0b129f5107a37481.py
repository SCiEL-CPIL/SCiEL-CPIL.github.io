<<<<<<< SEARCH
    # Verify schoolbook
    U3 = U_sb.reshape(n, m, R_sb)
    V3 = V_sb.reshape(m, p, R_sb)
    W3 = W_sb.reshape(p, n, R_sb)
    T_check = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    assert np.array_equal(T_check, target.reshape(n, m, p)), "Schoolbook verification failed"
=======
    # Verify schoolbook
    U3 = U_sb.reshape(n, m, R_sb)
    V3 = V_sb.reshape(m, p, R_sb)
    W3 = W_sb.reshape(p, n, R_sb)
    T_check = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    assert np.array_equal(T_check, target), "Schoolbook verification failed"
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Helper: verify a decomposition
    def verify(U_mat, V_mat, W_mat, R):
        """Verify if U, V, W (as numpy arrays) give the target tensor."""
        if time.time() > deadline:
            return False
        U3 = U_mat.reshape(n, m, R)
        V3 = V_mat.reshape(m, p, R)
        W3 = W_mat.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.array_equal(T, target.reshape(n, m, p))
=======
    # Helper: verify a decomposition
    def verify(U_mat, V_mat, W_mat, R):
        """Verify if U, V, W (as numpy arrays) give the target tensor."""
        if time.time() > deadline:
            return False
        U3 = U_mat.reshape(n, m, R)
        V3 = V_mat.reshape(m, p, R)
        W3 = W_mat.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.array_equal(T, target)
>>>>>>> REPLACE
