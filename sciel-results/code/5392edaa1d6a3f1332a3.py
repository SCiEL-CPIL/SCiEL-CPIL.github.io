<<<<<<< SEARCH
            CT C = C_mat.T @ C_mat
            CT C += 1e-8 * np.eye(r)
            CT C_inv = np.linalg.inv(CT C)
            V_new = T_flat_v @ C_mat @ CT C_inv  # shape (m*p, r)
=======
            CTC = C_mat.T @ C_mat
            CTC += 1e-8 * np.eye(r)
            CTC_inv = np.linalg.inv(CTC)
            V_new = T_flat_v @ C_mat @ CTC_inv  # shape (m*p, r)
>>>>>>> REPLACE
