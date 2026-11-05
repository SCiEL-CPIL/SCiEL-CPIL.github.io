    def evaluate(cell, fast=False):
        F_flat = get_fourier_inv_eps(cell)
        F_mat = F_flat[diff_idx]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        kps = k_points[:5] if fast else k_points
        
        for k in kps:
            kg = k + G_list
            kg_sq = np.sum(kg**2, axis=1)
            M = F_mat * kg_sq[None, :]
            eigs = np.linalg.eigvalsh(M)
            w4 = np.sqrt(eigs[3])
            w5 = np.sqrt(eigs[4])
            if w4 > max_omega_4: max_omega_4 = w4
            if w5 < min_omega_5: min_omega_5 = w5
            
        if min_omega_5 <= max_omega_4: return -1.0
        return 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
