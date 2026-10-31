    def construct():
        n_basis = 4
        n_fine = 16384
        t_fine = np.linspace(-0.5, 0.5, n_fine)
        dt = t_fine[1] - t_fine[0]
        
        # Precompute indices for fine grid
        idx = np.floor((t_fine + 0.25) / _dx).astype(int)
        idx = np.clip(idx, 0, N - 1)
        in_support = (t_fine >= -0.25) & (t_fine < 0.25)
        idx_in_support = idx[in_support]
        
        bounds = []
        for _ in range(n_basis):
            bounds.append((-0.25, 0.25))
            bounds.append((0.01, 0.25))
            bounds.append((0.0, 5.0))
            
        def objective(params):
            heights = _gaussian_mixture_heights(params, n_basis)
            l2_sq = np.sum(heights ** 2) * _dx
            if l2_sq <= 1e-15:
                return 1e10
            f_fine = np.zeros(n_fine)
            f_fine[in_support] = heights[idx_in_support]
            conv_fine = fftconvolve(f_fine, f_fine, mode='full') * dt
            max_conv = np.max(conv_fine)
            return max_conv / l2_sq
            
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=300,
            tol=1e-8,
            popsize=15,
            init='sobol',
            polish=True,
            updating='immediate',
            workers=1
        )
        
        best_params = result.x
        best_heights = _gaussian_mixture_heights(best_params, n_basis)
        return list(np.maximum(best_heights, 0.0))
    