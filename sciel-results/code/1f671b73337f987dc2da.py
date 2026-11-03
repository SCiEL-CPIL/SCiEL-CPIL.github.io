    def optimize_for_r(r_target, x_start):
        bounds = (np.array([-1.0] * (2*n)), np.array([1.0] * (2*n)))
        try:
            result = least_squares(
                residuals, 
                x_start, 
                args=(r_target,),
                bounds=bounds,
                method='trf',
                max_nfev=1000,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9
            )
            max_res = np.max(np.abs(result.fun))
            return result.x, max_res
        except Exception:
            return x_start, np.inf
