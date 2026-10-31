    if result.success:
        z_opt = result.x
        x_opt = z_opt[0:N]
        y_opt = z_opt[N:2*N]
        r_opt = z_opt[2*N:3*N]
        
        if _check_validity(x_opt, y_opt, r_opt):
            return x_opt, y_opt, r_opt, np.sum(r_opt)
        else:
            # Try to fix small violations by clamping?
            # Or just return invalid score
            return None, None, None, -1e10
