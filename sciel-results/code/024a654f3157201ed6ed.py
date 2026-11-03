    def update_and_get_term(h, ac, i, old_val):
        idx_plus = i + k_arr
        idx_minus = i - k_arr
        
        mask_plus = idx_plus < N
        mask_minus = idx_minus >= 0
        
        term_sum = np.zeros(N-1)
        
        if np.any(mask_plus):
            term_sum[mask_plus] = h[idx_plus[mask_plus]]
        if np.any(mask_minus):
            term_sum[mask_minus] += h[idx_minus[mask_minus]]
            
        delta = 2.0 * old_val * term_sum
        ac[1:] += delta
        return delta

    # In loop:
    delta = update_and_get_term(h, ac, i, old_val)
    # ...
    # Revert:
    h[i] = old_val
    ac[1:] -= delta
