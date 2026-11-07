if window_count == n_succ_window:
    p = success_count / n_succ_window
    if p > 0.2:
        sigma *= 1.5
    elif p < 0.2:
        sigma *= 0.75
    # else unchanged
    success_count = 0
    window_count = 0
