def _E_cpl_vec(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """Vectorized E(a) for CPL. a can be array."""
    Om_de = 1.0 - Om_m - Om_r
    a = np.asarray(a, dtype=float)
    
    # Ensure w0, wa are broadcastable to a
    # If a is (N, 8), w0/wa should be (N, 1)
    if a.ndim > 1:
        w0_arr = np.asarray(w0, dtype=float)
        wa_arr = np.asarray(wa, dtype=float)
        if w0_arr.ndim == 1:
            w0_arr = w0_arr[:, np.newaxis]
        if wa_arr.ndim == 1:
            wa_arr = wa_arr[:, np.newaxis]
    else:
        w0_arr = np.asarray(w0, dtype=float)
        wa_arr = np.asarray(wa, dtype=float)

    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa_arr * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0_arr + wa_arr)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)
