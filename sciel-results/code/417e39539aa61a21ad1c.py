def _score_fast(coeffs, num_samples=1024):
    n = len(coeffs)
    zs = _get_zs_1024()
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)
