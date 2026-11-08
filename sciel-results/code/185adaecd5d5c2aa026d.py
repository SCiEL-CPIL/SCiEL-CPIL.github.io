def neg_score(p):
    # p is (13, 2)
    # Normalize: center at origin
    p_centered = p - np.mean(p, axis=0)
    # Scale: make max norm 1? Or just let the score function handle it?
    # The score function divides by hull area.
    # If we scale p by s, hull area scales by s^2, triangle area scales by s^2. Ratio is invariant.
    # However, if s -> 0 or inf, we lose precision.
    # Let's normalize to unit variance or max radius 1.
    max_norm = np.max(np.linalg.norm(p_centered, axis=1))
    if max_norm > 0:
        p_norm = p_centered / max_norm
    else:
        p_norm = p_centered
        
    s = fast_score(p_norm)
    return -s
