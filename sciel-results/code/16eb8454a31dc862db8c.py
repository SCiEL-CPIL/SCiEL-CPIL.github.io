def _value(c0,c1,c2):
    # normalize
    n = np.linalg.norm([c0,c1,c2])
    if n < 1e-12: return BIG
    c0,c1,c2 = c0/n, c1/n, c2/n
    coeffs = _q_coeffs(c0,c1,c2)  # [a5..a0]
    roots = np.roots(coeffs)
    best = None
    for rt in roots:
        if abs(rt.imag) < 1e-8*max(1,abs(rt.real)):
            yr = rt.real
            if yr > 1e-9:
                # check sign change
                eps = 1e-6*max(1,abs(yr))
                # evaluate Q at yr-eps and yr+eps
                m1 = _eval_Q(coeffs, yr-eps)
                m2 = _eval_Q(coeffs, yr+eps)
                if m1*m2 < 0:  # sign change -> odd multiplicity
                    if best is None or yr>best: best=yr
    if best is None: return BIG
    return best/(2*np.pi)
