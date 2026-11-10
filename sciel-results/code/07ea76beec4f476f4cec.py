    valid_r = None
    for r in range(m_add):
        compatible = True
        for m2, r2 in current_set.items():
            if r % gcd(m_add, m2) == r2 % gcd(m_add, m2):
                compatible = False
                break
        if compatible:
            valid_r = r
            break
    