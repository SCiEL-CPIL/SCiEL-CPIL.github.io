def _to_frac_safe(x):
    return Fraction(int(round(x * 2))) / 2
