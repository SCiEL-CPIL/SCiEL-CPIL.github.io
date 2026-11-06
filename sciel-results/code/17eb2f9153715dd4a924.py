def construct():
    # P(x) = 1 + x + x^2 + x^3 + x^4 + x^5 - x^6 - x^7 - x^8 - x^9 - x^10 - x^11
    coeffs = [0] * 12
    coeffs[0] = 1
    coeffs[1] = 1
    coeffs[2] = 1
    coeffs[3] = 1
    coeffs[4] = 1
    coeffs[5] = 1
    coeffs[6] = -1
    coeffs[7] = -1
    coeffs[8] = -1
    coeffs[9] = -1
    coeffs[10] = -1
    coeffs[11] = -1
    return coeffs
