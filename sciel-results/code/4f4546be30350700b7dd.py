def construct():
    # P(x) = 1 + x^2 + x^4 + x^6 + x^8 + x^10 - x^12 - x^14 - x^16 - x^18 - x^20 - x^22
    # This has 12 nonzero terms with coefficients in {-1, 1}
    # The square P^2 will have many cancellations due to the structure
    coeffs = [0] * 23
    # Positive terms at even degrees 0, 2, 4, 6, 8, 10
    for i in range(6):
        coeffs[2*i] = 1
    # Negative terms at even degrees 12, 14, 16, 18, 20, 22
    for i in range(6):
        coeffs[12 + 2*i] = -1
    return coeffs
