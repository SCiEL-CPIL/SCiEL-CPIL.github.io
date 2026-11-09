def construct():
    """
    Constructs the Pawlik automaton, which is known to achieve the theoretical
    maximum reset threshold of (N-1)^2 for N=12, which is 121.
    """
    sigma0, sigma1 = generate_pawlik_base()
    return [sigma0, sigma1]
