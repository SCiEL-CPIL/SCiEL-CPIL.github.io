def construct():
    # Generate the Odd-Even Transposition sorting network.
    # This is a valid sorting network for N wires.
    # For N=18, it has 153 comparators.
    network = _odd_even_transposition(N)
    
    # We skip pruning because verifying each candidate network takes too long
    # (approx 3s per check * 153 candidates > 60s limit).
    
    return network
