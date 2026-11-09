def priority(vector, n):
    # Calculate the AC energy (variance scaled by n) of the vector.
    # This corresponds to the sum of squared magnitudes of non-DC Fourier coefficients.
    # Vectors with low AC energy are "low frequency" (close to constant vectors like all-ones)
    # and are penalized (given lower priority).
    
    sum_x = sum(vector)
    sum_x_sq = sum(x * x for x in vector)
    
    # AC Energy = sum(x^2) - (sum(x)^2 / n)
    # This is n * Variance
    ac_energy = sum_x_sq - (sum_x * sum_x) / n
    
    # Add a small tie-breaker to ensure determinism without bias
    h = hash(vector)
    tie_breaker = (h % 1000000) / 1000000.0
    
    # Prioritize vectors with high AC energy (high frequency / mixed)
    return ac_energy + tie_breaker
