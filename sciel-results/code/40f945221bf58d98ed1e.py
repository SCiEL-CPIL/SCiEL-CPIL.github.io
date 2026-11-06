def construct():
    # Find largest m such that [m,100] has no subset summing to 200
    # Condition: no single, pair, or triple sums to 200
    # Single: m<=100<200, so no single is 200
    # Pair: max pair = 100+99=199<200, so no pair
    # Triple: min triple = m+(m+1)+(m+2) = 3m+3 > 200 => m > 65.67 => m>=66
    # For m>=66, [m,100] is safe
    # For m<66, there exist triples summing to 200
    m = 66
    return list(range(m, 101))
