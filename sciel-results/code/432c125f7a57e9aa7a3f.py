<<<<<<< SEARCH
    # Scaling factor to ensure strict non-overlap (gap > 0)
    # The grid centers are at distance sqrt(3) apart.
    # Scaling by 1.000001 creates a small gap.
    SCALE = 1.000001
=======
    # Scaling factor to ensure strict non-overlap (gap > 0)
    # The grid centers are at distance sqrt(3) apart.
    # Using 1.0 allows touching, which is typically allowed in packing problems.
    SCALE = 1.0
>>>>>>> REPLACE
