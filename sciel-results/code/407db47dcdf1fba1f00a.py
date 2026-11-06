        for p in small_primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            if start < p * p:
                start = p * p
            
            # Vectorized SPF marking
            indices = np.arange(start, R + 1, p) - L
            mask = spf[indices] == 0
            if np.any(mask):
                spf[indices[mask]] = p
