<<<<<<< SEARCH
        M = k * sigma_k
        if M <= limit:
            if M in groups:
                groups[M].append(k)
            else:
                groups[M] = [k]
        else:
            # Since k is increasing, M is generally increasing.
            # However, sigma(k)/k can drop.
            # But for k > sqrt(limit), M >= k^2 > limit.
            # So we can break if k > sqrt(limit).
            # sqrt(10^18) = 10^9.
            # So we won't break early based on M > limit until k is huge.
            pass
=======
        M = k * sigma_k
        # For N=50M, M is always < 10^18.
        if M in groups:
            groups[M].append(k)
        else:
            groups[M] = [k]
>>>>>>> REPLACE
