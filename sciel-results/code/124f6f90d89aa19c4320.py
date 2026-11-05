        candidates_a = []
        # Small a
        for a in range(2, 20):
            if a < n//2:
                candidates_a.append(a)
        # Large a
        for a in range(n//2 - 20, n//2):
            candidates_a.append(a)
        # Fractional
        for d in [3, 4, 5, 6, 7, 8]:
            a = n // d
            if 1 < a < n//2:
                candidates_a.append(a)
        # Unique
        candidates_a = list(set(candidates_a))
