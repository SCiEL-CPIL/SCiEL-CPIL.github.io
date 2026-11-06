        if n_jobs > 30:
            # Sample moves
            num_samples = 500
            for _ in range(num_samples):
                # Random swap
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j: continue
                # ... evaluate
                # Random insertion
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j: continue
                # ... evaluate
        else:
            # Full scan
            # ...
