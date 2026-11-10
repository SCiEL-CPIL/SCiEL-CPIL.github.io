<<<<<<< SEARCH
    start_time = time.time()
    deadline = 600.0  # safety margin

    # Try increasing K until we run out of time
    best_result = None
    best_a = -np.inf

    # Start with a reasonable K and increase
    for K in [5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 60:
            break

        # Compute Mobius function
        mu = _compute_mobius(K)

        # Only include non-zero entries to keep m small
        mask = mu[1:K+1] != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[domain].astype(np.float64)

        if len(domain) == 0:
            continue

        # Compute M(x) for x in [1, 10K] using difference array
        max_x = 10 * K
        diff = np.zeros(max_x + 2, dtype=np.float64)

        for i in range(len(domain)):
            k = int(domain[i])
            f_k = values[i]
            # floor(x/k) = j for x in [j*k, min((j+1)*k - 1, max_x)]
            for j in range(1, max_x // k + 1):
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                if start > max_x:
                    break
                diff[start] += f_k * j
                diff[end + 1] -= f_k * j

        # Prefix sum to get M
        M = np.cumsum(diff)
        M = M[:max_x + 1]

        # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
        if np.max(M[1:max_x+1]) > 1.0001:
            continue

        # Compute objective a = -sum_k f(k) * log(k) / k
        k_float = domain.astype(np.float64)
        a = -np.sum(values * np.log(k_float) / k_float)

        if a > best_a:
            best_a = a
            best_result = (domain, values)

        if time.time() - start_time > deadline - 30:
            break

    if best_result is None:
        # Fallback to the seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result
=======
    start_time = time.time()
    deadline = 600.0  # safety margin

    # Try increasing K until we run out of time
    best_result = None
    best_a = -np.inf

    # Start with a reasonable K and increase
    for K in [5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 60:
            break

        # Compute Mobius function
        mu = _compute_mobius(K)

        # Only include non-zero entries to keep m small
        mask = mu[1:K+1] != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[domain].astype(np.float64)

        if len(domain) == 0:
            continue

        # Compute M(x) for x in [1, 10K] using vectorized approach
        # M(x) = sum_k f(k) * floor(x/k)
        # For each k, floor(x/k) changes at multiples of k
        # We can compute M(x) by iterating over k and adding f(k)*floor(x/k) for all x
        # This is O(m * max_x) which is too slow for large K
        # Instead, use the identity: M(x) = sum_{k|x} f(k) * (x/k) is not quite right
        # Actually M(x) = sum_k f(k) * floor(x/k)
        # Let's use a different approach: for each x, M(x) = sum_k f(k) * floor(x/k)
        # We can vectorize this by creating an array of x values and computing floor(x/k) for each k
        
        max_x = 10 * K
        x_vals = np.arange(1, max_x + 1, dtype=np.int64)
        
        # M(x) = sum_k f(k) * floor(x/k)
        # Vectorize: for each k, compute floor(x/k) and multiply by f(k), then sum
        # This is O(m * max_x) operations but vectorized
        M = np.zeros(max_x, dtype=np.float64)
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for start_idx in range(0, len(domain), batch_size):
            end_idx = min(start_idx + batch_size, len(domain))
            batch_domain = domain[start_idx:end_idx]
            batch_values = values[start_idx:end_idx]
            
            # For each k in batch, add f(k) * floor(x/k) to M
            for i in range(len(batch_domain)):
                k = batch_domain[i]
                f_k = batch_values[i]
                M += f_k * (x_vals // k)
        
        # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
        if np.max(M) > 1.0001:
            continue

        # Compute objective a = -sum_k f(k) * log(k) / k
        k_float = domain.astype(np.float64)
        a = -np.sum(values * np.log(k_float) / k_float)

        if a > best_a:
            best_a = a
            best_result = (domain, values)

        if time.time() - start_time > deadline - 30:
            break

    if best_result is None:
        # Fallback to the seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result
>>>>>>> REPLACE
