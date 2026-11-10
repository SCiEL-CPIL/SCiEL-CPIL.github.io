for idx in range(m):
    k = domain[idx]
    mu_k = values[idx]
    j_max = x_max // k
    if j_max == 0:
        continue
    # Add mu_k at positions k, 2k, 3k, ..., j_max*k
    positions = np.arange(k, j_max * k + 1, k, dtype=np.int64)
    diff[positions] += mu_k
    # Subtract j_max * mu_k at position min((j_max+1)*k, x_max+1)
    end_pos = min((j_max + 1) * k, x_max + 1)
    diff[end_pos] -= j_max * mu_k
