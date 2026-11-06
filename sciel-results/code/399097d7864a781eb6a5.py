    for k in range(1, n + 2):
        val = old_dp_w[k] if k <= n - 1 else 0
        if k - 1 >= 0 and k - 1 <= n - 1:
            val |= (old_dp_w[k-1] << x)
        new_list[k] = val
    