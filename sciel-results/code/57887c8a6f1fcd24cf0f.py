    if result_17:
        n_old = 17
        n_new = 18
        num_old = n_old * (n_old - 1) // 2
        num_new = n_new * (n_new - 1) // 2
        colors_18 = [-1] * num_new
        
        for idx_old in range(num_old):
            # Find i, j for idx_old in K_17
            # Inverse of idx = i*n - i(i+1)/2 + j - i - 1
            # This is tedious. Easier to iterate i, j.
            pass
        
        # Easier: Iterate i, j for K_18
        idx_new = 0
        for i in range(n_new):
            for j in range(i + 1, n_new):
                if i < n_old and j < n_old:
                    # Find idx_old for (i,j) in K_17
                    idx_old = i * n_old - i * (i + 1) // 2 + j - i - 1
                    colors_18[idx_new] = result_17['colors'][idx_old]
                else:
                    colors_18[idx_new] = random.randint(0, 3)
                idx_new += 1
