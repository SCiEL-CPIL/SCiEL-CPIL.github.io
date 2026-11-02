# Inside construct
for r2 in [17, 16]:
    for seed in range(10):
        res = try_find_m332_gradient(r2, max_iter=5000, lr=0.05, seed=seed)
        if res:
            # Combine
            # Verify
            # Return
            pass
