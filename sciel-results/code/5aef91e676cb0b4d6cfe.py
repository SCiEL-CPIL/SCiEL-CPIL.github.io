<<<<<<< SEARCH
        pivot = rng.randint(1, n - 2)
        max_len = min(n - 1 - pivot, 20)
        seg_len = rng.randint(2, max_len)
        if seg_len > n - 1 - pivot:
            continue
=======
        pivot = rng.randint(1, n - 2)
        max_len = min(n - 1 - pivot, 20)
        if max_len < 2:
            continue
        seg_len = rng.randint(2, max_len)
>>>>>>> REPLACE
