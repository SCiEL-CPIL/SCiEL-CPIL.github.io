    P = np.array(sorted(list(pows)), dtype=np.int32)
    # ...
    sums = P[:, None] + P[None, :]
    # sums is int32
    max_sum = 2 * L
    counts = np.bincount(sums, minlength=max_sum + 1)
    # counts is int64 by default? No, bincount returns int64.
    # We can cast counts to int32 if max count < 2*10^9.
    # Max count is K ~ 4500. So int32 is fine.
    counts = counts.astype(np.int32)
