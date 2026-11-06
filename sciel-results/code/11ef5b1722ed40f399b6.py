def is_sidon_fast(subset):
    sums = [False] * 257
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if sums[s]:
                return False
            sums[s] = True
    return True
