# Precompute multiples masks
multiples_mask = [0] * (MAX_SUM + 1)
for i in range(1, MAX_SUM + 1):
    mask = 0
    for j in range(i * 2, MAX_SUM + 1, i):
        mask |= (1 << j)
    multiples_mask[i] = mask
