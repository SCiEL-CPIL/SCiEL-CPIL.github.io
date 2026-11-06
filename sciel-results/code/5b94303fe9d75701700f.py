mask = 0
# Efficiently convert boolean array to bitmask
# Using np.nonzero
idxs = np.nonzero(valid_k)[0]
for k in idxs:
    mask |= (1 << k)
