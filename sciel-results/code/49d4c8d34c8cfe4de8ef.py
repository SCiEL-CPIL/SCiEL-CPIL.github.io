    bits_i = (inputs & mask_i) != 0
    bits_j = (inputs & mask_j) != 0
    min_bit = bits_i & bits_j
    max_bit = bits_i | bits_j
    result = inputs & ~(mask_i | mask_j)
    result = result | np.where(min_bit, mask_i, np.uint64(0))
    result = result | np.where(max_bit, mask_j, np.uint64(0))
    