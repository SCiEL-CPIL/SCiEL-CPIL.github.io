def _mask_to_list(mask):
    """Convert bitmask to sorted list of vertex indices."""
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
    return result
