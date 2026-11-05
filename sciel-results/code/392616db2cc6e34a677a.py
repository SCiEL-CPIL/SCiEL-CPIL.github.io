def construct():
    candidates = []
    # Generate numbers with base 3 digits 0 or 1
    # Max value 512
    # 3^0 to 3^5 cover up to 364
    # 3^6 is 729
    # So we use bits 0..5
    
    for mask in range(1, 1 << 6):  # 1 to 63
        val = 0
        bit = 1
        m = mask
        while m:
            if m & 1:
                val += bit
            m >>= 1
            bit *= 3
        if val <= 512:
            candidates.append(val)
            
    # Sort and take first 48
    candidates.sort()
    return candidates[:48]
