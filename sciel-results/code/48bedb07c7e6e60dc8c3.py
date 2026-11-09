import itertools

        bits = []
        m = rest_mask
        while m:
            lsb = m & -m
            bits.append(lsb)
            m ^= lsb
            
        if pick_size > len(bits):
            return []
            
        forbidden = []
        for combo in itertools.combinations(bits, pick_size):
            subset_mask = 0
            for b in combo:
                subset_mask |= b
            c_mask = core | subset_mask
            forbidden.append(c_mask)
