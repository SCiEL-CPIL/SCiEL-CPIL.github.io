        m = old_mask
        while m:
            lsb = m & -m
            old_j = lsb.bit_length() - 1
            new_j = new_index[old_j]
            new_mask |= (1 << new_j)
            m ^= lsb
