    def is_valid(S_mask, new_point_idx):
        s = S_mask
        while s:
            lsb = s & -s
            i = lsb.bit_length() - 1
            s ^= lsb
            s2 = S_mask
            while s2:
                lsb2 = s2 & -s2
                j = lsb2.bit_length() - 1
                s2 ^= lsb2
                if i == j:
                    continue
                if (conflict_mask[i][j] >> new_point_idx) & 1:
                    return False
        return True
    