    elif r_size == 4:
        # D is a 4-subset of m_u
        u = m_u
        bits = []
        temp = u
        while temp:
            lsb = temp & -temp
            bits.append(lsb)
            temp ^= lsb
        # Iterate 4-combinations of bits
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                for k in range(j+1, len(bits)):
                    for l in range(k+1, len(bits)):
                        m_d = bits[i] | bits[j] | bits[k] | bits[l]
                        m_c = m_inter | m_d # m_inter is 0
                        if m_c in MASK_TO_ID:
                            conflicts.append(MASK_TO_ID[m_c])
