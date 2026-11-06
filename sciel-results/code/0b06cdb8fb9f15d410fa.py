    def reduce_code(code, counts, neighbor_bits):
        # code is a list of ints
        # counts is np.array
        # We need to update C_bits too?
        # Let's pass C_bits and return updated code, counts, C_bits
        C_set = set(code)
        C_bits = 0
        for c in code:
            C_bits |= (1 << c)
            
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if redundant
                # min(counts[neighbors[c]]) >= 2
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    C_bits &= ~(1 << c)
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code, counts, C_bits
    