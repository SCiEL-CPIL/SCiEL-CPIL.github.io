    def get_metrics(v_int8):
        # v_int8 is int8 array of 0s and 1s
        conv = np.convolve(v_int8, v_int8, mode='full')
        targets_ordered = conv[target_min:target_max+1]
        
        even_indices = np.arange(0, target_len, 2)
        diag = np.zeros(target_len, dtype=np.int32)
        v_indices = 64 + even_indices // 2
        diag[even_indices] = v_int8[v_indices]
        
        unordered = (targets_ordered.astype(np.int32) + diag) // 2
        num_zeros = np.sum(unordered == 0)
        max_mult = np.max(unordered)
        return max_mult, num_zeros, unordered
