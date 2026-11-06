    def generate_sums(weights_part, size):
        idx = np.arange(size, dtype=np.int32)
        k = 0
        temp_size = size
        while temp_size > 1:
            temp_size //= 3
            k += 1
        digits = np.zeros((size, k), dtype=np.uint8)
        rem = idx.copy()
        for j in range(k):
            digits[:, j] = rem % 3
            rem //= 3
        # Convert to coeffs
        # digits 0->-1, 1->0, 2->1
        # coeffs = digits - 1
        # Must be signed
        coeffs = digits.astype(np.int8)
        coeffs -= 1
        sums = coeffs.dot(weights_part)
        return idx, sums
