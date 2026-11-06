    def _verify_network_fast(comparators):
        """Verify using zero-one principle over all 2^16 inputs (Vectorized)."""
        # Create all 2^16 bit patterns
        # range(2^16) -> binary representation
        # This might use a lot of memory: 65536 * 16 bytes = ~1MB. Fine.
        masks = np.arange(1 << N, dtype=np.uint32)
        # Extract bits
        # bits[:, i] = (masks >> i) & 1
        bits = np.zeros((1 << N, N), dtype=np.uint8)
        for i in range(N):
            bits[:, i] = (masks >> i) & 1
        
        # Apply network
        wires = bits.copy()
        for i, j in comparators:
            min_val = np.minimum(wires[:, i], wires[:, j])
            max_val = np.maximum(wires[:, i], wires[:, j])
            wires[:, i] = min_val
            wires[:, j] = max_val
        
        # Check if sorted
        # wires[:, :-1] <= wires[:, 1:]
        diffs = wires[:, 1:] - wires[:, :-1]
        if np.any(diffs < 0):
            return False
        return True
    