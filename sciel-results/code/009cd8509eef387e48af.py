    # Precompute popcount for 16-bit values
    popcount_table = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
    
    def fast_popcount(arr):
        # arr is uint64
        # Split into 4 16-bit parts
        a = (arr & 0xFFFF).astype(np.uint16)
        b = ((arr >> 16) & 0xFFFF).astype(np.uint16)
        c = ((arr >> 32) & 0xFFFF).astype(np.uint16)
        d = ((arr >> 48) & 0xFFFF).astype(np.uint16)
        
        return popcount_table[a].astype(np.uint32) + \
               popcount_table[b].astype(np.uint32) + \
               popcount_table[c].astype(np.uint32) + \
               popcount_table[d].astype(np.uint32)
    