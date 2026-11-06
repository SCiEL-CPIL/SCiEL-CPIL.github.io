# Precompute popcount for all 256 possible 8-bit values
popcount8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def batch_popcount(arr):
    """Compute popcount for each uint32 in arr using 8-bit lookup."""
    arr = arr.astype(np.uint8).reshape(-1, 4)  # each uint32 -> 4 uint8s
    return popcount8[arr].sum(axis=1)
