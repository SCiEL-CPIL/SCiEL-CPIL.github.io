packed_bytes = np.packbits(packed_bits, axis=1, bitorder='little') # shape (num_blocks, 8)
packed_uint64 = packed_bytes.view(np.uint64) # shape (num_blocks,)
