# Inside swap loop
blocks_copy = blocks[:]
counts_copy = counts[:]

# ... perform swap and try_delete ...

if len(blocks) < orig_len:
    swaps_done += 1
else:
    blocks = blocks_copy
    counts = counts_copy
