for i in range(len(blocks)):
    ...
    if not np.any(coverage == 0):
        blocks.pop(i)
        changed = True
    else:
        update_coverage(b, 1)
        i += 1 # Skip to next block if we didn't remove
