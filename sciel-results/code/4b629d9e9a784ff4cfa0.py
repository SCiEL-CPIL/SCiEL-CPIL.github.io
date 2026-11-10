    for mask in range(1, 1024):
        s, sz = mask_info[mask]
        if sz < min_size_for_residue[s]:
            min_size_for_residue[s] = sz
            best_mask_for_residue[s] = mask
