<<<<<<< SEARCH
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
=======
    subsets_of_block = [[] for _ in range(n_k)]
    subsets_of_block_arr = [None] * n_k  # numpy arrays for fast indexing
    blocks_of_subset = [[] for _ in range(n_t)]

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
        subsets_of_block_arr[j] = np.array(subsets_of_block[j], dtype=np.int32)
>>>>>>> REPLACE

