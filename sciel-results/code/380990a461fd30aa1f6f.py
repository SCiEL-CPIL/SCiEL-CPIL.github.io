for i, q in enumerate(S_list[:-1]): # All except the new one
    D_dict[q].add(dists[i])
