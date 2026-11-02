<<<<<<< SEARCH
            if r < 0.55 and len(not_in_set) > 0:
                # ADD: pick vertex with fewest in-set conflicts
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1

            elif r < 0.75 and len(in_set_indices) > 0:
                # REMOVE a random in-set vertex
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    stuck += 1
                else:
                    stuck += 1

            else:
                # SWAP: remove one in-set, add one not-in-set
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        # Clean swap
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                        stuck += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                cc[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            cc[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck = 0
                            else:
                                stuck += 1
                        else:
                            stuck += 1
=======
            if r < 0.40 and len(not_in_set) > 0:
                # ADD: pick vertex with fewest in-set conflicts
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1

            elif r < 0.50 and len(in_set_indices) > 0:
                # REMOVE a random in-set vertex
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    stuck += 1
                else:
                    stuck += 1

            elif r < 0.70 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP: remove one in-set, add one not-in-set
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add = not_in_set[rng.integers(len(not_in_set))]
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                if len(other_conflicts) == 0:
                    # Clean swap
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    cc[neighbors[v_remove]] -= 1
                    cc[neighbors[v_add]] += 1
                    stuck += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        cc[neighbors[v_add]] += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck = 0
                        else:
                            stuck += 1
                    else:
                        stuck += 1

            else:
                # 2-for-1: remove two in-set vertices, add one not-in-set vertex
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    # Pick a not-in-set vertex with small conflict count
                    counts = cc[not_in_set]
                    # Prefer vertices with 1-3 conflicts for 2-for-1 to work
                    mask = (counts >= 1) & (counts <= 3)
                    if np.any(mask):
                        cands = not_in_set[mask]
                        v_add = cands[rng.integers(len(cands))]
                        nbrs_in_set = np.where(conflicts[v_add] & in_set)[0]
                        # Try to remove a subset of nbrs_in_set (size 1 or 2)
                        if len(nbrs_in_set) == 1:
                            # Just remove that one
                            to_remove = nbrs_in_set
                            delta = 1 - len(to_remove)  # 0
                        elif len(nbrs_in_set) == 2:
                            to_remove = nbrs_in_set
                            delta = 1 - len(to_remove)  # -1
                        elif len(nbrs_in_set) == 3:
                            # Remove 2 of the 3
                            pair = rng.choice(3, size=2, replace=False)
                            to_remove = nbrs_in_set[pair]
                            delta = 1 - len(to_remove)  # -1
                        else:
                            to_remove = None

                        if to_remove is not None and len(to_remove) > 0:
                            accept = delta >= 0 or (rng.random() < np.exp(delta / T))
                            if accept:
                                in_set[to_remove] = False
                                for u in to_remove:
                                    cc[neighbors[u]] -= 1
                                in_set[v_add] = True
                                current_size += delta
                                cc[neighbors[v_add]] += 1
                                if current_size > local_best_size:
                                    local_best_size = current_size
                                    local_best_in_set = in_set.copy()
                                    stuck = 0
                                else:
                                    stuck += 1
                            else:
                                stuck += 1
                        else:
                            stuck += 1
                    else:
                        stuck += 1
                else:
                    stuck += 1
>>>>>>> REPLACE
