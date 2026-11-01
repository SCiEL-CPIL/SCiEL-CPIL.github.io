    def greedy_min_degree(randomize=False):
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]

            if len(cands) > 1:
                if randomize:
                    # Pick randomly from candidates
                    chosen = cands[rng.integers(len(cands))]
                else:
                    scores = np.zeros(len(cands))
                    for idx, c in enumerate(cands):
                        nbrs_c = neighbors[c]
                        alive_nbrs = nbrs_c[alive[nbrs_c]]
                        if len(alive_nbrs) > 0:
                            scores[idx] = np.sum(degrees[alive_nbrs])
                        else:
                            scores[idx] = 0
                    chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set
