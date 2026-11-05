    def _score_masks(adj_masks, deg):
        best = 0
        for u in range(N):
            for v in range(u+1, N):
                if (adj_masks[u] >> v) & 1:
                    common = adj_masks[u] & adj_masks[v]
                    # Iterate over set bits in common
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                        common ^= lsb
        return best
    