    def calc_score_limited(adj, deg, limit):
        max_sum = 0
        for u in range(n):
            au = adj[u]
            du = deg[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    dv = deg[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = du + dv + deg[w]
                        if s >= limit:
                            return s # Early exit, not better than limit
                        if s > max_sum:
                            max_sum = s
                        c ^= lsb
        return max_sum
