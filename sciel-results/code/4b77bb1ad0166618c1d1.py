    def find_c6_random(n, adj, rng):
        for c in range(4):
            adj_c = adj[c]
            edges_c = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_c[i] >> j) & 1]
            rng.shuffle(edges_c)
            for u, v in edges_c:
                adj_c[u] &= ~(1 << v)
                adj_c[v] &= ~(1 << u)
                if has_path_len_5_bitset(u, v, adj_c, n):
                    adj_c[u] |= (1 << v)
                    adj_c[v] |= (1 << u)
                    return c, u, v
                adj_c[u] |= (1 << v)
                adj_c[v] |= (1 << u)
        return None
