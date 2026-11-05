    def is_valid(adj, n):
        for k in range(4):
            for i in range(n):
                # Neighbors j such that i < j < n
                # adj[k][i] has bits for all neighbors.
                # We want j > i and j < n.
                mask = ((1 << n) - 1) & ~((1 << (i + 1)) - 1)
                m = adj[k][i] & mask
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    # Check common neighbors w such that w < n
                    # adj[k][i] & adj[k][j]
                    # We only care if ANY common neighbor exists.
                    # The common neighbor w must be < n.
                    # Since we are checking validity of the subgraph 0..n-1,
                    # any common neighbor w < n forms a triangle.
                    # But adj[k][i] might have bits >= n?
                    # In our construction, we only set bits < N_TARGET (50).
                    # If we check n < 50, bits >= n are irrelevant for the triangle
                    # *if* the triangle is within 0..n-1.
                    # Wait. If w >= n, it's not in the subgraph.
                    # So we should mask the common neighbors too.
                    common = (adj[k][i] & adj[k][j]) & ((1 << n) - 1)
                    if common:
                        return False
                    m ^= lb
        return True
