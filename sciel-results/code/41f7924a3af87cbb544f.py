    def _compute_score_bitmask(adj):
        n = 16
        deg = [bin(x).count('1') for x in adj]
        max_sum = 0
        # Iterate u, v, w with u < v < w
        for u in range(n):
            if not adj[u]: continue
            # Potential neighbors v > u
            # We can iterate v in adj[u] where v > u
            # But simpler: nested loops
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)): continue
                # Common neighbors of u and v that are > v? No, just w > v
                # To ensure u < v < w, we check w > v
                # common = adj[u] & adj[v]
                # We only care about w > v
                # Mask for w > v: ~((1 << (v+1)) - 1)
                mask_gt_v = ~((1 << (v+1)) - 1) & ((1 << n) - 1)
                common = adj[u] & adj[v] & mask_gt_v
                while common:
                    # Get lowest set bit
                    lsb = common & -common
                    w = lsb.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb
        return max_sum
    