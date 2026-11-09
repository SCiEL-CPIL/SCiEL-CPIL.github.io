    def is_valid(adj):
        for i in range(n):
            mask_i = adj[i]
            for j in range(i + 1, n):
                common = mask_i & adj[j]
                if popcount(common) > 2:
                    return False
        return True
