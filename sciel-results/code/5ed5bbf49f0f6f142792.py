    def color_within(group, c_start):
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                idx = _edge_idx(group[i], group[j])
                colors[idx] = c_start + (i * (n - 1) - i * (i + 1) // 2 + (j - i - 1))
