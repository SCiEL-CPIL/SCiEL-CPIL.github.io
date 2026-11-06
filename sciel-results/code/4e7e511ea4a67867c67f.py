    def get_seed_colors(n):
        colors = [np.random.randint(4) for _ in range(n * (n - 1) // 2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i < 32 and j < 32:
                    colors[idx] = base_colors[idx]
                # else remains random
                idx += 1
        return colors
