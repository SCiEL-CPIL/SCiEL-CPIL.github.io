    def compute_distincts(col):
        k4_colors = col[k4s_np] # (1820, 6)
        sorted_k4 = np.sort(k4_colors, axis=1)
        # distincts = 1 + number of changes
        # change at i if sorted[i] != sorted[i-1]
        # We can do:
        # diffs = sorted_k4[:, 1:] != sorted_k4[:, :-1]
        # distincts = 1 + diffs.sum(axis=1)
        diffs = sorted_k4[:, 1:] != sorted_k4[:, :-1]
        return 1 + diffs.sum(axis=1)
    