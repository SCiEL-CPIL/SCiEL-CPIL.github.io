    def get_L(counts):
        idx = np.argmax(counts < 2)
        if counts[idx] >= 2:
            return M_MAX
        return idx
