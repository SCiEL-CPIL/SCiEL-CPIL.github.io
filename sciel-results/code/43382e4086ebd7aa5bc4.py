    def is_covering_code(code_list_ints):
        covered = np.zeros(total_words, dtype=bool)
        for c in code_list_ints:
            covered[covers[c]] = True
        return np.all(covered)
