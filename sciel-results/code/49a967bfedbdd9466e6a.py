    def enforce_d4_symmetry(cell):
        """
        Enforce D4 symmetry by copying from the fundamental domain.
        Fundamental domain: 0 <= i <= j and i + j < size
        """
        N = size
        sym_cell = np.zeros((N, N), dtype=int)
        for i in range(N // 2):
            for j in range(i, N - i):
                val = cell[i, j]
                # Orbit of (i, j):
                # (i, j)
                # (j, i)
                # (N-1-i, N-1-j)
                # (N-1-j, N-1-i)
                sym_cell[i, j] = val
                sym_cell[j, i] = val
                sym_cell[N-1-i, N-1-j] = val
                sym_cell[N-1-j, N-1-i] = val
        return sym_cell
