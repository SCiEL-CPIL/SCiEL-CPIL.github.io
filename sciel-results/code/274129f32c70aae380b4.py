        def _make_3x3(entries):
            mat = np.zeros((3, 3), dtype=float)
            for (i, j), val in entries.items():
                mat[i, j] = val
            return mat

        terms = []
        # Term 0
        terms.append((_make_3x3({(0,0): 1, (1,1): 1, (2,2): 1}), 
                      _make_3x3({(0,0): 1, (1,1): 1, (2,2): 1}), 
                      _make_3x3({(0,0): 1, (1,1): 1, (2,2): 1})))
        # ... and so on for 19 terms.
