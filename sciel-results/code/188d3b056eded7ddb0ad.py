    def get_333_decomposition():
        # Returns U, V, W of shape 9x23
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # Terms are (u, v, w)
        terms = [
            ([1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]),
            # ... 21 more terms
        ]
        for r, (u, v, w) in enumerate(terms):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        return U, V, W
    