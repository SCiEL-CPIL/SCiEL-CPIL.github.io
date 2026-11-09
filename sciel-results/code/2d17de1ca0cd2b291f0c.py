    def verify(U, V, W):
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_rec, T)
