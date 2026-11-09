        valid = np.allclose(T, expected, atol=1e-6)
        return bool(valid)
