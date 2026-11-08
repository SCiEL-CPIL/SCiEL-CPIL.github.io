<<<<<<< SEARCH
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
=======
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": np.array(U, dtype=float),
        "V": np.array(V, dtype=float),
        "W": np.array(W, dtype=float)
    }
>>>>>>> REPLACE
