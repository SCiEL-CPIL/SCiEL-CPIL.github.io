<<<<<<< SEARCH
    ax = x[:, None] + (c[None, :] * np.arange(_N) / _N)
    bx = ax + _W
    # y-intervals at height c
    ay = y[:, None] + (c[None, :] * np.arange(_N) / _N)
    by = ay + _W
=======
    shifts = c[None, :] * np.arange(_N)[:, None] / _N
    ax = x[:, None] + shifts
    bx = ax + _W
    # y-intervals at height c
    ay = y[:, None] + shifts
    by = ay + _W
>>>>>>> REPLACE
