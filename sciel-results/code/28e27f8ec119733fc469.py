BCLR_3x3_TERMS = [
    # (u, v, w)
    # u: A indices 0..8 (row-major)
    # v: B indices 0..8 (row-major)
    # w: C indices 0..8 (col-major: col*3+row)
    
    # Term 1
    ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
    # Term 2
    ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
    # ...
]
