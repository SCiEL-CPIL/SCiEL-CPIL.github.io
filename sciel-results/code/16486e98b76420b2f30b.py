# A block: (rows, cols)  ;  B block: (rows, cols)  ;  C block: (rows, cols)
# The product A_block * B_block must have shape (A_rows, B_cols).
blocks = {
    # key: (A_geom, B_geom, C_geom, dec)
    "3x3": ( ([0,1,2],[0,1,2]), ([0,1,2],[0,1,2]), ([0,1,2],[0,1,2]), "3x3" ),
    "3x2": ( ([0,1,2],[3,4]),   ([3,4],[0,1,2]),   ([0,1,2],[3,4]),   "3x2" ),  # A 3x2 * B 2x3 -> C 3x3? 
    ...
}
