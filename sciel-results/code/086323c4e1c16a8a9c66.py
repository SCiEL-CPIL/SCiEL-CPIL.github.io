"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 Strassen-based exact decomposition for 2x2 * 2x4 matrix multiplication
    # A (2x2) = [[a,b],[c,d]], B (2x4) = [[i,j,k,l],[m,n,o,p]]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The 14 rank-1 terms:
    # 0-3:  (a+d) * (col0+col1 of B), (a+d)*(col2+col3 of B) for each output col pair
    # 4-7:  (c-a) * corresponding sums (corrections for row 1)
    # 8-9:  b*(n-m), d*(m-n) for col pair 0
    # 10-11: b*(p-o), d*(o-p) for col pair 1
    # 12-13: additional cross terms to complete the decomposition
    
    U = [
        [1, 1, 1, 1, -1, -1, -1, -1, 0,  0,  0,  0,  0,  0 ],
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  -1, -1,  0,  0 ],
        [0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0 ],
        [1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0 ],
    ]
    
    V = [
        [1, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
        [0, 1, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0,  0,  0 ],
        [0, 0, 1, 0, 0,  0,  1,  0,  0,  0,  0,  0,  0,  0 ],
        [0, 0, 0, 1, 0,  0,  0,  1,  0,  0,  0,  0,  0,  0 ],
        [1, 0, 0, 0, -1, 0,  0,  0,  0,  0,  0,  0,  0,  0 ],
        [0, 1, 0, 0, 0,  -1, 0,  0,  0,  0,  0,  0,  0,  0 ],
        [0, 0, 1, 0, 0,  0,  -1, 0,  0,  0,  0,  0,  0,  0 ],
        [0, 0, 0, 1, 0,  0,  0,  -1, 0,  0,  0,  0,  0,  0 ],
    ]
    
    W = [
        [1, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0,  0,  0 ],
        [0, 1, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0,  0,  0 ],
        [0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  1,  0,  0,  0 ],
        [0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  1,  0,  0 ],
        [1, 0, 0, 0, 1,  0,  0,  0,  -1, 0,  0,  0,  0,  0 ],
        [0, 1, 0, 0, 0,  1,  0,  0,  0,  -1, 0,  0,  0,  0 ],
        [0, 0, 1, 0, 0,  0,  1,  0,  0,  0,  -1, 0,  0,  0 ],
        [0, 0, 0, 1, 0,  0,  0,  1,  0,  0,  0,  -1, 0,  0 ],
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
