    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # Split B into [0:2, 0:2], [0:2, 2:4], [0:2, 4:5]
    add_strassen_2x2x2((0,0), (0,0), (0,0))
    add_strassen_2x2x2((0,0), (0,2), (0,2))
    add_schoolbook_2x2x1((0,0), (0,4), (0,4))
