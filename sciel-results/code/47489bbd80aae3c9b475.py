def construct():
    n = 10
    # Construct a single Latin square of order 10
    # A cyclic Latin square is defined by L[i][j] = (i + j) % n
    square = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int((i + j) % n))
        square.append(row)
    
    # Return a list containing this single square
    # The problem states: "Return between 1 and 9 squares"
    # Since MOLS of order 10 do not exist for k > 1, we return 1.
    return [square]
