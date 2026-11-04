def lin(block, i, j):
    br, bc = divmod(block, 2)
    return (br*3+i)*6 + (bc*3+j)
