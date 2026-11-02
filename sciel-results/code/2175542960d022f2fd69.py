def map_A(local_r, local_c, block_r, block_c):
    # block_r, block_c are 0 or 1
    global_r = block_r * 2 + local_r
    global_c = block_c * 2 + local_c
    return global_r * 4 + global_c # m=4

def map_B(local_r, local_c, block_r, block_c):
    global_r = block_r * 2 + local_r
    global_c = block_c * 2 + local_c
    return global_r * 7 + global_c # p=7

def map_C(local_r, local_c, block_r, block_c):
    # C is p x n.
    # local_r is col of B (0..1), local_c is row of A (0..1)
    global_r = block_r * 2 + local_r
    global_c = block_c * 2 + local_c
    return global_r * 4 + global_c # n=4
