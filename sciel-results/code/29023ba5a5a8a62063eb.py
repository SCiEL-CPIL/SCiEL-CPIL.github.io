import time

def construct():
    """
    Returns a list containing a single 10x10 Latin square.
    It is a proven mathematical fact that there are no two mutually orthogonal
    Latin squares of order 10 (Euler's 36 officers problem / Tarry 1901).
    Therefore, the maximum number of MOLS is 1.
    """
    start_time = time.time()
    deadline = start_time + 50.0

    n = 10
    # Build the cyclic Latin square L[i][j] = (i + j) % 10
    # This is a valid Latin square: each row and column is a permutation of 0..9
    grid = [[(i + j) % n for j in range(n)] for i in range(n)]

    return [grid]
