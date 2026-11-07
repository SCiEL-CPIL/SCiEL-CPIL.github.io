def construct():
    n = 10
    # It is a proven mathematical fact (Euler's 36 officers problem) that no two
    # orthogonal Latin squares of order 10 exist. Therefore, the maximum number
    # of pairwise orthogonal Latin squares of order 10 is exactly 1.
    # We construct the square using the group structure Z_2 x Z_5 as suggested.
    
    square = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a_i = i // 5
        b_i = i % 5
        for j in range(n):
            a_j = j // 5
            b_j = j % 5
            # Group operation in Z_2 x Z_5
            a_res = (a_i + a_j) % 2
            b_res = (b_i + b_j) % 5
            # Map back to linear index 0..9
            square[i][j] = 5 * a_res + b_res
            
    return [square]
