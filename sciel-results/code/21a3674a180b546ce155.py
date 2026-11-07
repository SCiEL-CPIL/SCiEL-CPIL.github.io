def construct():
    n = 10
    squares = []
    
    # Generate 6 squares based on Z2 x Z5 structure
    # i = 5*a + b, j = 5*c + d
    # a,c in {0,1}, b,d in {0,1,2,3,4}
    # X = a+c, Y = b+d
    
    for k in range(5):
        sq = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            a = i // 5
            b = i % 5
            for j in range(n):
                c = j // 5
                d = j % 5
                X = (a + c) % 2
                Y = (b + d) % 5
                sq[i][j] = 5 * X + (Y + k * X) % 5
        squares.append(sq)
        
    # 6th square
    sq6 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a = i // 5
        b = i % 5
        for j in range(n):
            c = j // 5
            d = j % 5
            X = (a + c) % 2
            Y = (b + d) % 5
            sq6[i][j] = 5 * ((X + Y) % 2) + Y
    squares.append(sq6)
    
    return squares
