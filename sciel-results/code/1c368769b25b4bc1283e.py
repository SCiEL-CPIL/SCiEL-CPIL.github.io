import time

def construct():
    start_time = time.time()
    
    n = 10
    # Generate 4 MOLS of order 10 from 4 MOLS of order 5
    def make_ls_5(k):
        return [[(i + k*j) % 5 for j in range(5)] for i in range(5)]
    
    def lift_square(L5):
        L10 = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(5):
            for j in range(5):
                val = L5[i][j]
                L10[2*i][2*j] = val
                L10[2*i][2*j+1] = val + 5
                L10[2*i+1][2*j] = val + 2
                L10[2*i+1][2*j+1] = val + 7
        return L10

    mols_5 = [make_ls_5(k) for k in [1, 2, 3, 4]]
    mols_10 = [lift_square(L5) for L5 in mols_5]
    
    # Try to add a 5th square
    # Candidate: Cyclic 1
    s5 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_orthogonal(A, B):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((A[i][j], B[i][j]))
        return len(pairs) == n*n

    valid_squares = mols_10
    if all(is_orthogonal(s5, sq) for sq in valid_squares):
        valid_squares.append(s5)
        
    # Return as lists of lists of ints
    return [[int(x) for x in row] for row in sq] for sq in valid_squares]
