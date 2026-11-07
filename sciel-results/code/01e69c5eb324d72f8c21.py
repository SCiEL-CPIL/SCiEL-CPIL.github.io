import time

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Base square
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    # Find a second square of the form A1[i][j] = (s[i] + j) % n
    # Conditions:
    # 1. s is a permutation of 0..n-1
    # 2. k[i] = (s[i] - i) % n is a permutation of 0..n-1
    
    # Try to find s via backtracking
    s = [0] * n
    k = [0] * n
    used_s = [False] * n
    used_k = [False] * n
    
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        for val in range(n):
            if not used_s[val]:
                k_val = (val - i) % n
                if not used_k[k_val]:
                    s[i] = val
                    k[i] = k_val
                    used_s[val] = True
                    used_k[k_val] = True
                    
                    if backtrack(i + 1):
                        return True
                    
                    used_s[val] = False
                    used_k[k_val] = False
        return False
    
    if backtrack(0):
        A1 = [[(s[i] + j) % n for j in range(n)] for i in range(n)]
        
        # Verify A0 and A1 are Latin squares
        def is_latin(sq):
            for i in range(n):
                row = set(sq[i])
                if len(row) != n:
                    return False
                col = set(sq[r][i] for r in range(n))
                if len(col) != n:
                    return False
            return True
        
        # Verify orthogonality
        def are_orthogonal(sq1, sq2):
            pairs = set()
            for i in range(n):
                for j in range(n):
                    pairs.add((sq1[i][j], sq2[i][j]))
            return len(pairs) == n * n
        
        if is_latin(A0) and is_latin(A1) and are_orthogonal(A0, A1):
            # Ensure all values are Python ints
            A0 = [[int(v) for v in row] for row in A0]
            A1 = [[int(v) for v in row] for row in A1]
            squares.append(A1)
        else:
            # Should not happen, but fallback: return just A0
            squares = [A0]
    else:
        # Backtracking failed, return just A0
        squares = [A0]
    
    # Final type check: ensure all are Python lists of Python ints
    result = []
    for sq in squares:
        result.append([list(row) for row in sq])
    
    return result
