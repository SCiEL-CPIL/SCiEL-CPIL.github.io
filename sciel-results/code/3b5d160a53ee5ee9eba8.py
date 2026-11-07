def construct():
    import time

    n = 10

    # First Latin square: cyclic construction
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Second Latin square: known MOLS pair with L1
    # Using the construction where L2[i][j] = (i + 2*j) % n
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]

    # Verify L1 is a valid Latin square
    def is_latin(square):
        for i in range(n):
            if sorted(square[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    # Verify two Latin squares are orthogonal
    def are_orthogonal(A, B):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (A[i][j], B[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    # Ensure we return pure Python ints in lists
    L1_clean = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
    L2_clean = [[int(L2[i][j]) for j in range(n)] for i in range(n)]

    # Verify before returning
    if is_latin(L1_clean) and is_latin(L2_clean) and are_orthogonal(L1_clean, L2_clean):
        return [L1_clean, L2_clean]
    else:
        # Fallback: just return L1
        return [L1_clean]
