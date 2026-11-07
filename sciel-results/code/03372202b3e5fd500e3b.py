<<<<<<< SEARCH
def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares = [base]

    # Try to add squares one by one
    for _ in range(8):
        if time.time() > deadline - 10.0:
            break
        nxt = _build_square(squares, deadline)
        if nxt is None:
            break
        squares.append(nxt)

    # Final safety validation
    if _validate(squares):
        # ensure plain Python ints and lists
        out = []
        for L in squares:
            out.append([[int(L[r][c]) for c in range(N)] for r in range(N)])
        return out

    # Fallback: single valid square
    return [[int((i + j) % N) for j in range(N)] for i in range(N)]
=======
def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s
    
    # Use a known valid first square (addition table mod 10)
    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares = [base]
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Try to add squares one by one
    for attempt in range(8):
        if time.time() > deadline - 15.0:
            break
        nxt = _build_square(squares, deadline)
        if nxt is None:
            break
        squares.append(nxt)
    
    # Final safety validation
    if _validate(squares):
        out = []
        for L in squares:
            out.append([[int(L[r][c]) for c in range(N)] for r in range(N)])
        return out

    # Fallback: single valid square
    return [[int((i + j) % N) for j in range(N)] for i in range(N)]
>>>>>>> REPLACE
