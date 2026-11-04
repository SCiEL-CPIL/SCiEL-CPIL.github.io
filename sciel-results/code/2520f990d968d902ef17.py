<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find a better solution by reducing rank
    # First, try the block structure approach
    result = _try_block_structure(n, m, p, deadline)
    if result is not None:
        best = result
    
    # Verify the result
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find a better solution by reducing rank
    result = _try_block_structure(n, m, p, deadline)
    if result is not None:
        best = result
    
    # Verify the result
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best
>>>>>>> REPLACE
