import time

def construct():
    start_time = time.time()
    deadline = 150.0  # Safety margin from 120s limit? The prompt says 120s limit, evaluator kills at 150s.
    
    # Use the known optimal Paley tournament on 67 vertices
    p = 67
    
    # Check if time is already exceeded (unlikely)
    if time.time() > deadline - 1.0:
        # Fallback to a smaller, definitely valid tournament if we were running a search, 
        # but since this is deterministic, we just return.
        pass

    # Compute quadratic residues modulo p
    # x is a QR if x = y^2 mod p for some y != 0
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Construct orientation
    # i -> j iff (j - i) % p is in squares
    # Note: 0 is not in squares because x != 0.
    # For i < j, diff = (j - i) % p.
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': p, 'orientation': orientation}
