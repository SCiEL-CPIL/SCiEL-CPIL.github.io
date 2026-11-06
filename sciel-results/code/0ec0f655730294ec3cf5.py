BUFFER = None

def verify(net):
    global BUFFER
    if BUFFER is None:
        BUFFER = np.empty_like(INPUTS)
    
    BUFFER[:] = INPUTS
    
    for i, j in net:
        a = BUFFER[:, i]
        b = BUFFER[:, j]
        min_ab = a & b
        max_ab = a | b
        BUFFER[:, i] = min_ab
        BUFFER[:, j] = max_ab
        
    return np.all(BUFFER[:, 1:] >= BUFFER[:, :-1])
