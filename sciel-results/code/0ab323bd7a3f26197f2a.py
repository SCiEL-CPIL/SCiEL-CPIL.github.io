def _init_26_structured(field):
    rank = 26
    U = np.zeros((9, rank), dtype=np.int32)
    V = np.zeros((12, rank), dtype=np.int32)
    W = np.zeros((12, rank), dtype=np.int32)
    
    # First 23 columns: Random
    rng = np.random.RandomState(42)
    U[:, :23] = rng.randint(0, field, size=(9, 23))
    V[:, :23] = rng.randint(0, field, size=(12, 23))
    W[:, :23] = rng.randint(0, field, size=(12, 23))
    
    # Last 3 columns: Structured for k=3
    # We want sum_{c=23}^{25} U_{ij,c} V_{j3,c} W_{3i,c} = 1
    # Let U_{ij,c} = 9, V_{j3,c} = 1, W_{3i,c} = 1
    # Sum = 9 * 3 = 27 = 1 mod 13
    for c in range(23, 26):
        U[:, c] = 9
        # V index j*4+3. j=0,1,2 -> 3, 7, 11
        V[3, c] = 1
        V[7, c] = 1
        V[11, c] = 1
        # W index 3*3+i. i=0,1,2 -> 9, 10, 11
        W[9, c] = 1
        W[10, c] = 1
        W[11, c] = 1
        # Ensure other V, W entries for these columns are 0
        # V rows 0,1,2,4,5,6,8,9,10 should be 0
        # W rows 0..8 should be 0
        # They are already 0 from init.
        
    return U, V, W
