import numpy as np

def construct():
    q = 17
    Q = set((i*i) % q for i in range(1, q))
    Q.add(0)
    
    g = np.zeros(q+1, dtype=int)
    for i in Q:
        g[i] = 1
        
    # Construct the circulant matrix
    C = np.zeros((q+1, q+1), dtype=int)
    for i in range(q+1):
        C[i] = np.roll(g, i)
        
    # Find a basis for the row space
    # Use Gaussian elimination
    basis = []
    for row in C:
        # Check if row is in the span of basis
        # We can do this by adding it to the basis and checking rank
        # Or simpler: perform row reduction on the matrix formed by basis + [row]
        new_basis = basis + [row]
        M = np.array(new_basis)
        if M.size == 0:
            continue
        # Rank of M
        rank = np.linalg.matrix_rank(M)
        if rank == len(new_basis):
            basis.append(row)
        else:
            # Row is dependent, skip
            pass
            
    # Now basis has 9 vectors
    # Generate all linear combinations
    codewords = []
    for mask in range(2**len(basis)):
        word = np.zeros(q+1, dtype=int)
        for i in range(len(basis)):
            if (mask >> i) & 1:
                word = (word + basis[i]) % 2
        codewords.append(word)
        
    # Shorten by fixing 0-th coordinate to 0
    shortened_codewords = []
    for word in codewords:
        if word[0] == 0:
            new_word = word[1:]
            val = 0
            for i in range(17):
                if new_word[i] == 1:
                    val |= (1 << i)
            shortened_codewords.append(val)
            
    return shortened_codewords
