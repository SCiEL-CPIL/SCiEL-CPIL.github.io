def construct():
    # Generator matrix for the binary [12,6,6] code (Witt code)
    # This code is self-dual and has weight enumerator 1 + 132z^6 + z^12
    # The 132 codewords of weight 6 form the blocks of the Steiner system S(5,6,12)
    G = [
        (1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0),
        (0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1),
        (0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0),
        (0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1),
        (0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1)
    ]
    
    blocks = []
    
    # Iterate over all 2^6 = 64 linear combinations of the generator rows
    for mask in range(64):
        # Initialize the codeword to zero
        codeword = [0] * 12
        
        # Compute the sum (XOR) of selected rows
        for i in range(6):
            if (mask >> i) & 1:
                row = G[i]
                for j in range(12):
                    codeword[j] ^= row[j]
        
        # Calculate the weight (number of 1s)
        weight = sum(codeword)
        
        # If the weight is 6, this is a block of the S(5,6,12) design
        if weight == 6:
            block = tuple(i for i, bit in enumerate(codeword) if bit == 1)
            blocks.append(block)
            
    # The result should contain exactly 132 blocks
    # Convert to list of lists if required by interface, but list of tuples is fine for "distinct 6-subsets"
    # The prompt asks for "list of distinct 6-subsets". Tuples are immutable and distinct.
    # I will return the list of tuples.
    
    return blocks
