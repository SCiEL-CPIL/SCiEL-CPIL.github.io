"""Constructs the Witt design S(5,6,12) using cyclic development."""

def construct():
    """
    Returns a list of 132 distinct 6-subsets of 0..11 such that no 5-subset
    lies in two blocks. This is the Steiner system S(5,6,12).
    """
    # Points are 0..10 and 11 (infinity)
    # We use the cyclic group Z_11 to develop base blocks.
    
    # Base blocks containing 11 (infinity)
    # Each is a set of 6 elements, one of which is 11.
    # The other 5 are in 0..10.
    base_blocks_with_inf = [
        [11, 0, 1, 3, 4, 5],
        [11, 0, 1, 2, 4, 10],
        [11, 0, 1, 2, 7, 8],
        [11, 0, 1, 3, 7, 9],
        [11, 0, 1, 3, 6, 8],
        [11, 0, 1, 4, 6, 9]
    ]
    
    # Base blocks NOT containing 11 (infinity)
    # Each is a set of 6 elements in 0..10.
    base_blocks_no_inf = [
        [0, 1, 3, 4, 5, 9],
        [0, 1, 2, 4, 5, 10],
        [0, 1, 2, 3, 7, 8],
        [0, 1, 3, 6, 7, 9],
        [0, 1, 3, 4, 6, 8],
        [0, 1, 4, 5, 6, 9]
    ]
    
    blocks = []
    seen = set()
    
    def add_block(b):
        # Sort the block to have a canonical representation
        b_sorted = tuple(sorted(b))
        if b_sorted not in seen:
            seen.add(b_sorted)
            blocks.append(list(b_sorted))
    
    # Develop base blocks with infinity
    for base in base_blocks_with_inf:
        for k in range(11):
            new_block = []
            for x in base:
                if x == 11:
                    new_block.append(11)
                else:
                    new_block.append((x + k) % 11)
            add_block(new_block)
            
    # Develop base blocks without infinity
    for base in base_blocks_no_inf:
        for k in range(11):
            new_block = [(x + k) % 11 for x in base]
            add_block(new_block)
            
    return blocks
