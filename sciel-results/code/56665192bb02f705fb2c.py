import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Points: 0..10 are GF(11) elements, 11 is infinity
    # Base block for S(5,6,12)
    # Standard base block: {0, 1, 3, 4, 5, infinity}
    base_block = [0, 1, 3, 4, 5, 11]
    
    blocks = set()
    
    # Generate all transformations in PGL(2,11)
    # Iterate over a,b,c,d in Z_11 such that ad-bc != 0
    # Use canonical representatives to avoid redundant work, 
    # though using a set for blocks handles duplicates anyway.
    # To be efficient, we just iterate all non-singular matrices.
    # There are 13200 such matrices.
    # 13200 * 6 operations is trivial.
    
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    det = (a * d - b * c) % 11
                    if det == 0:
                        continue
                    
                    # Apply transformation x -> (ax+b)/(cx+d)
                    new_block = []
                    for x in base_block:
                        if x == 11: # infinity
                            if c == 0:
                                y = 11
                            else:
                                # a/c
                                y = (a * pow(c, -1, 11)) % 11
                        else:
                            num = (a * x + b) % 11
                            den = (c * x + d) % 11
                            if den == 0:
                                y = 11
                            else:
                                y = (num * pow(den, -1, 11)) % 11
                        new_block.append(y)
                    
                    blocks.add(tuple(sorted(new_block)))
                    
                    if time.time() > deadline:
                        return [list(b) for b in blocks]
                        
    return [list(b) for b in blocks]
