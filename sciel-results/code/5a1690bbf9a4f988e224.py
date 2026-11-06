def construct():
    D = {1, 3, 4, 5, 9}
    blocks = []
    
    # 22 blocks with infinity
    for x in range(11):
        b1 = [11] + [(x + d) % 11 for d in D]
        b2 = [11] + [(x - d) % 11 for d in D]
        blocks.append(sorted(b1))
        blocks.append(sorted(b2))
        
    # 110 blocks without infinity
    # Base block: {0, 1, 3, 4, 5, 9}
    base = [0, 1, 3, 4, 5, 9]
    seen = set()
    for m in range(1, 11): # Multiplier
        for t in range(11): # Translation
            b = [(m * x + t) % 11 for x in base]
            b_sorted = tuple(sorted(b))
            if b_sorted not in seen:
                seen.add(b_sorted)
                blocks.append(list(b_sorted))
                
    return blocks
