def construct():
    points = []
    for y in range(6):
        for x in range(6):
            # Define the exclusion set
            # Corners: (0,0), (5,0), (0,5), (5,5)
            # Side midpoints (approx): (0,2), (5,2)
            if (x == 0 and y == 0) or (x == 5 and y == 0) or (x == 0 and y == 5) or (x == 5 and y == 5):
                continue
            if (x == 0 and y == 2) or (x == 5 and y == 2):
                continue
            points.append((x, y))
            
    # Verify count
    # y=0: x in {1,2,3,4} -> 4
    # y=1: x in {0,1,2,3,4,5} -> 6
    # y=2: x in {1,2,3,4} -> 4 (0,5 removed)
    # y=3: x in {0,1,2,3,4,5} -> 6
    # y=4: x in {0,1,2,3,4,5} -> 6
    # y=5: x in {1,2,3,4} -> 4
    # Total 30.
    
    return points
