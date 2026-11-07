"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    # The loops above produce Python ints and lists.
    return edges

# EVOLVE-BLOCK-END
