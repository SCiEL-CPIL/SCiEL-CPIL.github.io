import time

def construct():
    # The Clebsch graph is a triangle-free graph on 16 vertices with 40 edges
    # and chromatic number 4. It is the folded 5-cube.
    
    # Map 32 vertices of Q5 to 16 vertices of the folded cube
    # by identifying antipodal vertices (x and x XOR 31)
    
    # Assign indices 0..15 to the equivalence classes
    class_map = {}
    for i in range(32):
        j = i ^ 31
        r = min(i, j)
        if r not in class_map:
            class_map[r] = len(class_map)
            
    # Map each of the 32 nodes to its class representative index
    node_to_class = {}
    for i in range(32):
        j = i ^ 31
        r = min(i, j)
        node_to_class[i] = class_map[r]
        
    edges_set = set()
    for i in range(32):
        u = node_to_class[i]
        for k in range(5):
            j = i ^ (1 << k)
            v = node_to_class[j]
            if u < v:
                edges_set.add((u, v))
                
    edges = [list(e) for e in edges_set]
    return edges
