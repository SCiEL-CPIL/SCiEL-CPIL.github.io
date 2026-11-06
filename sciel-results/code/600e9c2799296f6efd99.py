# EVOLVE-BLOCK-START
def construct():
    edges = []
    # 9 disjoint K4s
    # Vertices 0 to 35
    for i in range(9):
        start = 4 * i
        # K4 on start, start+1, start+2, start+3
        for u in range(4):
            for v in range(u + 1, 4):
                edges.append([start + u, start + v])
    
    # 1 disjoint K7
    # Vertices 36 to 42
    start = 36
    for u in range(7):
        for v in range(u + 1, 7):
            edges.append([start + u, start + v])
            
    return {'n': 43, 'edges': edges}
# EVOLVE-BLOCK-END
