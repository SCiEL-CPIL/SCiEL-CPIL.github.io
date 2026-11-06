import time

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    # 4 parts of 5 vertices each: P0,P1,P2,P3
    parts = [list(range(5*i, 5*(i+1))) for i in range(4)]
    # colours for the 4 part-pairs
    col = { (0,1): 0, (1,2): 1, (2,3): 2, (3,0): 3 }
    edges = []
    for (a,b),c in col.items():
        for u in parts[a]:
            for v in parts[b]:
                edges.append([u, v, c])
    return edges
