import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # 1-factorization of K_16 using Z_15 + {15}
    # For k in 0..14:
    #   Edge (15, k) gets color k
    #   For i in 1..7: edge ((k+i)%15, (k-i)%15) gets color k
    
    color_of_edge = {}
    for k in range(15):
        a, b = min(15, k), max(15, k)
        color_of_edge[(a, b)] = k
        for i in range(1, 8):
            x = (k + i) % 15
            y = (k - i) % 15
            a, b = min(x, y), max(x, y)
            color_of_edge[(a, b)] = k
    
    # Verify completeness
    assert len(color_of_edge) == 120
    
    # Verify constraint: every K4 has >= 5 distinct colors
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        colors = set(color_of_edge[e] for e in edges)
        assert len(colors) >= 5, f"K4 {quad} has only {len(colors)} colors: {colors}"
    
    # Return in lexicographic order
    result = []
    for i in range(16):
        for j in range(i + 1, 16):
            result.append(color_of_edge[(i, j)])
    
    return result
