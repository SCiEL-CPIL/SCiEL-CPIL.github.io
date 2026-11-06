import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    # Deadline guard, though this computation is trivial
    start_time = time.time()
    
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Color is min(i, j) which is i since i < j
            colors[idx] = i
            idx += 1
            
    # Verify validity (optional but good for debugging, should be fast for N=16)
    # Precompute edge indices for fast lookup
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            idx += 1
            
    # Check all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d
    # The three cycles are:
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    import itertools
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Cycle 1: a-b-c-d-a
        c1 = [colors[edge_index[(a,b)]], colors[edge_index[(b,c)]], colors[edge_index[(c,d)]], colors[edge_index[(d,a)]]]
        if len(set(c1)) == 4:
            return [0] * NUM_EDGES # Fallback if invalid, though it shouldn't be
            
        # Cycle 2: a-b-d-c-a
        c2 = [colors[edge_index[(a,b)]], colors[edge_index[(b,d)]], colors[edge_index[(d,c)]], colors[edge_index[(c,a)]]]
        if len(set(c2)) == 4:
            return [0] * NUM_EDGES
            
        # Cycle 3: a-c-b-d-a
        c3 = [colors[edge_index[(a,c)]], colors[edge_index[(c,b)]], colors[edge_index[(b,d)]], colors[edge_index[(d,a)]]]
        if len(set(c3)) == 4:
            return [0] * NUM_EDGES
            
    return colors
