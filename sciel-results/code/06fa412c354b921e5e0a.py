import time
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0
    q = k
    
    # Choose d so that girth(CD(d,q)) >= g
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
    
    # Precompute field operations
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        Fmul = np.array(table, dtype=np.uint8)
        Fadd = np.zeros((q, q), dtype=np.uint8)
        Fsub = np.zeros((q, q), dtype=np.uint8)
        for i in range(q):
            for j in range(q):
                Fadd[i, j] = i ^ j
                Fsub[i, j] = i ^ j
    else:
        Fmul = np.array([[i*j % q for j in range(q)] for i in range(q)], dtype=np.uint8)
        Fadd = np.array([(i+j) % q for i in range(q) for j in range(q)], dtype=np.uint8).reshape(q, q)
        Fsub = np.array([(i-j) % q for i in range(q) for j in range(q)], dtype=np.uint8).reshape(q, q)
    
    # Build rules: rule[c] = (a, b) meaning y[c] depends on y[a] and x[b]
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d+1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    # Precompute transition tables for each side
    # For side 0 (point -> line): y[c] = Fadd(x[c], Fmul(y[a], x[b]))
    # For side 1 (line -> point): y[c] = Fsub(x[c], Fmul(x[a], y[b]))
    
    # The state is (side, x[1], x[2], ..., x[d])
    # x[0] is unused (None)
    # Total states: 2 * q^d
    
    # For efficiency, we'll do BFS level by level
    # Start state: (0, 0, 0, ..., 0)
    
    # Encode state as integer: side * q^d + x[1]*q^(d-1) + x[2]*q^(d-2) + ... + x[d]*q^0
    total_states = 2 * q ** d
    
    # visited: boolean array of size total_states
    visited = np.zeros(total_states, dtype=bool)
    
    # BFS queue
    from collections import deque
    queue = deque()
    
    start_state = 0  # side=0, all x=0
    visited[start_state] = True
    queue.append(start_state)
    
    # We need to build adjacency. Let's precompute transitions for all states.
    # For each state, compute its q neighbors.
    
    # Actually, let's precompute the full transition table
    # trans[state] = list of neighbor states
    
    # For q=4, d=5: total_states = 2*4^5 = 2048
    # Each state has q=4 neighbors, so 2048*4 = 8192 edges total
    
    # Precompute transitions
    # For a state (side, x1, x2, ..., xd):
    #   For each first in 0..q-1:
    #     y[1] = first
    #     For c in 2..d:
    #       a, b = rule[c]
    #       if side == 0: y[c] = Fadd(x[c], Fmul(y[a], x[b]))
    #       else:         y[c] = Fsub(x[c], Fmul(x[a], y[b]))
    #     neighbor_state = (1-side) * q^d + y[1]*q^(d-1) + ... + y[d]
    
    # Precompute for all states
    trans = [[] for _ in range(total_states)]
    
    for state in range(total_states):
        side = state // (q ** d)
        rem = state % (q ** d)
        # Decode x[1..d]
        x = [0] * (d + 1)  # x[0] unused
        for c in range(d, 0, -1):
            x[c] = rem % q
            rem //= q
        
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = Fadd[x[c], Fmul[y[a], x[b]]]
                else:
                    y[c] = Fsub[x[c], Fmul[x[a], y[b]]]
            
            # Encode neighbor state
            ns = (1 - side) * (q ** d)
            for c in range(1, d + 1):
                ns += y[c] * (q ** (d - c))
            trans[state].append(ns)
    
    # BFS
    n_vertices = 0
    state_to_idx = {}
    state_to_idx[start_state] = 0
    n_vertices = 1
    edges = []
    
    while queue:
        state = queue.popleft()
        idx = state_to_idx[state]
        if time.time() > deadline:
            break
        side = state // (q ** d)
        for ns in trans[state]:
            if ns not in state_to_idx:
                state_to_idx[ns] = n_vertices
                n_vertices += 1
                queue.append(ns)
            # Only add edge from side 0 to avoid duplicates
            if side == 0:
                edges.append((idx, state_to_idx[ns]))
    
    return n_vertices, edges
