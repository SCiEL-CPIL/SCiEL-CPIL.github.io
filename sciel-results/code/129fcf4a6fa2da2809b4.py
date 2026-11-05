import time

N = 40

def construct():
    start_time = time.time()
    
    # Ladder graph + 1 chord
    edges = []
    
    # Horizontal edges on top row: (0,2,...,38)
    for i in range(19):
        edges.append((2*i, 2*i + 2))
        
    # Horizontal edges on bottom row: (1,3,...,39)
    for i in range(19):
        edges.append((2*i + 1, 2*i + 3))
        
    # Vertical rungs
    for i in range(20):
        edges.append((2*i, 2*i + 1))
        
    # Chord to allow odd cycles
    edges.append((1, 2))
    
    # Total edges: 19 + 19 + 20 + 1 = 59
    # Value: 59 - 40 = 19
    # Score: 39 / 19 ~ 2.05
    
    cycles = []
    
    for L in range(3, N + 1):
        if L == 40:
            # Hamiltonian cycle: snake through all vertices
            # 0, 2, ..., 38, 39, 37, ..., 1, 0
            cycle = list(range(0, 40, 2)) + list(range(39, -1, -2))
            cycles.append(cycle)
        elif L % 2 == 0:
            # Even length L = 2k
            # Use rectangle of width k-1
            # Vertices: 0, 2, ..., 2(k-1), 2(k-1)+1, 2(k-2)+1, ..., 1
            # k = L / 2
            k = L // 2
            # Top: 0 to 2(k-1)
            top = list(range(0, 2*k, 2)) # 0, 2, ..., 2k-2
            # Right: 2k-2 to 2k-1
            right = [2*k - 1]
            # Bottom: 2k-1 to 1
            bot = list(range(2*k - 1, -1, -2)) # 2k-1, 2k-3, ..., 1
            cycle = top + right + bot
            # Check length: k + 1 + k = 2k+1?
            # Top has k elements (0..2k-2).
            # Right has 1 element.
            # Bot has k elements (2k-1..1).
            # Total 2k+1.
            # Wait, L=2k.
            # If L=4, k=2.
            # Top: 0, 2. (2 elems)
            # Right: 3. (1 elem)
            # Bot: 3, 1. (2 elems)
            # Cycle: 0, 2, 3, 1.
            # Edges: 0-2, 2-3, 3-1, 1-0.
            # Length 4. Correct.
            # My formula: Top range(0, 4, 2) -> [0, 2].
            # Right: 2*2-1 = 3.
            # Bot range(3, -1, -2) -> [3, 1].
            # Cycle: [0, 2, 3, 1].
            # This works.
            cycles.append(cycle)
        else:
            # Odd length L = 2k + 1
            # Use chord (1,2)
            # Path from 2 to 1 of length 2k in Ladder
            # Cycle: 1, 2, 4, ..., 2k, 2k+1, 2k-1, ..., 3, 1
            # k = (L - 1) / 2
            k = (L - 1) // 2
            if k == 1:
                # L=3
                cycle = [1, 2, 0] # 1-2 (chord), 2-0 (top), 0-1 (vert)
            else:
                # Top part: 2, 4, ..., 2k
                top_part = list(range(2, 2*k + 1, 2))
                # Right part: 2k+1
                right_part = [2*k + 1]
                # Bottom part: 2k-1, ..., 3
                bot_part = list(range(2*k - 1, 1, -2))
                # Cycle: 1, [top_part], [right_part], [bot_part]
                cycle = [1] + top_part + right_part + bot_part
            cycles.append(cycle)
            
    return {'edges': edges, 'cycles': cycles}
