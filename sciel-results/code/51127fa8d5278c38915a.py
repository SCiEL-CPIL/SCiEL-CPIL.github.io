import random
import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # New Approach: Cayley Graphs on Z_2 x Z_2 x Z_N
    # Generators are involutions: (delta1, delta2, s) with 2s = 0 mod N.
    # If N is even, s in {0, N/2}.
    # We map the 4 elements of Z_2 x Z_2 to shifts.
    
    start_time = time.time()
    timeout = 10
    
    # Moore bound for k=4, g=10
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    # We want n = 4*N. So N >= moore/4.
    min_N = moore // 4 + 1
    
    # Search N from min_N upwards
    for N in range(min_N, min_N + 50):
        if time.time() - start_time > timeout:
            break
            
        if N % 2 != 0:
            # If N is odd, 2s=0 implies s=0. All generators have s=0.
            # Graph is Z_2 x Z_2 x Z_N. Generators (1,0,0), (0,1,0), (1,1,0), (0,0,0)? No, (0,0,0) is identity.
            # We need 4 distinct generators.
            # If s=0 for all, we only have 3 non-identity elements in Z_2xZ_2.
            # So we can't form 4-regular with involutions if N is odd?
            # Unless we use non-involutions.
            # For simplicity, skip odd N or handle separately.
            # Let's just try even N first.
            continue
            
        half_N = N // 2
        # Possible shifts: 0, half_N
        # We assign a shift to each of the 4 Z_2xZ_2 components.
        # delta: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        # Note: (0,0) with shift 0 is identity, cannot be a generator.
        # So for delta=0, shift must be half_N.
        # For delta=1,2,3, shift can be 0 or half_N.
        
        # There are 2^3 = 8 valid configurations (delta 0 is fixed to half_N to be non-identity involution? 
        # Wait. If delta=0 and shift=0, g=(0,0,0)=e. Invalid.
        # If delta=0 and shift=half_N, g=(0,0,half_N). g^2=(0,0,0). Valid.
        # If delta!=0, shift=0 => g=(d1,d2,0). g^2=(0,0,0). Valid.
        # If delta!=0, shift=half_N => g=(d1,d2,half_N). g^2=(0,0,0). Valid.
        
        for mask in range(8):
            # mask bits 0,1,2 correspond to deltas 1,2,3
            # delta 0 is fixed to shift=half_N
            shifts = [0] * 4
            shifts[0] = half_N
            shifts[1] = half_N if (mask & 1) else 0
            shifts[2] = half_N if (mask & 2) else 0
            shifts[3] = half_N if (mask & 4) else 0
            
            # Check if all 4 generators are distinct
            # g_i = (d1_i, d2_i, shifts[i])
            # d1: 0->0, 1->1, 2->0, 3->1
            # d2: 0->0, 1->0, 2->1, 3->1
            gens = []
            for i in range(4):
                d1 = (i >> 0) & 1
                d2 = (i >> 1) & 1
                gens.append((d1, d2, shifts[i]))
            
            if len(set(gens)) != 4:
                continue
                
            # Build adjacency and check girth
            # Vertex index: (d1, d2, z) -> d1 * 2N + d2 * N + z
            # Total vertices: 4*N
            
            def get_index(d1, d2, z):
                return d1 * 2*N + d2 * N + z
            
            def get_coords(idx):
                d1 = (idx // (2*N)) & 1
                d2 = (idx // N) & 1
                z = idx % N
                return d1, d2, z
            
            # BFS from vertex 0 (0,0,0)
            # We need to check if girth < g.
            # Since graph is vertex transitive (Cayley), checking from 0 is sufficient for girth?
            # Yes.
            
            visited = [-1] * (4*N)
            parent = [-1] * (4*N)
            visited[0] = 0
            parent[0] = -1
            queue = deque([0])
            girth_found = 4*N # infinity
            
            while queue:
                u = queue.popleft()
                du = visited[u]
                if du >= g // 2:
                    continue
                
                d1_u, d2_u, z_u = get_coords(u)
                
                for i in range(4):
                    d1_g, d2_g, s_g = gens[i]
                    d1_v = d1_u ^ d1_g
                    d2_v = d2_u ^ d2_g
                    z_v = (z_u + s_g) % N
                    v = get_index(d1_v, d2_v, z_v)
                    
                    if visited[v] == -1:
                        visited[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        # Cycle length = du + dv + 1
                        cycle_len = du + visited[v] + 1
                        if cycle_len < girth_found:
                            girth_found = cycle_len
                        if girth_found < g:
                            break
                if girth_found < g:
                    break
            
            if girth_found >= g:
                # Construct edges
                edges = set()
                for u in range(4*N):
                    d1_u, d2_u, z_u = get_coords(u)
                    for i in range(4):
                        d1_g, d2_g, s_g = gens[i]
                        d1_v = d1_u ^ d1_g
                        d2_v = d2_u ^ d2_g
                        z_v = (z_u + s_g) % N
                        v = get_index(d1_v, d2_v, z_v)
                        if u < v:
                            edges.add((u, v))
                        else:
                            edges.add((v, u))
                
                return 4*N, list(edges)
    
    # If not found, fallback to LUW
    return luw_construct(k, g)
