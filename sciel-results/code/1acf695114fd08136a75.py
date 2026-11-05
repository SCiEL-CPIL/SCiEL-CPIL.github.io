import time

N = 40

def construct():
    start_time = time.time()
    
    # The theoretical minimum number of edges for a pancyclic graph on N=40 is 59.
    # Construction:
    # 1. A Hamiltonian cycle C_40: 0-1-2-...-39-0 (40 edges)
    # 2. Chords from vertex 0 to vertices 2, 3, ..., 20 (19 chords)
    #    Each chord (0, m) creates cycles of length m+1 and 41-m.
    #    m=2 -> lengths 3, 39
    #    m=3 -> lengths 4, 38
    #    ...
    #    m=20 -> lengths 21, 21
    #    This covers all lengths 3..39.
    # Total edges = 40 + 19 = 59.
    # Score = 37 / (59 - 40) = 37/19 ~ 1.9474.
    
    edges = set()
    
    # 1. Add base cycle edges
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Add chords
    for m in range(2, 21):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    edge_list = list(edges)
    
    # 3. Generate Cycles
    # We need cycles of length 3, 4, ..., 40.
    # The cycles are determined by the chords.
    # For length k:
    #   If k == 40: use the base cycle [0, 1, ..., 39].
    #   Else, we find a chord (0, m) such that k is in {m+1, 41-m}.
    #   Specifically:
    #   If k <= 21, use chord (0, k-1). The cycle is 0-1-...-(k-1)-0.
    #   If k > 21, use chord (0, 41-k). The cycle is 0-39-...-(41-k)-0.
    
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Base cycle
            cycles.append(list(range(N)))
        elif k <= 21:
            # Cycle formed by path 0->1->...->(k-1) and chord (k-1)->0
            # Vertices: 0, 1, ..., k-1
            cycle_verts = list(range(k))
            cycles.append(cycle_verts)
        else:
            # Cycle formed by path 0->39->...->m and chord m->0
            # where 41-m = k => m = 41-k
            m = 41 - k
            # Vertices: 0, 39, 38, ..., m
            # Note: 39 down to m inclusive
            cycle_verts = [0] + list(range(39, m - 1, -1))
            cycles.append(cycle_verts)
            
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    valid = True
    for c in res['cycles']:
        # Check distinct vertices
        if len(set(c)) != len(c):
            valid = False
            break
        # Check edges
        for i in range(len(c)):
            u = c[i]
            v = c[(i+1)%len(c)]
            if (u,v) not in edges and (v,u) not in edges:
                valid = False
                break
        if not valid:
            break
    print(f"Edges: {len(res['edges'])}, Valid: {valid}")
    if valid:
        print(f"Score: {37 / (len(res['edges']) - 40)}")
