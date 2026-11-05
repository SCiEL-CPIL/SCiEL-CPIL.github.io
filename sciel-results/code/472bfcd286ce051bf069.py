import time

N = 40

def construct():
    start_time = time.time()
    
    # The theoretical minimum number of edges for a pancyclic graph on N=40 is 59.
    # This is derived from the fact that we need to cover 37 cycle lengths (3..39) 
    # with chords in a Hamiltonian cycle. Each chord (0, m) covers lengths m+1 and 41-m.
    # Max coverage per chord is 2. So we need at least ceil(37/2) = 19 chords.
    # Total edges = 40 (Hamiltonian) + 19 (chords) = 59.
    
    edges = set()
    
    # 1. Add base Hamiltonian cycle edges
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Add chords using a greedy set cover approach to ensure minimum count
    # We need to cover lengths 3..39
    target_lengths = set(range(3, N)) # 3..39
    
    # Define all possible chords and the lengths they cover
    # Chord (0, m) covers m+1 and 41-m
    # m ranges from 2 to 38
    # Note: m and 40-m are symmetric in terms of coverage sets?
    # m=2 -> {3, 39}
    # m=38 -> {39, 3}
    # They are the same set.
    
    covered = set()
    selected_chords = []
    
    # To be deterministic and fast, we can just pick the optimal set directly.
    # The optimal set is m in {2, 3, ..., 20}.
    # Let's verify this covers everything.
    # m=2 -> 3, 39
    # m=3 -> 4, 38
    # ...
    # m=19 -> 20, 22
    # m=20 -> 21, 21
    # Union: {3, 4, ..., 20, 21, 22, ..., 39} = {3..39}. Correct.
    
    # We add these 19 chords.
    for m in range(2, 21):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    edge_list = list(edges)
    
    # 3. Generate Cycles
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
