"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START

def _validate(edges, cycles):
    """Validate that all cycle edges exist in the edge set and vertices are distinct."""
    edge_set = set()
    for e in edges:
        u, v = e
        edge_set.add((min(u, v), max(u, v)))
    
    for k, cyc in enumerate(cycles):
        length = k + 3
        if len(cyc) != length:
            return False
        if len(set(cyc)) != length:
            return False
        for i in range(length):
            u = cyc[i]
            v = cyc[(i + 1) % length]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True

def _construct_optimal():
    """
    Construct a sparse pancyclic graph on 40 vertices.
    Uses a 40-cycle and 19 chords from vertex 0 to vertices 2..20.
    Total edges: 59.
    """
    edges = []
    # 40-cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    # Chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.append((0, k))
    
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            cycles.append(list(range(N)))
        elif length == N - 1:
            # Length 39: skip vertex 1, use chord (0,2)
            cycles.append([0] + list(range(2, N)))
        elif length <= 20:
            # Length k: use chord (0, k-1), cycle 0, 1, ..., k-1
            cycles.append(list(range(length)))
        else:
            # Length k > 20: use chord (0, 41-k)
            # Cycle 0, (41-k), (41-k+1), ..., 39
            j = 41 - length
            cycles.append([0] + list(range(j, N)))
            
    return {'edges': edges, 'cycles': cycles}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    result = _construct_optimal()
    
    if _validate(result['edges'], result['cycles']):
        return result
    
    # Fallback: star from 0 + path 1-2-...-39
    # Edges: (0,i) for i=1..39, (i,i+1) for i=1..38 => 39+38=77 edges
    # Cycles: [0,1,...,k-1] for k=3..40
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = [list(range(k)) for k in range(3, N + 1)]
    return {'edges': edges, 'cycles': cycles}

# EVOLVE-BLOCK-END
