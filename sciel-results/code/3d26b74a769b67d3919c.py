import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin for 60s limit
    
    # 1. Define Edges
    # Rim: (i, (i+1)%N)
    edges = set()
    for i in range(N):
        edges.add(tuple(sorted((i, (i+1) % N))))
        
    # Chords: (0, d) for d in 2..20
    chords = []
    for d in range(2, N//2 + 1):
        e = tuple(sorted((0, d)))
        edges.add(e)
        chords.append(d)
        
    edge_list = list(edges)
    
    # 2. Define Cycles
    cycles = []
    
    for k in range(3, N + 1):
        if k == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        else:
            # Find d such that k is covered
            # Case 1: k = d + 1 => d = k - 1
            # Valid if 2 <= d <= 20
            d1 = k - 1
            if 2 <= d1 <= N//2:
                # Cycle: 0, 1, ..., d1
                cyc = list(range(d1 + 1))
            else:
                # Case 2: k = 41 - d => d = 41 - k
                # Valid if 2 <= d <= 20
                d2 = N + 1 - k
                if 2 <= d2 <= N//2:
                    # Cycle: 0, N-1, N-2, ..., d2
                    # Vertices: 0, then N-1 down to d2
                    cyc = [0] + list(range(N - 1, d2 - 1, -1))
                else:
                    # Should not happen with this chord selection
                    # Fallback: try to find any valid chord
                    found = False
                    for d in chords:
                        if k == d + 1 or k == N + 1 - d:
                            if k == d + 1:
                                cyc = list(range(d + 1))
                            else:
                                cyc = [0] + list(range(N - 1, d - 1, -1))
                            found = True
                            break
                    if not found:
                        # Error case, return invalid to trigger 0 score? 
                        # Or just empty?
                        # Better to raise or return minimal valid?
                        # With d=2..20, all k in 3..39 are covered.
                        pass
                        
        cycles.append(cyc)
        
    # Check time
    if time.time() > deadline:
        # Return best effort (which is already computed)
        pass
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
