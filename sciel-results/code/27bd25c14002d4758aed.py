import time

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    if dimension != 3:
        return [[i, 0, 0] for i in range(N)]
        
    K = 300  # Beam width
    
    start_point = (0, 0, 0)
    # State: (energy, chain, occupied, occupied_h)
    # chain: list of points
    # occupied: set of points
    # occupied_h: set of points that are H
    
    beam = [(0, [start_point], {start_point}, {start_point} if sequence[0]=='H' else set())]
    
    start_time = time.time()
    deadline = start_time + 300
    
    for i in range(1, N):
        candidates = []
        res_is_h = sequence[i] == 'H'
        
        for energy, chain, occupied, occupied_h in beam:
            last_point = chain[-1]
            new_chain_base = chain + [None] # Placeholder
            
            for n in _neighbours(last_point):
                if n in occupied:
                    continue
                
                new_energy = energy
                if res_is_h:
                    # Check contacts
                    # A contact is a neighbor in occupied_h that is not last_point
                    for nn in _neighbours(n):
                        if nn in occupied_h and nn != last_point:
                            new_energy -= 1
                
                # Update sets
                new_occupied = occupied | {n}
                new_occupied_h = occupied_h | ({n} if res_is_h else set())
                
                # Heuristic: Free space
                # Count neighbors of n not in new_occupied
                free_count = 0
                for nn in _neighbours(n):
                    if nn not in new_occupied:
                        free_count += 1
                
                candidates.append((new_energy, -free_count, chain + [n], new_occupied, new_occupied_h))
        
        # Sort by energy, then free space
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        # Keep top K
        beam = []
        for c in candidates[:K]:
            beam.append((c[0], c[2], c[3], c[4]))
            
        if time.time() > deadline:
            break
            
    if not beam:
        return [[i, 0, 0] for i in range(N)]
        
    best_chain = beam[0][1]
    return [list(p) for p in best_chain]
