"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time

def construct():
    # Theoretical minimum for pancyclic graph on 40 vertices is 59 edges.
    # Construction: Hamiltonian cycle (0-1-...-39-0) + 19 chords from vertex 0.
    # Chords: (0, 2), (0, 3), ..., (0, 20).
    # This covers all cycle lengths 3 to 40.
    
    edges = []
    # Add Hamiltonian cycle edges
    for i in range(N):
        edges.append((i, (i + 1) % N))
        
    # Add chords
    for k in range(2, 21):
        edges.append((0, k))
        
    # Convert to list of lists for output format
    edge_list = [list(e) for e in edges]
    
    # Generate cycles
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        elif length == 21:
            # Chord (0, 20) creates cycle 0-1-...-20-0
            cycles.append(list(range(21)))
        else:
            if length <= 20:
                # Chord (0, length-1) creates cycle 0-1-...-(length-1)-0
                k = length - 1
                cycles.append(list(range(k + 1)))
            else:
                # Chord (0, 41-length) creates the other cycle
                # The cycle goes 0 -> 39 -> ... -> (41-length) -> 0
                k = 41 - length
                # Vertices are 0, 39, 38, ..., k
                # Note: range(N-1, k-1, -1) gives 39 down to k
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
                
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
