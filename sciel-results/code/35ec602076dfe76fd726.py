N = 40

# EVOLVE-BLOCK-START
def construct():
        edges_set = set()
        edges_list = []
        
        def add_edge(a, b):
            a, b = sorted((a, b))
            if a != b and (a, b) not in edges_set:
                edges_set.add((a, b))
                edges_list.append((a, b))
                
        # Hamiltonian Cycle: 0-1-2-...-39-0
        for i in range(N):
            add_edge(i, (i + 1) % N)
            
        # Add 19 chords to cover all cycle lengths 3..39
        # Chord (0, k) for k in 2..20
        # This creates cycles of length k+1 and 41-k
        for k in range(2, 21):
            add_edge(0, k)
            
        cycles = []
        
        # Cycle of length 40 is the Hamiltonian cycle
        # But we need cycles for 3..40 in order.
        
        # k=3: 0-1-2-0 (Chord 0-2, path 0-1-2)
        # General construction for cycle of length L:
        # If L <= 20:
        #   Use chord (0, L-1).
        #   Cycle: 0 - 1 - ... - (L-1) - 0
        #   Length: (L-1) - 0 + 1 (chord) = L.
        #   Vertices: 0, 1, ..., L-1.
        #   Edges: (0,1), (1,2), ..., (L-2, L-1), (L-1, 0).
        #   (L-1, 0) is the chord (0, L-1).
        #   This works for L=3..20.
        
        # If L > 20:
        #   We need a cycle of length L.
        #   The chord (0, 41-L) creates a cycle of length 41-L and L.
        #   The cycle of length L uses the "long" path.
        #   Path: 0 - 39 - 38 - ... - (41-L) - 0
        #   Length: 39 - (41-L) + 1 (chord) = L.
        #   Vertices: 0, 39, 38, ..., 41-L.
        #   Edges: (0,39), (39,38), ..., (42-L, 41-L), (41-L, 0).
        #   (41-L, 0) is the chord (0, 41-L).
        
        for L in range(3, N + 1):
            if L <= 20:
                # Short cycle
                cyc = list(range(0, L))
            else:
                # Long cycle
                # Vertices from 41-L to 39, then 0
                start = 41 - L
                cyc = list(range(start, N)) + [0]
            
            cycles.append(cyc)
            
        return {'edges': edges_list, 'cycles': cycles}
    # EVOLVE-BLOCK-END
