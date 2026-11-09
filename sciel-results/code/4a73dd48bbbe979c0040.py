"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    chords_d = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    chords = [(0, d) for d in chords_d]
    
    edges = cycle_edges + chords
    
    cycles = []
    for length in range(3, N + 1):
        if length == 40:
            cyc = list(range(40))
        elif length in (4, 5, 8, 10, 20):
            stride = 40 // length
            cyc = [(i * stride) % 40 for i in range(length)]
        else:
            d1 = length - 1
            d2 = 41 - length
            if d1 in chords_d:
                cyc = list(range(d1 + 1))
            elif d2 in chords_d:
                cyc = [0] + list(range(39, d2 - 1, -1))
            else:
                raise ValueError(f"Cannot construct cycle of length {length}")
        cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
