N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    cycles = []
    for k in range(3, N + 1):
        cycles.append(list(range(k)))
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
