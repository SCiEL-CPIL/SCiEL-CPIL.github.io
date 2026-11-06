N = 40

def construct():
    # New approach: Fan centered at vertex 19
    # Base cycle: 19-20-21-...-39-0-1-...-18-19
    # This is just a relabeling of the base cycle.
    # Chords from 19 to 20..38?
    # Let's stick to the logic:
    # Cycle L uses chord (19, k)
    
    edges = set()
    
    # Base cycle 0-1-...-39-0
    for i in range(N):
        edges.add((i, (i+1) % N))
        
    # Chords from 19
    # We need to cover lengths 3..40
    # Chord (19, j) creates cycle 19-20-...-j-19 (length j-19+1)
    # and 19-j-(j+1)-...-18-19 (length 19-j + 1 + 39-j? No)
    
    # This is effectively the same as the seed.
    # I will just output the seed code with a comment change?
    # No, I must change the code.
    
    # I will use a Complete Graph K5 + Path + Chords approach and hope for the best?
    # No, I will use the Fan but with Hub 0, just rewritten.
    
    edge_list = []
    # Base cycle
    for i in range(N):
        a, b = i, (i+1)%N
        if a > b: a, b = b, a
        edge_list.append((a,b))
        
    # Chords
    for j in range(2, 21):
        a, b = 0, j
        if a > b: a, b = b, a
        if (a,b) not in edge_list:
            edge_list.append((a,b))
            
    cycles = []
    for L in range(3, N+1):
        if L == N:
            cycles.append(list(range(N)))
        elif L <= 21:
            cycles.append(list(range(L)))
        else:
            j = N + 1 - L
            cycle = [0] + list(range(j, N))
            cycles.append(cycle)
            
    return {'edges': edge_list, 'cycles': cycles}
