import time
from collections import defaultdict

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Base: Hamiltonian cycle 0-1-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((i, (i + 1) % N))
    
    # All possible chords (non-adjacent pairs)
    all_chords = []
    for i in range(N):
        for j in range(i+2, N):
            if not (i == 0 and j == N-1):  # not the closing edge
                all_chords.append((i, j))
    
    # For each chord, compute which cycle lengths it enables (using base cycle paths)
    # A chord (i,j) creates two fundamental cycles with the base cycle:
    # Path 1: i -> i+1 -> ... -> j (if j > i) or i -> i+1 -> ... -> N-1 -> 0 -> ... -> j
    # Path 2: i -> i-1 -> ... -> j (the other direction)
    
    def cycle_lengths_for_chord(i, j):
        """Return set of cycle lengths created by chord (i,j) with base cycle."""
        lengths = set()
        # Distance from i to j going forward (increasing)
        if i < j:
            d1 = j - i  # path i->i+1->...->j
            d2 = N - d1  # path i->i-1->...->0->N-1->...->j
        else:
            d1 = i - j
            d2 = N - d1
        
        # Cycle 1: chord + path of length d1 => total d1+1
        # Cycle 2: chord + path of length d2 => total d2+1
        if d1 + 1 >= 3 and d1 + 1 <= N:
            lengths.add(d1 + 1)
        if d2 + 1 >= 3 and d2 + 1 <= N:
            lengths.add(d2 + 1)
        return lengths
    
    # Target: cycles of length 3..40
    # Length 40 is already covered by base cycle
    needed = set(range(3, N))  # 3..39
    
    # Greedy: repeatedly pick chord that covers most uncovered lengths
    chords = []
    covered = set()
    remaining = set(needed)
    
    chord_lengths = {}
    for chord in all_chords:
        chord_lengths[chord] = cycle_lengths_for_chord(*chord)
    
    while remaining and time.time() < deadline:
        best_chord = None
        best_gain = 0
        for chord in all_chords:
            if chord in chords:
                continue
            gain = len(chord_lengths[chord] & remaining)
            if gain > best_gain:
                best_gain = gain
                best_chord = chord
        if best_gain == 0:
            break
        chords.append(best_chord)
        covered |= chord_lengths[best_chord]
        remaining -= chord_lengths[best_chord]
    
    # If greedy didn't cover all, fall back to fan construction
    if remaining:
        chords = [(0, k) for k in range(2, N // 2 + 1)]
        covered = set()
        for chord in chords:
            covered |= chord_lengths.get(chord, set())
    
    # Build edge list
    edges = list(base_edges)
    for chord in chords:
        edges.append(chord)
    
    # Generate cycle certificates
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        else:
            # Find a chord that creates a cycle of length L
            found = False
            for chord in chords:
                if L in chord_lengths.get(chord, set()):
                    i, j = chord
                    # Determine which path gives length L
                    if i < j:
                        d1 = j - i
                        d2 = N - d1
                    else:
                        d1 = i - j
                        d2 = N - d1
                    
                    if d1 + 1 == L:
                        # Path i->i+1->...->j, plus chord j->i
                        if i < j:
                            path = list(range(i, j+1))
                        else:
                            path = list(range(i, N)) + list(range(0, j+1))
                        cyc = path
                    elif d2 + 1 == L:
                        # Path i->i-1->...->j, plus chord j->i
                        if i > j:
                            path = list(range(i, j-1, -1))
                        else:
                            path = list(range(i, -1, -1)) + list(range(N-1, j-1, -1))
                        cyc = path
                    found = True
                    break
            if not found:
                # Fallback: use fan-style cycle
                if L <= N // 2 + 1:
                    k = L - 1
                    cyc = list(range(k + 1))
                else:
                    k = N - L + 1
                    cyc = [0] + list(range(k, N))
        
        cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
