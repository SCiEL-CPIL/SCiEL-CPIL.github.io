"""Sparse pancyclic graphs on 40 vertices.
Strategy: Hamiltonian cycle (40 edges) + 19 chords (0,k) for k=2..20.
Total 59 edges. Each chord (0,k) with k in 2..20 creates cycles of length k+1 and 41-k.
This covers all lengths 3..40 exactly once each.
Value = 59 - 40 = 19.
19 chords is the theoretical minimum for this family (37 lengths / 2 per chord, ceil = 19).
"""
import time

N = 40
_DEADLINE = 80  # seconds, safety margin before 90s kill

def construct():
    t0 = time.time()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Add chords (0, k) for k = 2, 3, ..., 20
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = sorted(edges)
    
    # Build cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if time.time() - t0 > _DEADLINE:
            # Safety: return best effort (should never trigger for this O(1) construction)
            break
        if L == 40:
            cycles.append(list(range(N)))
        elif L <= 21:
            k = L - 1
            cycles.append(list(range(k + 1)))
        else:
            k = 41 - L
            cycles.append([0] + list(range(39, k - 1, -1)))
    
    return {'edges': edge_list, 'cycles': cycles}
