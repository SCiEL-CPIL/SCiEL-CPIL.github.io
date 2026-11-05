# EVOLVE-BLOCK-START
import random

def construct():
    n = 40
    # Hamiltonian cycle edges
    edges = set()
    for i in range(n):
        edges.add((i, (i + 1) % n))
    
    # We need to cover cycle lengths 3 to n-1.
    # A chord (u, v) with distance d (1 < d < n-1) covers lengths d+1 and n-d+1.
    # We need to select a minimum set of chords such that all lengths 3..n-1 are covered.
    
    # Map length to the chord that covers it
    length_to_chord = {}
    
    # Generate all possible chords (i, j) with i < j and not adjacent
    # Distance d = (j - i) % n. We consider d in range 2 to n-2.
    # To avoid duplicates, we can fix i=0? No, chords can be anywhere.
    # But due to symmetry, the "span" d is what matters for coverage.
    # However, the specific vertices matter for the cycle path.
    
    # Let's use a greedy set cover on the lengths.
    uncovered_lengths = set(range(3, n))  # 3 to n-1
    
    chosen_chords = []
    
    # Precompute chords and their coverage
    # To be efficient, we can just iterate.
    # There are ~ n^2/2 chords. n=40, so 780 chords.
    # For each chord, coverage is 2 lengths.
    
    all_chords = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            d = j - i
            if d > n / 2:
                d = n - d
            # Chord (i, j) covers d+1 and n-d+1
            c1 = d + 1
            c2 = n - d + 1
            all_chords.append(((i, j), d, c1, c2))
            
    while uncovered_lengths:
        best_chord = None
        best_cover = set()
        for chord, d, c1, c2 in all_chords:
            if chord in [c[0] for c in chosen_chords]:
                continue
            cover = set()
            if c1 in uncovered_lengths:
                cover.add(c1)
            if c2 in uncovered_lengths:
                cover.add(c2)
            if len(cover) > len(best_cover):
                best_cover = cover
                best_chord = (chord, d, c1, c2)
        
        if best_chord is None:
            break
        
        chosen_chords.append(best_chord)
        uncovered_lengths -= best_cover
        # Record which chord covers which length (first found wins)
        for L in best_cover:
            length_to_chord[L] = best_chord
            
    # Add chosen chords to edges
    for chord, d, c1, c2 in chosen_chords:
        i, j = chord
        edges.add((i, j))
        
    # Construct cycles
    cycles = []
    for L in range(3, n + 1):
        if L == n:
            cycles.append(list(range(n)))
        else:
            chord_data = length_to_chord.get(L)
            if chord_data:
                chord, d, c1, c2 = chord_data
                i, j = chord
                # Determine the path
                # If L == d + 1, the path is from i to j along the short arc (distance d)
                # If L == n - d + 1, the path is from j to i along the long arc? 
                # Actually, the cycle is formed by the chord and the path on the Hamiltonian cycle.
                # Path 1: i -> i+1 -> ... -> j. Length of path is d. Cycle length d+1.
                # Path 2: j -> j+1 -> ... -> i. Length of path is n-d. Cycle length n-d+1.
                
                if L == d + 1:
                    # Path from i to j in forward direction
                    verts = []
                    cur = i
                    for _ in range(d):
                        verts.append(cur)
                        cur = (cur + 1) % n
                    verts.append(j)
                    # Check if this is correct: i, i+1, ..., j
                    # The vertices are i, i+1, ..., i+d-1, i+d=j
                    # So verts should be [i, i+1, ..., j]
                    # My loop:
                    # iter 0: append i, cur=i+1
                    # ...
                    # iter d-1: append i+d-1, cur=i+d=j
                    # append j
                    # So verts = [i, i+1, ..., i+d-1, j].
                    # This is a path of length d.
                    # Cycle is verts + [i]? No, the chord is (i, j).
                    # The cycle is i -> i+1 -> ... -> j -> i.
                    # So the vertex list is [i, i+1, ..., j].
                    cycles.append(verts)
                else: # L == n - d + 1
                    # Path from j to i in forward direction (length n-d)
                    verts = []
                    cur = j
                    for _ in range(n - d):
                        verts.append(cur)
                        cur = (cur + 1) % n
                    verts.append(i)
                    # verts = [j, j+1, ..., i]
                    # Cycle is j -> ... -> i -> j (via chord)
                    cycles.append(verts)
            else:
                # Fallback, should not happen
                cycles.append(list(range(L)))
                
    return {'edges': sorted(edges), 'cycles': cycles}
# EVOLVE-BLOCK-END
