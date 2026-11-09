import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the 40-cycle rim (40 edges)
    edges_set = set()
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))

    # Two-hub decomposition:
    # Hub 0 (vertex 0) with chords to vertices 2..13 (12 chords)
    # Hub 20 (vertex 20) with chords to vertices 22..33 (12 chords)
    # Total: 40 + 24 = 64 edges. Objective = 24.
    # This is worse than current, so let's try a different approach.
    
    # Actually, let me try to optimize the current single-hub approach.
    # Current: 58 edges, objective 18.
    
    # Try: Can we reduce by 1 more chord?
    # The missing lengths were 20, 21, 22.
    # Current uses chord (1,21) to cover all three.
    # What if we use a different chord?
    
    # Let me try: hub 0 with chords to 2..17 (16 chords), plus chord (1,20)
    # Chord (0,d) covers d+1 and 41-d for d=2..17: lengths 3..18 and 24..39
    # Missing: 19, 20, 21, 22, 23
    # Chord (1,20): distance 19, covers length 20 directly
    # Need to cover 19, 21, 22, 23
    
    # This is getting complex. Let me stick with the current proven solution
    # and just ensure it's correct.
    
    # Current proven solution: 58 edges
    for d in range(2, 19):
        e = norm_edge(0, d)
        if e not in edges_set:
            edges_set.add(e)
    
    # Add the special chord (1, 21)
    e_special = norm_edge(1, 21)
    if e_special not in edges_set:
        edges_set.add(e_special)

    # Build cycles for lengths 3..40 using the chord structure.
    all_cycles = {N: list(range(N))}  # length 40: full rim

    # Cycles from fan chords (0, d) for d=2..18
    for d in range(2, 19):
        len_fwd = d + 1          # 3..19
        len_bwd = N - d + 1      # 39..23
        if len_fwd not in all_cycles:
            all_cycles[len_fwd] = list(range(0, d + 1))
        if len_bwd not in all_cycles:
            all_cycles[len_bwd] = [0] + list(range(d, N))

    # Cycle for length 21: direct from chord (1, 21)
    # Path: 1 -> 2 -> ... -> 21 -> 1
    all_cycles[21] = list(range(1, 22))

    # Cycle for length 22: composite (0, 2) and (1, 21)
    # Path: 0 -> 1 -> 21 -> 20 -> ... -> 2 -> 0
    all_cycles[22] = [0, 1, 21] + list(range(20, 1, -1))

    # Cycle for length 20: composite (0, 4) and (1, 21)
    # Path: 0 -> 1 -> 21 -> 20 -> ... -> 4 -> 0
    all_cycles[20] = [0, 1, 21] + list(range(20, 3, -1))

    # Verify all 38 cycles (lengths 3..40) are present and valid
    valid = True
    for k in range(3, N + 1):
        cyc = all_cycles.get(k)
        if cyc is None or len(cyc) != k or len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            if norm_edge(cyc[i], cyc[(i + 1) % k]) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if valid:
        return {'edges': [list(e) for e in edges_set],
                'cycles': [all_cycles[k] for k in range(3, N + 1)]}
    else:
        # Fallback: fan graph (77 edges, objective 37)
        fs = set()
        for i in range(1, N):
            fs.add(norm_edge(0, i))
        for i in range(1, N - 1):
            fs.add(norm_edge(i, i + 1))
        return {'edges': [list(e) for e in fs],
                'cycles': [list(range(k)) for k in range(3, N + 1)]}
