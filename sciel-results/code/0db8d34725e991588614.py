"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    base = {tuple(e) for e in P['edges']}
    base.add(tuple(reversed(e)) for e in P['edges'])
    
    n = P['n']
    evens = [i for i in range(n) if i % 2 == 0]
    odds = [i for i in range(n) if i % 2 == 1]
    
    # Hubs
    HE = 12
    HO = 11
    
    added = []
    
    # Connect Evens to HO
    for e in evens:
        if e == HE:
            continue
        edge = (e, HO)
        if edge not in base and (HO, e) not in base:
            added.append([e, HO])
            
    # Connect Odds to HE
    for o in odds:
        if o == HO:
            continue
        edge = (o, HE)
        if edge not in base and (HE, o) not in base:
            added.append([o, HE])
            
    return added
# EVOLVE-BLOCK-END
