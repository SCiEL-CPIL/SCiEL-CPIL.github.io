"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    
    # Build adjacency set for O(1) edge lookup
    adj = set()
    for u, v in P['edges']:
        adj.add((min(u, v), max(u, v)))
        adj.add((max(u, v), min(u, v)))
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def make_jump_cycle(step):
        """Create a Hamiltonian cycle using step jumps mod n."""
        if gcd(step, n) != 1:
            return None
        cycle = []
        cur = 0
        for _ in range(n):
            cycle.append(cur)
            cur = (cur + step) % n
        # Verify all edges exist in the host graph
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if (min(u, v), max(u, v)) not in adj:
                return None
        return cycle
    
    # Find valid jump steps that produce Hamiltonian cycles
    valid_steps = []
    for s in range(1, n):
        if gcd(s, n) == 1:
            c = make_jump_cycle(s)
            if c is not None:
                valid_steps.append(s)
    
    # Greedily select edge-disjoint jump cycles
    used_edges = set()
    cycles = []
    for s in valid_steps:
        if time.time() > deadline:
            break
        c = make_jump_cycle(s)
        if c is None:
            continue
        # Check if all edges in this cycle are unused
        ok = True
        for i in range(n):
            u = c[i]
            v = c[(i + 1) % n]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                ok = False
                break
        if ok:
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                used_edges.add((min(u, v), max(u, v)))
            cycles.append(c)
    
    # Verify the result before returning
    if cycles:
        used_edges_final = set()
        valid = True
        for cyc in cycles:
            if len(cyc) != n or len(set(cyc)) != n:
                valid = False
                break
            for i in range(n):
                u = cyc[i]
                v = cyc[(i + 1) % n]
                ek = (min(u, v), max(u, v))
                if ek not in adj:
                    valid = False
                    break
                if ek in used_edges_final:
                    valid = False
                    break
                used_edges_final.add(ek)
            if not valid:
                break
        if valid:
            return [[int(v) for v in c] for c in cycles]
    
    # Fallback: return a single known valid cycle (0,1,2,...,20)
    fallback = list(range(n))
    # Verify fallback
    for i in range(n):
        u = fallback[i]
        v = fallback[(i + 1) % n]
        if (min(u, v), max(u, v)) not in adj:
            # Should not happen, but return empty-safe result
            return [[0, 1, 2]]
    return [fallback]
# EVOLVE-BLOCK-END
