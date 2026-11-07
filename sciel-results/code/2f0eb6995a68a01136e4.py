import random, time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    
    # Check if this is a circulant graph C(n, {1, 2, 5})
    is_circulant = True
    steps = {1, 2, 5}
    for i in range(n):
        expected = set()
        for d in steps:
            expected.add((i + d) % n)
            expected.add((i - d) % n)
        actual = set()
        for u, v in P['edges']:
            if u == i:
                actual.add(v)
            elif v == i:
                actual.add(u)
        if actual != expected:
            is_circulant = False
            break
    
    if is_circulant:
        # For C(21, {1,2,5}), each step d with gcd(d,21)=1 gives a Hamiltonian cycle
        # gcd(1,21)=1, gcd(2,21)=1, gcd(5,21)=1
        # Cycle 1: step +1
        cycle1 = list(range(n))
        # Cycle 2: step +2
        cycle2 = [(i * 2) % n for i in range(n)]
        # Cycle 3: step +5
        cycle3 = [(i * 5) % n for i in range(n)]
        
        all_cycles = [cycle1, cycle2, cycle3]
        used = set()
        valid = True
        for cyc in all_cycles:
            if len(cyc) != n or len(set(cyc)) != n:
                valid = False
                break
            for i in range(n):
                a, b = cyc[i], cyc[(i + 1) % n]
                k = _edge_key(a, b)
                if k not in full_edges:
                    valid = False
                    break
                if k in used:
                    valid = False
                    break
                used.add(k)
            if not valid:
                break
        
        if valid and len(used) == 3 * n:
            return [[int(v) for v in cyc] for cyc in all_cycles]
    
    # Fallback: search-based approach
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_hamiltonian_cycle(adj_cur, start, deadline):
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        
        def dfs():
            if time.time() > deadline:
                return False
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            u = path[-1]
            nbrs = list(adj_cur[u])
            nbrs.sort(key=lambda w: sum(1 for x in adj_cur[w] if not visited[x]))
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False
        
        if dfs():
            return list(path)
        return None
    
    def validate_cycles(cycles):
        used = set()
        for cyc in cycles:
            if len(cyc) < 3 or len(set(cyc)) != len(cyc):
                return False
            if any(v < 0 or v >= n for v in cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges or k in used:
                    return False
                used.add(k)
        return True
    
    best = None
    best_count = 0
    attempt = 0
    
    while time.time() < deadline - 2.0:
        attempt += 1
        rng = random.Random(attempt * 7919 + 13)
        adj_cur = [set(s) for s in adj]
        cycles = []
        for c in range(3):
            if time.time() > deadline - 1.0:
                break
            found = None
            for trial in range(10):
                start = rng.randrange(n)
                cyc = find_hamiltonian_cycle(adj_cur, start, deadline - 1.0)
                if cyc is not None:
                    found = cyc
                    break
            if found is None:
                break
            cycles.append(found)
            for i in range(len(found)):
                a, b = found[i], found[(i + 1) % len(found)]
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)
            if any(len(adj_cur[v]) < 2 for v in range(n)):
                break
        
        if cycles and validate_cycles(cycles) and len(cycles) > best_count:
            best_count = len(cycles)
            best = [[int(v) for v in cyc] for cyc in cycles]
            if best_count >= 3:
                return best
    
    if best:
        return best
    
    adj_cur = [set(s) for s in adj]
    for start in range(n):
        cyc = find_hamiltonian_cycle(adj_cur, start, deadline - 0.1)
        if cyc and validate_cycles([cyc]):
            return [[int(v) for v in cyc]]
    
    return [[int(v) for v in range(n)]]
