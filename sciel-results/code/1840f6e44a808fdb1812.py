import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def verify_no_regular_6plus(n, adj):
        """Check no induced subgraph on >=6 vertices is regular.
        adj is an n x n numpy boolean adjacency matrix."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                idx = np.array(combo)
                sub_adj = adj[np.ix_(idx, idx)]
                degrees = sub_adj.sum(axis=1)
                if len(set(degrees.tolist())) == 1:
                    return False
        return True

    def verify_fast(n, adj, k=6):
        """Fast check for size-k regular induced subgraphs using precomputation."""
        # Precompute for each vertex its degree to each other vertex
        for combo in itertools.combinations(range(n), k):
            idx = list(combo)
            degrees = []
            for v in idx:
                d = 0
                for u in idx:
                    if u != v and adj[v][u]:
                        d += 1
                degrees.append(d)
            if all(d == degrees[0] for d in degrees):
                return False
        return True

    def edges_from_adj(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def make_circulant(n, shifts):
        """Create circulant graph: vertex i connected to i+shift mod n for each shift."""
        adj = np.zeros((n, n), dtype=bool)
        for s in shifts:
            for i in range(n):
                j = (i + s) % n
                if i != j:
                    adj[i][j] = True
                    adj[j][i] = True
        return adj

    def make_twin_circulant(n, shifts1, shifts2, fraction=0.5):
        """Mix two circulant graphs."""
        adj1 = make_circulant(n, shifts1)
        adj2 = make_circulant(n, shifts2)
        # Use adj1 where True, adj2 where True, with some mixing
        # Actually just combine: edge exists if in either
        # Better: use a deterministic hash to pick
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                in1 = adj1[i][j]
                in2 = adj2[i][j]
                if in1 and in2:
                    adj[i][j] = True
                    adj[j][i] = True
                elif in1 != in2:
                    # Pick based on a deterministic rule
                    key = i * n + j
                    if key % 2 == 0:
                        adj[i][j] = True
                        adj[j][i] = True
                # else: neither has it, skip
        return adj

    # Strategy 1: Try circulant graphs for n=20
    if time.time() < deadline - 30:
        for n in [20, 19, 18]:
            if time.time() > deadline - 30:
                break
            
            # Try various shift sets for circulant graphs
            # For n=20, try different combinations of shifts
            shift_options = []
            for s in range(1, n // 2 + 1):
                shift_options.append(s)
            
            # Try single-shift circulants first
            for s in shift_options:
                if time.time() > deadline - 30:
                    break
                adj = make_circulant(n, [s])
                if verify_fast(n, adj, 6):
                    if verify_no_regular_6plus(n, adj):
                        return {'n': n, 'edges': edges_from_adj(n, adj)}
            
            # Try two-shift circulants
            for s1 in shift_options:
                for s2 in shift_options:
                    if s1 >= s2:
                        continue
                    if time.time() > deadline - 30:
                        break
                    adj = make_circulant(n, [s1, s2])
                    if verify_fast(n, adj, 6):
                        if verify_no_regular_6plus(n, adj):
                            return {'n': n, 'edges': edges_from_adj(n, adj)}
            
            # Try three-shift circulants (sparse)
            for s1 in range(1, min(4, n // 2 + 1)):
                for s2 in range(s1 + 1, min(6, n // 2 + 1)):
                    for s3 in range(s2 + 1, min(8, n // 2 + 1)):
                        if time.time() > deadline - 30:
                            break
                        adj = make_circulant(n, [s1, s2, s3])
                        if verify_fast(n, adj, 6):
                            if verify_no_regular_6plus(n, adj):
                                return {'n': n, 'edges': edges_from_adj(n, adj)}

    # Strategy 2: Random graph with much faster local search
    if time.time() < deadline - 15:
        for n in [20, 19, 18]:
            if time.time() > deadline - 15:
                break
            
            rng = random.Random(n * 31 + 7)
            
            for attempt in range(30):
                if time.time() > deadline - 15:
                    break
                
                # Initialize random graph
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < 0.45:
                            adj[i][j] = True
                            adj[j][i] = True
                
                # Local search: find violations and flip edges
                for step in range(2000):
                    if time.time() > deadline - 15:
                        break
                    
                    # Find a size-6 regular induced subgraph
                    violation = None
                    for combo in itertools.combinations(range(n), 6):
                        idx = list(combo)
                        degrees = []
                        for v in idx:
                            d = 0
                            for u in idx:
                                if u != v and adj[v][u]:
                                    d += 1
                            degrees.append(d)
                        if all(d == degrees[0] for d in degrees):
                            violation = combo
                            break
                    
                    if violation is None:
                        # Check larger sizes
                        found = False
                        for k in range(7, n + 1):
                            for combo in itertools.combinations(range(n), k):
                                idx = list(combo)
                                degrees = []
                                for v in idx:
                                    d = 0
                                    for u in idx:
                                        if u != v and adj[v][u]:
                                            d += 1
                                    degrees.append(d)
                                if all(d == degrees[0] for d in degrees):
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            # Valid!
                            return {'n': n, 'edges': edges_from_adj(n, adj)}
                        break
                    
                    # Flip a random edge in the violating set
                    u, v = rng.sample(list(violation), 2)
                    adj[u][v] = not adj[u][v]
                    adj[v][u] = adj[u][v]
                    
                    # Occasionally flip a random edge elsewhere
                    if step % 5 == 0:
                        u2 = rng.randint(0, n - 1)
                        v2 = rng.randint(0, n - 1)
                        if u2 != v2:
                            adj[u2][v2] = not adj[u2][v2]
                            adj[v2][u2] = adj[u2][v2]

    # Strategy 3: Greedy edge addition
    if time.time() < deadline - 10:
        for n in [20, 19, 18]:
            if time.time() > deadline - 10:
                break
            
            adj = np.zeros((n, n), dtype=bool)
            edge_order = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng = random.Random(n * 101 + 3)
            rng.shuffle(edge_order)
            
            for (u, v) in edge_order:
                if time.time() > deadline - 10:
                    break
                # Tentatively add
                adj[u][v] = True
                adj[v][u] = True
                # Check if any 6-subset containing u and v is now regular
                bad = False
                for combo in itertools.combinations(range(n), 6):
                    if u in combo and v in combo:
                        idx = list(combo)
                        degrees = []
                        for x in idx:
                            d = 0
                            for y in idx:
                                if x != y and adj[x][y]:
                                    d += 1
                            degrees.append(d)
                        if all(d == degrees[0] for d in degrees):
                            bad = True
                            break
                    if time.time() > deadline - 10:
                        break
                if bad:
                    adj[u][v] = False
                    adj[v][u] = False
            
            if verify_no_regular_6plus(n, adj):
                return {'n': n, 'edges': edges_from_adj(n, adj)}

    # Fallback: small verified graph
    n = 10
    adj = np.zeros((n, n), dtype=bool)
    rng = random.Random(42)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i][j] = True
                adj[j][i] = True
    if verify_no_regular_6plus(n, adj):
        return {'n': n, 'edges': edges_from_adj(n, adj)}

    return {'n': 5, 'edges': []}
