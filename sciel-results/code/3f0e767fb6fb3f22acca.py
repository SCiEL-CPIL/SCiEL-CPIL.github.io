"""Sparse pancyclic graphs on 40 vertices."""
import time
import random

N = 40

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _validate(edges, cycles):
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        k = _edge_key(u, v)
        if k in edge_set:
            return False
        edge_set.add(k)
    if len(cycles) != N - 2:
        return False
    for i, cyc in enumerate(cycles):
        k = i + 3
        if len(cyc) != k:
            return False
        if len(set(cyc)) != k:
            return False
        for j in range(k):
            a, b = cyc[j], cyc[(j + 1) % k]
            if _edge_key(a, b) not in edge_set:
                return False
    return True

def _score(edges, cycles):
    return len(edges) - 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    best_edges = None
    best_cycles = None
    best_score = float('inf')

    # Initial seed: fan graph (star + path)
    def seed_fan():
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        return edges, cycles

    edges, cycles = seed_fan()
    if _validate(edges, cycles):
        best_edges = edges[:]
        best_cycles = cycles[:]
        best_score = _score(edges, cycles)

    # Try to improve by finding a sparse pancyclic graph
    # Strategy: start with a known pancyclic structure and try to remove edges
    # A cycle of length 40 contains 40 edges and is pancyclic if we can find
    # cycles of all lengths 3..40 within it. A single 40-cycle is not pancyclic
    # (no triangles), so we need extra edges.

    # Known result: a graph with a Hamiltonian cycle plus a few chords can be pancyclic.
    # Let's try: Hamiltonian cycle 0-1-2-...-39-0, plus chords to create triangles and shorter cycles.

    # Approach: Use a "ladder" or "windmill" like structure.
    # Windmill: triangles sharing a common vertex. But that's not pancyclic up to 40.

    # Better: Start with a 40-cycle, add chords to make it pancyclic.
    # A cycle C_n is pancyclic if n is even? No.
    # Let's try a different approach: build incrementally.

    # We'll use a heuristic search:
    # 1. Start with a base graph known to be pancyclic (e.g., complete graph, too many edges)
    # 2. Or start with a sparse graph and verify pancyclicity, adding edges as needed.

    # Let's try a specific construction:
    # Vertices 0..39.
    # Main cycle: 0-1-2-...-39-0 (40 edges)
    # Add chord 0-2 to create triangle 0-1-2-0 (cycle of length 3)
    # Add chord 0-3 to create cycle 0-1-2-3-0 (length 4)
    # ...
    # Add chord 0-k for k=2,3,...,39 to create cycles 0-1-2-...-k-0 of length k+1
    # This gives cycles of length 3 through 40.
    # Edges: 40 (cycle) + 38 (chords 0-2, 0-3, ..., 0-39) = 78 edges.
    # Score: 78 - 40 = 38.

    # Can we do better? Let's try to share edges more efficiently.

    # Alternative: Use two "hub" vertices.
    # Or: A graph that is the union of a 40-cycle and a star from vertex 0 to all others.
    # That's the same as above.

    # Let's try a more clever construction.
    # Consider a "double fan": two vertices connected to all others.
    # Vertices 0,1 are hubs. All other vertices 2..39 are leaves.
    # Edges: 0-1, 0-i for i=2..39, 1-i for i=2..39. Total: 1 + 38 + 38 = 77 edges.
    # Cycles:
    # Length 3: 0-1-2-0, 0-1-3-0, etc.
    # Length 4: 0-2-1-3-0, 0-2-1-4-0, etc.
    # Length k: 0-v1-v2-...-v_{k-2}-1-0? No, that's not a cycle.
    # 0-v1-1-v2-0 is length 4.
    # 0-v1-1-v2-v3-0 is not a cycle because 1-v2 and v2-v3 and v3-0... wait.
    # In a double fan, the only edges are from hubs to leaves and hub-hub.
    # So a cycle must alternate between hubs and leaves, or use hub-hub.
    # Cycle of length 3: 0-1-2-0 (uses 0-1, 1-2, 2-0)
    # Cycle of length 4: 0-2-1-3-0 (uses 0-2, 2-1, 1-3, 3-0)
    # Cycle of length 5: 0-2-1-3-? No, 3 is a leaf, only connected to 0 and 1.
    # So max cycle length in double fan is 4. Not pancyclic.

    # Let's go back to the single fan (star + path) which is pancyclic.
    # Edges: 39 (star) + 38 (path) = 77 edges. Score: 37.
    # Wait, the seed has 39 + 38 = 77 edges. Score = 77 - 40 = 37.

    # Can we reduce below 77?
    # A pancyclic graph on n vertices must have at least n edges (Hamiltonian cycle).
    # For n=40, min edges = 40. But a single cycle is not pancyclic.
    # We need extra edges to create shorter cycles.

    # Let's try: Hamiltonian cycle + minimal chords to make it pancyclic.
    # Cycle: 0-1-2-...-39-0.
    # To get a triangle, add chord 0-2. Cycle: 0-1-2-0.
    # To get a 4-cycle, add chord 0-3. Cycle: 0-1-2-3-0.
    # ...
    # To get a k-cycle, add chord 0-(k-1). Cycle: 0-1-...-(k-1)-0.
    # This requires chords 0-2, 0-3, ..., 0-39. That's 38 chords.
    # Total edges: 40 + 38 = 78. Score: 38. Worse than fan.

    # The fan is better: 77 edges.

    # Let's try to improve the fan.
    # Fan: star from 0, plus path 1-2-...-39.
    # Edges: (0,i) for i=1..39, (i,i+1) for i=1..38.
    # Total: 39 + 38 = 77.

    # Can we remove some edges and add others to maintain pancyclicity with fewer edges?
    # Unlikely, 77 is already close to the lower bound.

    # Let's just return the seed if we can't find something better.
    # But let's try a quick local search.

    # Actually, let's check if the seed is valid.
    # Cycles in seed: list(range(k)) for k=3..40.
    # Cycle of length k: 0-1-2-...-(k-1)-0.
    # Edges needed: (0,1), (1,2), ..., (k-2,k-1), (k-1,0).
    # In the fan:
    # (0,1) is in star. (1,2), ..., (k-2,k-1) are in path. (k-1,0) is in star.
    # So yes, the seed is valid.

    # Let's try a random search to see if we can find a sparser graph.
    # This is unlikely to succeed quickly, but let's try.

    if time.time() > deadline:
        return {'edges': best_edges, 'cycles': best_cycles}

    # Try a few random perturbations of the seed
    for attempt in range(100):
        if time.time() > deadline:
            break
        e, c = seed_fan()
        # Randomly remove an edge and add another
        if len(e) > 2:
            idx = random.randint(0, len(e) - 1)
            removed = e.pop(idx)
            # Add a random edge
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u != v and (u, v) not in e and (v, u) not in e:
                e.append((u, v))
        if _validate(e, c):
            s = _score(e, c)
            if s < best_score:
                best_score = s
                best_edges = e[:]
                best_cycles = c[:]

    return {'edges': best_edges, 'cycles': best_cycles}
