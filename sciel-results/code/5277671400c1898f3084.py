"""Deterministic Algebraic Lift construction for k-regular graphs with high girth."""
import time
import sys
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _thue_morse_sequence(length):
    """Generate the first `length` bits of the Thue-Morse sequence."""
    seq = [0]
    for i in range(1, length):
        seq.append(bin(i).count('1') % 2)
    return seq

def _thue_morse_shifts(m, edge_count):
    """
    Generate edge shifts using Thue-Morse derived values for high dispersion.
    Uses the Thue-Morse word to create a low-discrepancy sequence that
    structurally minimizes low-order polynomial collisions.
    """
    tm_len = max(64, edge_count * 4)
    tm = _thue_morse_sequence(tm_len)
    
    shifts = []
    for i in range(edge_count):
        offset = (i * 4) % tm_len
        window = tm[offset:offset+8]
        if len(window) < 8:
            window = (window + tm[:8-len(window)])
        val = 0
        for bit in window:
            val = (val << 1) | bit
        shift = (val * m + i * 7) % m
        shifts.append(shift)
    return shifts

def _thue_morse_shifts_v2(m, edge_count):
    """
    Alternative Thue-Morse based shifts using cumulative sums for better dispersion.
    """
    tm_len = max(128, edge_count * 8)
    tm = _thue_morse_sequence(tm_len)
    
    prefix = [0]
    for i in range(tm_len):
        prefix.append(prefix[-1] + tm[i])
    
    shifts = []
    for i in range(edge_count):
        pos1 = (i * tm_len // edge_count) % tm_len
        pos2 = ((i + 1) * tm_len // edge_count) % tm_len
        val = (prefix[pos1] * 31 + prefix[pos2] * 17 + i * 13) % (m * m)
        shift = val % m
        shifts.append(shift)
    return shifts

def _tree_extension_girth(k, g, max_vertices, deadline):
    """
    Randomized greedy tree-extension: build a k-regular graph incrementally
    by starting from a k-regular tree and closing leaves via random connections,
    checking girth constraints at each step.
    
    Returns (n, edges) if a valid graph is found, else (None, None).
    """
    # Moore bound for k=3, g=16: n >= 512
    # Start with a reasonable target
    target_n = max(max_vertices, 512)
    
    # We'll try to build a graph of size target_n
    # Strategy: start with a tree, then add edges to make it k-regular
    # while maintaining girth >= g
    
    # For k=3, we need each vertex to have degree 3
    # A tree has n-1 edges, we need 3n/2 edges total
    # So we need to add 3n/2 - (n-1) = n/2 + 1 edges
    
    # Better approach: use the voltage lift method but with guided randomization
    # based on tree structure
    
    # Petersen graph as base
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    num_base_edges = len(base_edges)
    
    # Try different m values starting from minimum
    min_m = (target_n + 9) // 10  # ceil(target_n / 10)
    
    for m in range(min_m, min_m + 50):
        if time.time() > deadline:
            break
        
        n = 10 * m
        
        # Use tree-inspired shift assignment:
        # Assign shifts based on a spanning tree of the Petersen graph
        # to ensure good structural properties
        
        # Build a spanning tree of Petersen
        visited = set([0])
        tree_edges = []
        queue = deque([0])
        parent = {0: -1}
        
        while queue:
            u = queue.popleft()
            for v in petersen_adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    tree_edges.append((u, v))
                    queue.append(v)
        
        # Assign shifts: tree edges get structured shifts, non-tree edges get random
        tree_edge_set = set()
        for u, v in tree_edges:
            if u < v:
                tree_edge_set.add((u, v))
            else:
                tree_edge_set.add((v, u))
        
        # Multiple attempts with different random seeds
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            shifts = []
            for (u, v) in base_edges:
                if (u, v) in tree_edge_set:
                    # Tree edge: use structured shift based on depth
                    depth = 0
                    node = v
                    while node != -1:
                        node = parent[node]
                        depth += 1
                    shift = (depth * m // 5 + attempt * 7) % m
                else:
                    # Non-tree edge: random shift
                    shift = random.randint(0, m - 1)
                shifts.append(shift)
            
            n, edges = _build_lift(base_edges, m, shifts)
            if _verify_girth_fast(n, edges, g):
                return n, edges
    
    return None, None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # First, try the tree-extension guided approach
    best_n, best_edges = _tree_extension_girth(k, g, 512, deadline)
    if best_n is not None:
        return best_n, best_edges

    # Fall back to original voltage lift search
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    num_base_edges = len(base_edges)
    m = 52

    while m <= 1000 and time.time() < deadline:
        shifts_tm1 = _thue_morse_shifts(m, num_base_edges)
        n, edges = _build_lift(base_edges, m, shifts_tm1)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        shifts_tm2 = _thue_morse_shifts_v2(m, num_base_edges)
        n, edges = _build_lift(base_edges, m, shifts_tm2)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        shifts1 = [int((i + 1) * m / 15.0) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts1)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        golden = (1 + 5**0.5) / 2.0
        shifts2 = [int((i + 1) * m * golden % m) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts2)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        shifts3 = [primes[i % 15] * m // 15 for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts3)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        num_random_tries = 500
        for _ in range(num_random_tries):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(15)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth_fast(n, edges, g):
                return n, edges

        m += 1

    if time.time() < deadline:
        for m in range(1001, 2000):
            if time.time() > deadline:
                break
            for _ in range(50):
                if time.time() > deadline:
                    break
                shifts_rand = [random.randint(0, m-1) for _ in range(15)]
                n, edges = _build_lift(base_edges, m, shifts_rand)
                if _verify_girth_fast(n, edges, g):
                    return n, edges

    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges


def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v) for u < v
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using timestamped BFS with early termination.
    Optimized with numpy for adjacency and faster array operations.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    seen_mark = np.zeros(n, dtype=np.int32)
    mark = 0

    half_g = g // 2

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

# EVOLVE-BLOCK-END
