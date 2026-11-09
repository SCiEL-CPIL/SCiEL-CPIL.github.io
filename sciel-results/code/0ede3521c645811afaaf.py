"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

_START = None

def _get_cycle_lengths_fast(graph, n):
    """
    Find all simple cycle lengths in the graph using a more efficient approach.
    graph: list of sets of neighbors
    n: number of vertices
    Returns set of cycle lengths, or None if no cycles found.
    """
    cycle_lengths = set()
    
    # For each starting vertex, do BFS/DFS to find cycles
    # We fix start as the smallest vertex in the cycle to avoid duplicates
    for start in range(n):
        # DFS with path tracking
        visited = {start}
        path = [start]
        
        def dfs(u, depth):
            nonlocal cycle_lengths
            if depth > 1:
                if start in graph[u]:
                    cycle_lengths.add(depth)
            
            for v in graph[u]:
                if v == start:
                    continue  # Don't revisit start in the middle
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    dfs(v, depth + 1)
                    path.pop()
                    visited.remove(v)
        
        dfs(start, 1)
        
        # Early exit: if we've found all possible lengths up to some bound
        # This is a heuristic optimization
    
    return cycle_lengths if cycle_lengths else None

def _check_triangle_free_fast(adj, n):
    """Check if graph is triangle-free using bitset or set operations."""
    for u in range(n):
        for v in adj[u]:
            if v > u:  # Only check each edge once
                # Check if any common neighbor of u and v
                common = adj[u] & adj[v]
                if common:
                    return False
    return True

def _get_degree_from_edges(edges, n):
    """Compute degrees from edge list."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg

def construct():
    global _START
    if _START is None:
        _START = time.time()
    DEADLINE = _START + 80  # Safety margin

    n = 12
    best_edges = None
    best_score = float('inf')
    
    # Fallback: Two K3,3 - gives cycle lengths {4, 6} = 2 distinct lengths
    fallback_edges = []
    for offset in (0, 6):
        A = [offset, offset + 1, offset + 2]
        B = [offset + 3, offset + 4, offset + 5]
        for a in A:
            for b in B:
                fallback_edges.append((a, b))
    
    def generate_random_3_regular(n):
        """Generate a random 3-regular simple graph using configuration model."""
        for _ in range(200):  # Reduced attempts per call, we'll call more often
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            
            if valid:
                # Verify degrees (should be 3 by construction, but check)
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == 3 for d in deg):
                    return list(edges)
        return None

    def evaluate_graph_fast(edges):
        """
        Evaluate a graph: check constraints and return number of distinct cycle lengths.
        Returns float('inf') if invalid.
        Optimized to do cheap checks first.
        """
        if not edges:
            return float('inf')
        
        # Quick degree check
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d < 3 for d in deg):
            return float('inf')
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check (cheap with set intersection)
        if not _check_triangle_free_fast(adj, n):
            return float('inf')
        
        # Get cycle lengths
        lengths = _get_cycle_lengths_fast(adj, n)
        if lengths is None:
            return float('inf')  # Must contain a cycle
        
        return len(lengths)

    def try_bipartite_structure():
        """
        Try to construct a bipartite graph with parts of size 6 each.
        Bipartite graphs are automatically triangle-free.
        We want minimum degree 3 and few distinct cycle lengths.
        
        Strategy: Start with a regular bipartite graph and modify.
        """
        # Try K6,6 first - but that has many cycle lengths
        # Try to find a 3-regular bipartite graph on 12 vertices
        # with few distinct cycle lengths
        
        # One approach: use a structured 3-regular bipartite graph
        # e.g., based on a circulant or other symmetric structure
        
        # Let's try: split into L={0,1,2,3,4,5} and R={6,7,8,9,10,11}
        # Each left vertex connects to 3 right vertices, each right to 3 left.
        
        # Try a specific construction: for left i, connect to right (i+0)%6, (i+1)%6, (i+2)%6
        # But this might create multi-edges or issues
        
        best_local = None
        best_local_score = float('inf')
        
        # Generate random 3-regular bipartite graphs
        for attempt in range(500):
            if time.time() > DEADLINE:
                break
            
            # Random permutation-based construction
            # Each left vertex picks 3 right vertices
            left_to_right = {}
            valid = True
            
            # Use a random matching approach
            # Create a 6x6 matrix where each row has exactly 3 ones
            # and each column has exactly 3 ones
            
            # Start with identity and permute
            perm1 = list(range(6))
            perm2 = list(range(6))
            perm3 = list(range(6))
            random.shuffle(perm1)
            random.shuffle(perm2)
            random.shuffle(perm3)
            
            # Check if we can form a valid 3-regular bipartite graph
            # Each right vertex should appear exactly 3 times across all perms
            right_counts = [0] * 6
            for i in range(6):
                for p in [perm1[i], perm2[i], perm3[i]]:
                    right_counts[p] += 1
            
            if any(c != 3 for c in right_counts):
                continue
            
            # Build edges
            edges = []
            for i in range(6):
                for p in [perm1[i], perm2[i], perm3[i]]:
                    edges.append((i, 6 + p))
            
            # Check for duplicate edges
            if len(edges) != len(set((min(u,v), max(u,v)) for u,v in edges)):
                continue
            
            score = evaluate_graph_fast(edges)
            if score < best_local_score:
                best_local_score = score
                best_local = edges
                if score == 1:
                    return edges, score
        
        return best_local, best_local_score

    # Run main search
    start_time = time.time()
    
    # First, try structured bipartite approach
    bip_edges, bip_score = try_bipartite_structure()
    if bip_edges and bip_score < best_score:
        best_score = bip_score
        best_edges = bip_edges
        if best_score == 1:
            pass  # Can't do better than 1
    
    # Then, run random 3-regular search
    attempts = 0
    while time.time() < DEADLINE and attempts < 10000:
        edges = generate_random_3_regular(n)
        if edges:
            score = evaluate_graph_fast(edges)
            if score < best_score:
                best_score = score
                best_edges = edges
                if best_score == 1:
                    break
        attempts += 1
    
    # Choose best result
    if best_edges and best_score < 2:
        final_edges = best_edges
    else:
        final_edges = fallback_edges

    # Normalize and return
    seen = set()
    out = []
    for u, v in final_edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append([int(u), int(v)])
    out.sort()
    
    return out

# EVOLVE-BLOCK-END
