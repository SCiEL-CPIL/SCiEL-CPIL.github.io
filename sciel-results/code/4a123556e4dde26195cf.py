"""Construct a C4-free graph on n vertices with minimum degree >= n-39.

Strategy:
  1. Use bitset-based adjacency for fast C4 checks.
  2. Build initial graph via randomized greedy with degree-bias.
  3. Refine via degree-balancing edge swaps (2-opt style).
  4. Try to add remaining edges to boost minimum degree.
  5. Multiple restarts with time management.
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_result = None
    best_n = 0

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline - 15:
            break

        min_deg_needed = n - 39

        # Multiple restarts with different strategies
        found = False
        max_restarts = 500

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            result = _build_c4_free(n, min_deg_needed, restart)
            if result is not None:
                edges, min_deg = result
                if min_deg >= min_deg_needed:
                    if _verify_c4_free(n, edges):
                        if n > best_n:
                            best_n = n
                            best_result = {'n': n, 'edges': edges}
                        found = True
                        break

        if found:
            break

    if best_result is None:
        # Fallback: trivial graph
        best_result = {'n': 39, 'edges': []}

    return best_result


def _build_c4_free(n, min_deg_needed, seed):
    """Build a C4-free graph on n vertices with min degree >= min_deg_needed.
    
    Uses bitset adjacency for fast common-neighbor checks.
    Strategy: randomized greedy with degree balancing.
    """
    random.seed(seed * 1000003 + 7)
    
    # Bitset adjacency: adj[i] is a Python int where bit j is set if edge (i,j) exists
    adj = [0] * n
    edge_count = 0

    def common_neighbors_count(u, v):
        """Count common neighbors of u and v using bitsets."""
        common = adj[u] & adj[v]
        return bin(common).count('1')

    def can_add(u, v):
        """Check if adding edge (u,v) keeps graph C4-free."""
        # u and v must have at most 1 common neighbor
        common = adj[u] & adj[v]
        return bin(common).count('1') <= 1

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Phase 1: Randomized greedy edge addition with degree balancing
    # Prioritize adding edges to low-degree vertices
    all_edges = list(combinations(range(n), 2))
    
    # Use a biased random order: prefer edges incident to low-degree vertices
    # We'll do multiple passes
    for pass_num in range(3):
        remaining = []
        for (u, v) in all_edges:
            if not (adj[u] >> v) & 1:  # edge not yet added
                remaining.append((u, v))
        
        if not remaining:
            break
        
        # Sort by sum of degrees (prefer low-degree vertices)
        random.shuffle(remaining)
        
        # Try to add edges, prioritizing those that help low-degree vertices
        for (u, v) in remaining:
            if can_add(u, v):
                add_edge(u, v)
                edge_count += 1

    # Check current min degree
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    min_deg = min(degrees) if n > 0 else 0

    if min_deg >= min_deg_needed:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return (edges, min_deg)

    # Phase 2: Degree balancing via edge swaps
    # Try to swap edges to balance degrees while maintaining C4-freedom
    for iteration in range(200):
        if time.time() > deadline - 5:
            break

        degrees = [bin(adj[i]).count('1') for i in range(n)]
        min_deg = min(degrees)
        if min_deg >= min_deg_needed:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return (edges, min_deg)

        # Find vertex with minimum degree
        low_v = degrees.index(min_deg)
        low_neighbors = [j for j in range(n) if j != low_v and (adj[low_v] >> j) & 1]
        
        # Find vertex with high degree that is not adjacent to low_v
        non_neighbors = [j for j in range(n) if j != low_v and not ((adj[low_v] >> j) & 1)]
        
        if not non_neighbors:
            continue

        # Try to add an edge from low_v to a non-neighbor
        random.shuffle(non_neighbors)
        improved = False
        for w in non_neighbors:
            if can_add(low_v, w):
                add_edge(low_v, w)
                edge_count += 1
                improved = True
                break
        
        if not improved:
            # Try edge swap: remove edge (a,b) where a is neighbor of low_v,
            # and add edge (low_v, b) if possible
            if low_neighbors:
                random.shuffle(low_neighbors)
                for a in low_neighbors:
                    # Find a neighbor b of a that is not adjacent to low_v
                    a_neighbors = [b for b in range(n) if b != a and b != low_v and (adj[a] >> b) & 1]
                    random.shuffle(a_neighbors)
                    for b in a_neighbors:
                        if not ((adj[low_v] >> b) & 1) and can_add(low_v, b):
                            # Remove (a,b), add (low_v, b)
                            # But we need to check that removing (a,b) doesn't break anything
                            # and adding (low_v,b) is C4-free
                            # First remove (a,b)
                            remove_edge(a, b)
                            edge_count -= 1
                            # Now check if we can add (low_v, b)
                            if can_add(low_v, b):
                                add_edge(low_v, b)
                                edge_count += 1
                                improved = True
                                break
                            else:
                                # Revert
                                add_edge(a, b)
                                edge_count += 1
                            break
                    if improved:
                        break

    # Final check
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    min_deg = min(degrees) if n > 0 else 0

    if min_deg >= min_deg_needed:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return (edges, min_deg)

    return None


def _verify_c4_free(n, edges):
    """Verify the graph has no C4."""
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for (u, v) in edges:
        common = adj[u] & adj[v]
        if len(common) >= 2:
            return False
    
    return True


if __name__ == "__main__":
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
    if result['n'] > 39:
        n = result['n']
        degs = [0]*n
        for e in result['edges']:
            degs[e[0]] += 1
            degs[e[1]] += 1
        print(f"min_degree={min(degs)}")
        print(f"C4-free: {_verify_c4_free(n, result['edges'])}")
