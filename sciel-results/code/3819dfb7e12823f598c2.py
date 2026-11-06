"""Improved program for Ramsey graph R(5,5) task."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _paley_graph(q):
    """Return (n, edges) for the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    residues.discard(0)
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _has_k_clique_or_independent(adj, n, k):
    """Check if graph has a k-clique or k-independent set using backtracking.
    Returns True if found."""
    
    # For k=5, we can check all 5-tuples or use a smarter approach
    # Given n <= ~45, checking all C(n,5) is feasible for small n
    # But for larger n, we need something smarter
    
    if n < k:
        return False
    
    # Use a simple approach: check all subsets of size k
    # For k=5 and n up to ~45, C(45,5) = 1,221,759 which is manageable
    from itertools import combinations
    
    for combo in combinations(range(n), k):
        # Check if it's a clique
        is_clique = True
        is_indep = True
        for idx in range(k):
            for idx2 in range(idx+1, k):
                u, v = combo[idx], combo[idx2]
                if adj[u][v]:
                    is_indep = False
                else:
                    is_clique = False
                if not is_clique and not is_indep:
                    break
            if not is_clique and not is_indep:
                break
        if is_clique or is_indep:
            return True
    return False


def _build_adj(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.uint8)
    for i, j in edges:
        adj[i][j] = 1
        adj[j][i] = 1
    return adj


def _is_valid(adj, n, k=5):
    """Check if graph has no k-clique and no k-independent set."""
    return not _has_k_clique_or_independent(adj, n, k)


def _random_edge_flip(adj, n, k=5, max_flips=100):
    """Try to improve the graph by flipping edges. Returns new adj or None."""
    current_adj = adj.copy()
    best_score = _compute_score(current_adj, n, k)
    
    for _ in range(max_flips):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Try flipping edge (i, j)
        current_adj[i][j] = 1 - current_adj[i][j]
        current_adj[j][i] = current_adj[i][j]
        
        # Check validity
        if not _has_k_clique_or_independent(current_adj, n, k):
            # Valid - check if it's better (more edges is not the goal, just validity)
            # Actually for this task, we just want a valid graph with max n
            # So any valid flip is fine
            return current_adj
        else:
            # Revert
            current_adj[i][j] = 1 - current_adj[i][j]
            current_adj[j][i] = current_adj[i][j]
    
    return None


def _compute_score(adj, n, k=5):
    """Compute a heuristic score for the graph."""
    # Higher score = better. We want to maximize n subject to validity.
    # As a heuristic, count the number of (k-1)-cliques and (k-1)-independent sets
    # to estimate how close we are to violating the constraint.
    # Fewer near-violations = more "room" to add vertices.
    # For now, just return n if valid.
    return n


def _try_add_vertex(adj, n, k=5):
    """Try to add a new vertex to the graph. Returns new adj or None."""
    new_n = n + 1
    new_adj = np.zeros((new_n, new_n), dtype=np.uint8)
    new_adj[:n, :n] = adj
    
    # Try different connection patterns for the new vertex
    # We need the new vertex to not form a k-clique or k-independent set with existing vertices
    
    # For each possible subset of neighbors, check validity
    # This is exponential in n, so we sample
    best_pattern = None
    best_validity = False
    
    # Try random patterns
    for _ in range(100):
        # Randomly decide connections
        for i in range(n):
            new_adj[n][i] = random.randint(0, 1)
            new_adj[i][n] = new_adj[n][i]
        
        if not _has_k_clique_or_independent(new_adj, new_n, k):
            return new_adj
    
    return None


def construct():
    """Main entry point. Returns (n, edges) for a valid (5,5)-free graph."""
    deadline = time.time() + 300.0 - 10.0  # Safety margin
    k = 5
    
    # Start with the Paley graph on 37 vertices
    n, edges = _paley_graph(37)
    adj = _build_adj(n, edges)
    
    # Verify it's valid
    if not _is_valid(adj, n, k):
        # Fallback to a smaller valid graph
        for q in range(36, 20, -1):
            if q % 4 == 1:  # Paley graphs require q ≡ 1 mod 4
                n, edges = _paley_graph(q)
                adj = _build_adj(n, edges)
                if _is_valid(adj, n, k):
                    break
        else:
            # Very small fallback
            return 4, [(0, 1), (1, 2), (2, 3)]
    
    best_n = n
    best_adj = adj.copy()
    
    # Try to extend the graph
    while time.time() < deadline:
        # Try to add a vertex
        new_adj = _try_add_vertex(adj, n, k)
        if new_adj is not None:
            adj = new_adj
            n = n + 1
            best_n = n
            best_adj = adj.copy()
        else:
            # Try edge flips to improve
            new_adj = _random_edge_flip(adj, n, k, max_flips=50)
            if new_adj is not None:
                adj = new_adj
                best_adj = adj.copy()
            else:
                # Try restarting with a different random seed
                break
    
    # Return the best result
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
