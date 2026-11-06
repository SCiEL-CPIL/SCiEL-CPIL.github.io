"""Construct large (5,5)-free graphs by greedy vertex addition + local search."""
import time
import random
import itertools

_DEADLINE = 330.0
_TIME_MARGIN = 15.0  # seconds of safety margin before the kill


def _build_paley(q):
    """Build Paley graph on q vertices (q must be prime, q ≡ 1 mod 4)."""
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    # Remove 0
    residues.discard(0)
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _edges_to_adj(n, edges):
    """Convert edge list to adjacency sets."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _has_clique_k(adj, n, k=5):
    """Check if graph has a clique of size k using backtracking."""
    # Use Bron-Kerbosch style or simple backtracking
    # For k=5, we can check all 5-tuples but that's O(n^5).
    # Better: for each 4-clique, check if any vertex is adjacent to all 4.
    
    # First find all 4-cliques
    # For n up to ~45, n^4 is manageable (~4M)
    # Actually let's use a smarter approach:
    # For each set of 4 vertices, check if they form a clique, then check 5th.
    
    # Even smarter: for k=5, iterate over all combinations of 5?
    # n=45: C(45,5) = 1,221,759 - that's fine.
    
    # But we want to be fast. Let's use: for each vertex v, 
    # check if its neighborhood contains a 4-clique.
    # C(44,4) per vertex * 45 vertices... too slow.
    
    # Let's just do: for each combination of 5 vertices, check if all pairs are edges.
    # Precompute edge set for O(1) lookup.
    edge_set = set()
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_set.add((i, j))
    
    for combo in itertools.combinations(range(n), k):
        is_clique = True
        for a, b in itertools.combinations(combo, 2):
            if (min(a,b), max(a,b)) not in edge_set:
                is_clique = False
                break
        if is_clique:
            return True
    return False


def _has_independent_k(adj, n, k=5):
    """Check if graph has an independent set of size k."""
    # Independent set in G = clique in complement of G
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    return _has_clique_k(comp_adj, n, k)


def _is_valid(adj, n, k=5):
    """Check no clique of size k and no independent set of size k."""
    if _has_clique_k(adj, n, k):
        return False
    if _has_independent_k(adj, n, k):
        return False
    return True


def _max_clique_size(adj, n):
    """Find the maximum clique size (for k up to 5, just check 5)."""
    return 5 if _has_clique_k(adj, n, 5) else 4


def _try_add_vertex(adj, n, k=5):
    """Try to add a new vertex (vertex n) by choosing edges to maximize validity.
    Returns (new_adj, new_n) if successful, else None."""
    # The new vertex can be connected to any subset of existing vertices.
    # We need: no K5 involving the new vertex, and no independent set of size 5 involving it.
    # 
    # For no K5 with new vertex: the neighbors of new vertex must not contain a K4.
    # For no independent set of size 5 with new vertex: the non-neighbors of new vertex must not contain an independent set of size 4.
    
    # Strategy: try adding vertex with edges to a random subset, check validity.
    # Better: use greedy - for each existing vertex, decide edge or not.
    
    # Actually, let's try a few random configurations and see if any works.
    best_result = None
    
    for attempt in range(50):
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        
        # Randomly decide edges to new vertex
        for i in range(n):
            if random.random() < 0.5:
                new_adj[i].add(n)
                new_adj[n].add(i)
        
        if _is_valid(new_adj, n + 1, k):
            return new_adj, n + 1
    
    return None


def _local_search(adj, n, k=5, max_iter=1000):
    """Try to flip edges to maintain validity while... actually we want to keep n fixed.
    This is more for ensuring validity after modifications."""
    # Not needed for the main approach. Skip.
    return adj, n


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _TIME_MARGIN
    
    # Start with Paley graph on 37 vertices
    n, edges = _build_paley(37)
    adj = _edges_to_adj(n, edges)
    
    # Verify it's valid
    assert _is_valid(adj, n), "Paley graph on 37 should be (5,5)-free"
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph by adding vertices one at a time
    while True:
        if time.time() > deadline - 5:
            break
        
        result = _try_add_vertex(adj, n)
        if result is None:
            # Try multiple random restarts of the addition
            success = False
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                # Try with different random seeds
                old_adj = [set(s) for s in adj]
                result = _try_add_vertex(adj, n)
                if result is not None:
                    adj, n = result
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    success = True
                    break
            if not success:
                break
        else:
            adj, n = result
            best_n = n
            best_adj = [set(s) for s in adj]
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges
