"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

# Constants
CLIQUE_SIZE = 5
INDEPENDENT_SIZE = 5
MAX_VERTICES = 46  # Upper bound from Angeltveit and McKay

def construct():
    """
    Construct a graph on as many vertices as possible with no K5 and no independent set of size 5.
    Returns (n, edges) where edges is a list of (i, j) pairs with 0 <= i < j < n.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Start with the Paley graph on 37 vertices
    best_n = 37
    best_edges = _build_paley37()
    
    # Try to extend the Paley graph by adding vertices one at a time
    # using a greedy/local search approach
    best_n, best_edges = _extend_graph(best_n, best_edges, deadline)
    
    # Try random restarts with different initial graphs if time permits
    if time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            n, edges = _random_graph_attempt(deadline)
            if n > best_n:
                best_n = n
                best_edges = edges
    
    return best_n, best_edges


def _build_paley37():
    """Build the Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges


def _build_adjacency(n, edges):
    """Build adjacency list from edge list."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _has_k5(adj, n):
    """Check if graph contains a K5."""
    for clique in itertools.combinations(range(n), 5):
        for i in range(5):
            for j in range(i + 1, 5):
                if clique[j] not in adj[clique[i]]:
                    break
            else:
                continue
            break
        else:
            return True
    return False


def _has_ind5(adj, n):
    """Check if graph contains an independent set of size 5."""
    for indep in itertools.combinations(range(n), 5):
        for i in range(5):
            for j in range(i + 1, 5):
                if indep[j] in adj[indep[i]]:
                    break
            else:
                continue
            break
        else:
            return True
    return False


def _is_valid(adj, n):
    """Check if graph is (5,5)-free."""
    return not _has_k5(adj, n) and not _has_ind5(adj, n)


def _try_add_vertex(adj, n, edges, new_vertex, deadline):
    """
    Try to add a new vertex with a specific set of edges to existing vertices.
    Returns (True, new_adj, new_edges) if valid, (False, None, None) otherwise.
    """
    new_n = n + 1
    new_adj = [set(s) for s in adj]
    new_adj.append(set())
    
    # new_vertex is a set of existing vertices to connect to
    for v in new_vertex:
        new_adj[new_n - 1].add(v)
        new_adj[v].add(new_n - 1)
    
    # Check for K5
    if _has_k5(new_adj, new_n):
        return False, None, None
    
    # Check for independent set of size 5
    if _has_ind5(new_adj, new_n):
        return False, None, None
    
    # Build new edges
    new_edges = list(edges)
    for v in new_vertex:
        if v < new_n - 1:
            new_edges.append((v, new_n - 1))
        else:
            new_edges.append((new_n - 1, v))
    
    return True, new_adj, new_edges


def _extend_graph(n, edges, deadline):
    """
    Try to extend the graph by adding vertices one at a time.
    Uses a greedy approach trying all possible connections for new vertices.
    """
    adj = _build_adjacency(n, edges)
    current_n = n
    current_edges = list(edges)
    
    # Try to add vertices one at a time
    for _ in range(MAX_VERTICES - current_n):
        if time.time() > deadline - 5:
            break
        
        added = False
        # Try all subsets of existing vertices for the new vertex
        # To make this tractable, we use a heuristic: try random subsets
        for attempt in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Random subset of existing vertices
            if current_n <= 20:
                # For small graphs, try more systematically
                for size in range(0, min(current_n + 1, 12)):
                    if time.time() > deadline - 5:
                        break
                    for subset in itertools.combinations(range(current_n), size):
                        valid, new_adj, new_edges = _try_add_vertex(
                            adj, current_n, current_edges, set(subset), deadline
                        )
                        if valid:
                            adj = new_adj
                            current_n += 1
                            current_edges = new_edges
                            added = True
                            break
                    if added:
                        break
            else:
                # For larger graphs, use random sampling
                size = random.randint(0, current_n)
                subset = set(random.sample(range(current_n), size))
                valid, new_adj, new_edges = _try_add_vertex(
                    adj, current_n, current_edges, subset, deadline
                )
                if valid:
                    adj = new_adj
                    current_n += 1
                    current_edges = new_edges
                    added = True
            
            if added:
                break
        
        if not added:
            # Try a more exhaustive search for this vertex
            if current_n <= 25 and time.time() < deadline - 20:
                found = _exhaustive_vertex_search(adj, current_n, current_edges, deadline)
                if found:
                    adj, current_n, current_edges = found
                else:
                    break
            else:
                break
    
    return current_n, current_edges


def _exhaustive_vertex_search(adj, n, edges, deadline):
    """
    Exhaustive search for a valid connection pattern for a new vertex.
    Returns (adj, n, edges) if found, None otherwise.
    """
    # For small n, we can be more exhaustive
    if n > 25:
        return None
    
    # Try all possible subsets
    for size in range(0, min(n + 1, 15)):
        if time.time() > deadline - 5:
            return None
        for subset in itertools.combinations(range(n), size):
            valid, new_adj, new_edges = _try_add_vertex(
                adj, n, edges, set(subset), deadline
            )
            if valid:
                return new_adj, n + 1, new_edges
    
    return None


def _random_graph_attempt(deadline):
    """
    Try to construct a (5,5)-free graph using random initialization and local search.
    Returns (n, edges) if successful, (0, []) otherwise.
    """
    # Start with a random graph and try to optimize
    best_n = 0
    best_edges = []
    
    for n in range(38, MAX_VERTICES + 1):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random graph with edge probability ~0.5
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append((i, j))
            
            adj = _build_adjacency(n, edges)
            
            # Local search: flip edges to eliminate K5 and independent sets
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                has_k5 = _has_k5(adj, n)
                has_ind5 = _has_ind5(adj, n)
                
                if not has_k5 and not has_ind5:
                    best_n = n
                    best_edges = edges
                    break
                
                # Try flipping a random edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                if i > j:
                    i, j = j, i
                
                # Flip edge (i, j)
                if j in adj[i]:
                    adj[i].remove(j)
                    adj[j].remove(i)
                    edges = [e for e in edges if not (e == (i, j) or e == (j, i))]
                else:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append((i, j))
            
            if best_n >= n:
                break
    
    return best_n, best_edges


# EVOLVE-BLOCK-END
