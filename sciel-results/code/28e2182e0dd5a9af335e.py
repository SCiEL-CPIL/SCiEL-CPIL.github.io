"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import time

def _has_k_clique(adj, n, k):
    """Check if graph has a clique of size k using backtracking."""
    if k <= 1:
        return True
    if n < k:
        return False
    
    # Sort vertices by degree (descending) for better pruning
    def find_clique(current, candidates):
        if len(current) == k:
            return True
        if not candidates:
            return False
        
        for i, v in enumerate(candidates):
            # Check if v is connected to all in current
            if all(v in adj[u] for u in current):
                new_current = current | {v}
                # New candidates must be neighbors of v and after v in original ordering
                new_candidates = [u for u in candidates[i+1:] if u in adj[v]]
                if find_clique(new_current, new_candidates):
                    return True
        return False
    
    all_vertices = list(range(n))
    return find_clique(set(), all_vertices)

def _has_independent_k(adj, n, k):
    """Check if graph has an independent set of size k."""
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n):
    """Check if graph is (5,5)-free: no K5 and no independent set of size 5."""
    if _has_k_clique(adj, n, 5):
        return False
    if _has_independent_k(adj, n, 5):
        return False
    return True

def _add_vertex(adj, n, new_vertex_adj):
    """Add a new vertex with given adjacency to existing vertices."""
    new_adj = [set(s) for s in adj]
    new_adj.append(set())
    for i in range(n):
        if i in new_vertex_adj:
            new_adj[i].add(n)
            new_adj[n].add(i)
    return new_adj

def _schlaefli_graph():
    """
    Construct the Schläfli graph on 27 vertices.
    The Schläfli graph is the complement of the line graph of K6.
    Vertices are 2-subsets of {0,1,2,3,4,5}.
    Two vertices are adjacent if the corresponding 2-subsets are disjoint.
    """
    vertices = list(itertools.combinations(range(6), 2))
    n = len(vertices)  # 27
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    adj = [set() for _ in range(n)]
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if i < j:
                # Check if v1 and v2 are disjoint
                if not set(v1) & set(v2):
                    adj[i].add(j)
                    adj[j].add(i)
    
    return adj, n

def _try_add_vertex_with_neighborhood(adj, n, neighborhood):
    """
    Try to add a new vertex with the given neighborhood.
    Returns True if successful, and updates adj and n in place via return.
    """
    # First check: neighborhood must be (4,4)-free
    # No K4 in neighborhood
    has_k4 = False
    for combo in itertools.combinations(neighborhood, 4):
        a, b, c, d = combo
        if (b in adj[a] and c in adj[a] and d in adj[a] and
            c in adj[b] and d in adj[b] and d in adj[c]):
            has_k4 = True
            break
    if has_k4:
        return False, None, n
    
    # No independent set of size 4 in neighborhood
    has_ind_4 = False
    for combo in itertools.combinations(neighborhood, 4):
        a, b, c, d = combo
        if (b not in adj[a] and c not in adj[a] and d not in adj[a] and
            c not in adj[b] and d not in adj[b] and d not in adj[c]):
            has_ind_4 = True
            break
    if has_ind_4:
        return False, None, n
    
    # Try adding the vertex
    new_adj = _add_vertex(adj, n, neighborhood)
    new_n = n + 1
    
    # Full validation
    if _is_valid(new_adj, new_n):
        return True, new_adj, new_n
    
    return False, None, n

def construct():
    # Start with Schläfli graph on 27 vertices
    adj, n = _schlaefli_graph()
    
    # Verify initial graph is valid
    if not _is_valid(adj, n):
        # Fallback to Paley if something goes wrong
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        adj = [set() for _ in range(q)]
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
        n = q
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    random.seed(42)
    start_time = time.time()
    time_limit = 280  # Leave some margin for overhead
    
    # Strategy 1: Try to add vertices by finding valid neighborhoods
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple times to find a valid neighborhood
        found = False
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate neighborhood using different strategies
            strategy = random.random()
            
            if strategy < 0.3:
                # Strategy A: Random subset
                subset_size = random.randint(max(5, n // 4), min(n - 5, n // 2))
                neighborhood = set(random.sample(range(n), subset_size))
            elif strategy < 0.6:
                # Strategy B: Neighborhood of a random vertex (modified)
                v = random.randint(0, n - 1)
                if len(adj[v]) >= 5:
                    neighborhood = set(random.sample(list(adj[v]), random.randint(5, min(len(adj[v]), n // 2))))
                else:
                    neighborhood = set(random.sample(range(n), random.randint(5, min(n - 5, n // 2))))
            elif strategy < 0.8:
                # Strategy C: Non-neighborhood of a random vertex (modified)
                v = random.randint(0, n - 1)
                non_neighbors = set(range(n)) - adj[v] - {v}
                if len(non_neighbors) >= 5:
                    neighborhood = set(random.sample(list(non_neighbors), random.randint(5, min(len(non_neighbors), n // 2))))
                else:
                    neighborhood = set(random.sample(range(n), random.randint(5, min(n - 5, n // 2))))
            else:
                # Strategy D: Union of two random vertices' neighborhoods
                v1 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                combined = (adj[v1] | adj[v2]) - {v1, v2}
                if len(combined) >= 5:
                    subset_size = random.randint(5, min(len(combined), n // 2))
                    neighborhood = set(random.sample(list(combined), subset_size))
                else:
                    neighborhood = set(random.sample(range(n), random.randint(5, min(n - 5, n // 2))))
            
            success, new_adj, new_n = _try_add_vertex_with_neighborhood(adj, n, neighborhood)
            
            if success:
                adj = new_adj
                n = new_n
                best_n = n
                best_adj = [set(s) for s in adj]
                found = True
                break
        
        if not found:
            # Try local search: modify existing graph structure
            # Pick a random vertex and try to adjust its connections
            if n > 27 and random.random() < 0.5:
                v = random.randint(0, n - 1)
                # Try adding/removing a few edges
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    
                    u = random.randint(0, n - 1)
                    if u == v:
                        continue
                    
                    trial_adj = [set(s) for s in adj]
                    if u in trial_adj[v]:
                        # Try removing edge
                        trial_adj[v].remove(u)
                        trial_adj[u].remove(v)
                    else:
                        # Try adding edge
                        trial_adj[v].add(u)
                        trial_adj[u].add(v)
                    
                    if _is_valid(trial_adj, n):
                        adj = trial_adj
                        if n > best_n:
                            best_n = n
                            best_adj = [set(s) for s in adj]
                        break
    
    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges
# EVOLVE-BLOCK-END
