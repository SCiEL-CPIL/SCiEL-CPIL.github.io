import random
import itertools

def construct():
    """
    Construct a C5-free graph with no independent set of size 9.
    Use random greedy edge addition with backtracking, targeting n=35.
    """
    # Try n from 35 down to 28
    for n in range(35, 27, -1):
        # Multiple random trials with different seeds
        for trial in range(500):
            random.seed(trial * 7919 + n * 31)
            
            # Initialize adjacency as bitmasks for speed
            adj = [0] * n  # adj[i] is a bitmask of neighbors of i
            
            # All possible edges shuffled
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            # Track edges added
            edges = []
            
            # Greedy: add edge if it doesn't create C5
            # C5 check: adding (u,v) creates C5 if there's a path u-a-b-c-v
            # i.e., a in N(u), c in N(v), and N(a) & N(c) != empty
            for u, v in all_edges:
                creates_c5 = False
                # Get neighbors of u and v as sets for intersection
                nu = _bitmask_to_set(adj[u])
                nv = _bitmask_to_set(adj[v])
                
                for a in nu:
                    for c in nv:
                        # Check if a and c have a common neighbor
                        if _common_neighbor_exists(adj, a, c):
                            creates_c5 = True
                            break
                    if creates_c5:
                        break
                
                if not creates_c5:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
            
            # Now check if alpha < 9
            if _alpha_less_than_9(n, adj):
                return {'n': n, 'edges': edges}
        
        # If no valid graph found for this n after all trials, try smaller n
    
    # Fallback: return a known valid small graph
    return _fallback_graph()

def _bitmask_to_set(mask):
    """Convert bitmask to set of indices."""
    s = set()
    i = 0
    while mask:
        if mask & 1:
            s.add(i)
        mask >>= 1
        i += 1
    return s

def _common_neighbor_exists(adj, a, c):
    """Check if vertices a and c have a common neighbor using bitmasks."""
    # Common neighbors = adj[a] & adj[c]
    return (adj[a] & adj[c]) != 0

def _alpha_less_than_9(n, adj):
    """
    Check if the independence number is less than 9.
    Uses branch and bound with greedy coloring bound.
    adj is a list of bitmasks.
    Returns True if alpha < 9, False otherwise.
    """
    # Convert to list of neighbor sets for easier manipulation
    neighbor_sets = [set() for _ in range(n)]
    for i in range(n):
        mask = adj[i]
        j = 0
        while mask:
            if mask & 1:
                neighbor_sets[i].add(j)
            mask >>= 1
            j += 1
    
    # Use a more efficient branch and bound
    # State: candidate set (bitmask), current independent set size
    # We want to find if there's an independent set of size 9
    
    def find_independent_set_k(k, candidates, current_size):
        """
        Try to find independent set of size k from candidates.
        candidates: set of vertex indices
        current_size: size of current independent set
        Returns True if found.
        """
        if current_size + k == 0:
            return True
        if not candidates:
            return False
        if current_size + len(candidates) < k:
            return False
        
        # Greedy coloring bound for upper bound on alpha in candidates
        # This gives an upper bound on how many more vertices we can add
        color_bound = _greedy_coloring_bound(candidates, neighbor_sets)
        if current_size + color_bound < k:
            return False
        
        # Pick a vertex with highest degree in the induced subgraph
        # to maximize pruning
        best_v = None
        best_deg = -1
        for v in candidates:
            deg = len(neighbor_sets[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        # Option 1: Include best_v
        new_candidates = candidates - neighbor_sets[best_v] - {best_v}
        if find_independent_set_k(k - 1, new_candidates, current_size + 1):
            return True
        
        # Option 2: Exclude best_v
        new_candidates = candidates - {best_v}
        if find_independent_set_k(k, new_candidates, current_size):
            return True
        
        return False
    
    return not find_independent_set_k(9, set(range(n)), 0)

def _greedy_coloring_bound(candidates, neighbor_sets):
    """
    Greedy coloring of the subgraph induced by candidates.
    The number of colors is an upper bound on the clique number,
    but for independence, we use: alpha <= n / chi (not quite right).
    
    Actually, for upper bounding alpha, we can use:
    In any coloring with chi colors, one color class has size at least |V|/chi,
    and that color class is an independent set, so alpha >= |V|/chi.
    That's a lower bound, not upper.
    
    For upper bound on alpha, we can use:
    alpha <= sum over colors of (max independent set in that color class)
    But color classes are already independent, so alpha <= |V| trivially.
    
    Better approach: Use the fact that in a graph with maximum degree d,
    alpha >= n/(d+1). But we need upper bound.
    
    Let me use a different pruning: if the number of candidates is small,
    just do exhaustive search.
    
    For now, return len(candidates) as a loose upper bound.
    """
    return len(candidates)

def _fallback_graph():
    """Return a known valid graph with n=28."""
    # Disjoint union of 7 K4's
    edges = []
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': 28, 'edges': edges}
