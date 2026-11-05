"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _has_k5(adj, n):
    """Check if graph has a clique of size 5 using backtracking."""
    # Use a simple backtracking approach
    # For efficiency, we can use the fact that if we have a K4, adding any vertex
    # adjacent to all 4 gives a K5.
    
    # Collect all K4s first? That might be expensive. Let's use a different approach.
    # Actually, for n up to ~42, we can try all 5-tuples? No, that's C(42,5) ~ 850k, which is fine.
    # But wait, C(42,5) = 42!/(5!*37!) = (42*41*40*39*38)/(120) = 10,188,012 / 120 = 84,900.
    # 84,900 checks is very fast.
    
    # Let's just check all 5-combinations.
    # Actually, even faster: for each vertex, check if its neighborhood contains a K4.
    # If adj[v] contains a K4, then {v} + that K4 is a K5.
    
    for v in range(n):
        # Get neighbors of v
        neigh = sorted(adj[v])
        m = len(neigh)
        if m < 4:
            continue
        # Check if neigh contains a K4
        # Map neigh to 0..m-1 for easier indexing
        # Or just use indices directly
        # A K4 in neigh means 4 vertices all mutually adjacent
        # Check all 4-tuples in neigh
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    # Check if neigh[i], neigh[j], neigh[k] form a triangle with a 4th?
                    # No, we need a K4. So we need 4 vertices.
                    # Let's just check all 4-combinations of neigh.
                    pass
        
        # Better: check all 4-combinations of neigh for being a clique
        # C(m,4) where m <= n-1. For n=42, m<=41, C(41,4) ~ 100k. 42*100k = 4.2M. Acceptable.
        # But we can prune: only check if the 4 vertices are mutually adjacent.
        
        # Even better: use a recursive check for K4 in the induced subgraph on neigh.
        # But for simplicity and speed in Python, let's use a bitset or just nested loops with pruning.
        
        # Let's use a different strategy: for each 4-set in the whole graph, check if it's a K4,
        # then check if any vertex is adjacent to all 4.
        # Number of 4-sets: C(42,4) ~ 111k. For each, check 4 edges (6 actually) and then check
        # if any of the remaining 38 vertices is adjacent to all 4. 111k * 38 ~ 4.2M. Fine.
        
        pass
    
    # Let's just do the direct 5-combination check. It's simpler and fast enough.
    for combo in itertools.combinations(range(n), 5):
        is_clique = True
        for a, b in itertools.combinations(combo, 2):
            if b not in adj[a]:
                is_clique = False
                break
        if is_clique:
            return True
    return False


def _has_ind5(adj, n):
    """Check if graph has an independent set of size 5."""
    # Independent set in G is a clique in complement.
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    return _has_k5(comp_adj, n)


def _is_valid(adj, n):
    """Check if graph is (5,5)-free."""
    if _has_k5(adj, n):
        return False
    if _has_ind5(adj, n):
        return False
    return True


def _try_expand(n, adj, max_n, time_limit):
    """Try to expand the graph by adding vertices and flipping edges."""
    start_time = time.time()
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # We'll try to add vertices one by one
    current_n = n
    current_adj = [s.copy() for s in adj]
    
    # For each new vertex, we need to decide its connections to existing vertices.
    # A new vertex v can be connected to any subset of existing vertices,
    # as long as no K5 or ind5 is formed.
    
    # Heuristic: try adding a vertex connected to a random subset,
    # or try to maximize connectivity while avoiding violations.
    
    # For efficiency, let's try a few random strategies:
    # 1. Add vertex connected to all (check if valid)
    # 2. Add vertex connected to none (check if valid)
    # 3. Add vertex connected to random half
    
    for _ in range(200):  # Try 200 times to add a vertex
        if current_n >= max_n:
            break
        if time.time() - start_time > time_limit:
            break
        
        # Try adding a new vertex with random connections
        new_vertex = current_n
        new_adj = [set() for _ in range(current_n + 1)]
        
        # Copy existing adjacencies
        for i in range(current_n):
            new_adj[i] = current_adj[i].copy()
        
        # Choose random connections for new vertex
        # Try different densities
        for density in [0.3, 0.5, 0.7]:
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                
                new_adj[new_vertex] = set()
                for i in range(current_n):
                    if random.random() < density:
                        new_adj[new_vertex].add(i)
                        new_adj[i].add(new_vertex)
                
                # Check validity
                if _is_valid(new_adj, current_n + 1):
                    current_n += 1
                    current_adj = new_adj
                    best_n = current_n
                    best_adj = [s.copy() for s in current_adj]
                    break
        
        # Also try edge flipping on current graph
        if current_n < max_n and time.time() - start_time < time_limit:
            # Try flipping a random edge
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j:
                    continue
                
                # Flip edge (i, j)
                if j in current_adj[i]:
                    current_adj[i].remove(j)
                    current_adj[j].remove(i)
                else:
                    current_adj[i].add(j)
                    current_adj[j].add(i)
                
                if _is_valid(current_adj, current_n):
                    best_n = current_n
                    best_adj = [s.copy() for s in current_adj]
                else:
                    # Revert
                    if j in current_adj[i]:
                        current_adj[i].remove(j)
                        current_adj[j].remove(i)
                    else:
                        current_adj[i].add(j)
                        current_adj[j].add(i)
    
    return best_n, best_adj


def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    start_time = time.time()
    time_limit = 300  # Safety margin
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_37_edges()
    adj = _build_adjacency(n, edges)
    
    # Verify it's valid
    if not _is_valid(adj, n):
        # Should not happen, but just in case
        return n, edges
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to expand
    # Use a time budget for the expansion phase
    expansion_time = min(200, time_limit - (time.time() - start_time))
    if expansion_time > 10:
        expanded_n, expanded_adj = _try_expand(n, adj, 46, expansion_time)
        if expanded_n > best_n:
            best_n = expanded_n
            best_adj = expanded_adj
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if j in best_adj[i]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
