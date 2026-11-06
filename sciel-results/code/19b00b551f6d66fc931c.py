"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a (5,5)-free graph. We start with the Paley graph on 37 vertices
    (known to be (5,5)-free) and attempt to add a 38th vertex via local search.
    If we cannot add a 38th vertex while maintaining the property, we return
    the best valid graph found (at least the 37-vertex seed).
    
    Returns: (n, edges)
    """
    deadline = time.time() + 290  # Safety margin for 300s limit
    q = 37
    
    # --- Step 1: Construct Paley Graph on 37 vertices ---
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def is_valid(adj, n):
        """Check if graph has no K5 and no independent set of size 5."""
        # Check for K5
        for clique in combinations(range(n), 5):
            valid = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = clique[i], clique[j]
                    if not (adj[u] & (1 << v)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return False
        # Check for Independent Set of size 5
        for ind_set in combinations(range(n), 5):
            valid = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = ind_set[i], ind_set[j]
                    if (adj[u] & (1 << v)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return False
        return True

    def check_clique_5_adjacent_to(new_v, adj, n):
        """Check if adding new_v creates a K5.
        A K5 involving new_v must contain new_v and 4 other vertices that form a K4 
        and are all connected to new_v."""
        neighbors = [v for v in range(n) if (adj[new_v] & (1 << v))]
        # Check if any 4 neighbors form a K4
        for quad in combinations(neighbors, 4):
            valid = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = quad[i], quad[j]
                    if not (adj[u] & (1 << v)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return True
        return False

    def check_indep_5_adjacent_to(new_v, adj, n):
        """Check if adding new_v creates an independent set of size 5.
        An independent set of size 5 involving new_v must contain new_v and 4 other 
        vertices that are non-neighbors of each other and of new_v."""
        non_neighbors = [v for v in range(n) if not (adj[new_v] & (1 << v))]
        # Check if any 4 non-neighbors form an independent set (K4 in complement)
        for quad in combinations(non_neighbors, 4):
            valid = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = quad[i], quad[j]
                    if (adj[u] & (1 << v)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return True
        return False

    def try_add_vertex(adj, n):
        """Attempt to add a new vertex (n) by finding a suitable adjacency pattern."""
        # The new vertex needs to avoid completing any K4 in its neighborhood
        # and avoid completing any independent K4 in its non-neighborhood.
        
        # For each existing vertex v, we decide if (new_v, v) is an edge.
        # This is a constraint satisfaction problem.
        
        # Heuristic: Randomly assign adjacency and check constraints.
        # Or: Try to set adjacency based on avoiding specific structures.
        
        # Let's try a simple randomized search for a few iterations.
        for _ in range(500):
            # Randomly initialize adjacency for new vertex
            new_adj = 0
            for v in range(n):
                if random.random() < 0.5:
                    new_adj |= (1 << v)
            
            # Check validity
            if check_clique_5_adjacent_to(n, [adj[i] for i in range(n)] + [new_adj], n):
                continue
            if check_indep_5_adjacent_to(n, [adj[i] for i in range(n)] + [new_adj], n):
                continue
            
            # Also need to ensure that the existing graph remains valid (it does)
            # And that the new vertex doesn't create other issues?
            # The checks above cover K5 and Ind5 involving the new vertex.
            
            # If valid, apply
            new_adj_list = adj + [new_adj]
            for v in range(n):
                if new_adj & (1 << v):
                    new_adj_list[v] |= (1 << n)
            
            return new_adj_list, n + 1
        
        return None

    # Initial check: Is the Paley graph valid? (It should be)
    if not is_valid(adj, q):
        # This shouldn't happen, but fallback
        return 0, []

    best_adj = adj[:]
    best_n = q

    # Try to add vertex 38
    result = try_add_vertex(best_adj, best_n)
    if result:
        new_adj, new_n = result
        # Verify the whole graph
        if is_valid(new_adj, new_n):
            best_adj = new_adj
            best_n = new_n

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append((i, j))

    return best_n, edges

# EVOLVE-BLOCK-END
