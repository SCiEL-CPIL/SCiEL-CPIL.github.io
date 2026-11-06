"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 401
    
    # Start with K20,20 plus one edge
    base_edges = set()
    for a in range(20):
        for b in range(20, 40):
            base_edges.add((a, b))
    base_edges.add((0, 1))
    
    def compute_max_book(edge_set):
        """Compute the largest book size for a given edge set."""
        # Build adjacency sets for fast lookup
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, count common neighbors (triangles containing that edge)
        max_book = 0
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    def edge_key(e):
        return (min(e[0], e[1]), max(e[0], e[1]))
    
    def get_all_possible_edges():
        """Generate all possible edges not in the current set."""
        current = set()
        for e in edge_set:
            current.add(edge_key(e))
        
        possible = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in current:
                    possible.append((i, j))
        return possible
    
    # Initial solution
    edge_set = set(base_edges)
    current_score = compute_max_book(edge_set)
    best_edge_set = set(edge_set)
    best_score = current_score
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    safety_margin = 10.0
    
    # Local search: try to swap edges to reduce max book
    random.seed(42)
    
    def try_improve(edge_set, current_score, deadline):
        """Attempt to improve the current solution via edge swaps."""
        nonlocal best_edge_set, best_score
        
        # Build adjacency for current set
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find edges with high book sizes
        high_book_edges = []
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            if common >= current_score:
                high_book_edges.append((u, v, common))
        
        if not high_book_edges:
            return edge_set, current_score
        
        # Try removing a high-book edge and adding a different one
        for u, v, book_size in high_book_edges:
            if time.time() > deadline - safety_margin:
                break
            
            # Try to remove (u, v) and add some other edge
            # We want to add an edge that doesn't create too many new triangles
            candidate_removals = [(u, v)]
            
            for rem_u, rem_v in candidate_removals:
                # Remove the edge
                new_edge_set = set(edge_set)
                new_edge_set.discard((rem_u, rem_v))
                new_edge_set.discard((rem_v, rem_u))
                
                if len(new_edge_set) != len(edge_set) - 1:
                    continue
                
                # Try adding various edges
                # Prefer edges within the same partition or between partitions with low common neighbors
                added = False
                for i in range(n):
                    if time.time() > deadline - safety_margin:
                        break
                    for j in range(i + 1, n):
                        if (i, j) in new_edge_set or (j, i) in new_edge_set:
                            continue
                        
                        # Estimate new book size for this edge
                        # Count common neighbors in new_edge_set
                        common = 0
                        for k in range(n):
                            if k != i and k != j:
                                if (i, k) in new_edge_set or (k, i) in new_edge_set:
                                    if (j, k) in new_edge_set or (k, j) in new_edge_set:
                                        common += 1
                        
                        # Check if this would reduce max book
                        # Also need to check if removing (rem_u, rem_v) reduces any other book
                        # This is complex, so we'll just compute the new max book
                        if common < current_score:
                            # Promising candidate, compute full score
                            new_edge_set.add((i, j))
                            new_score = compute_max_book(new_edge_set)
                            
                            if new_score < best_score:
                                best_score = new_score
                                best_edge_set = set(new_edge_set)
                                edge_set = set(new_edge_set)
                                current_score = new_score
                            added = True
                            break
                    if added:
                        break
                if added:
                    break
        
        return edge_set, current_score
    
    # Iterative improvement
    iterations = 0
    while time.time() < deadline - safety_margin:
        iterations += 1
        edge_set, current_score = try_improve(edge_set, current_score, deadline - safety_margin)
        
        # Occasionally try random restarts with small perturbations
        if iterations % 50 == 0:
            # Try a random perturbation
            if len(edge_set) > 0:
                # Remove a random edge
                random_edge = random.choice(list(edge_set))
                perturbed = set(edge_set)
                perturbed.discard(random_edge)
                
                # Add a random edge
                possible = get_all_possible_edges()
                if possible:
                    new_edge = random.choice(possible)
                    perturbed.add(new_edge)
                    
                    if len(perturbed) == target_edges:
                        perturbed_score = compute_max_book(perturbed)
                        if perturbed_score < best_score:
                            best_score = perturbed_score
                            best_edge_set = set(perturbed)
                            edge_set = set(perturbed)
                            current_score = perturbed_score
    
    # Return the best solution found
    result = []
    for u, v in best_edge_set:
        result.append([u, v])
    return result
# EVOLVE-BLOCK-END
