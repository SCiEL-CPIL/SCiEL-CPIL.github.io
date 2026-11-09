"""Program for the suite task; uses greedy sequential vertex addition."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Build a graph on as many vertices as possible (up to 22) such that
    no set of 6 or more vertices induces a regular subgraph.
    
    Strategy: Greedy sequential vertex addition. Start with a base graph
    and add vertices one by one. For each new vertex, try all possible
    edge patterns to existing vertices and pick the one that minimizes
    the number of new regular induced subgraphs of size >= 6.
    
    We use a randomized restart approach to try to find a larger graph.
    """
    random.seed(42)
    
    def is_regular(adj, subset):
        """Check if the induced subgraph on subset is regular."""
        if len(subset) < 2:
            return True
        degrees = []
        s = set(subset)
        for v in subset:
            deg = sum(1 for u in subset if u != v and adj[v][u])
            degrees.append(deg)
        return len(set(degrees)) == 1
    
    def count_regular_subgraphs(adj, n, min_size=6):
        """Count all subsets of size >= min_size that induce a regular subgraph."""
        count = 0
        for size in range(min_size, n + 1):
            for subset in itertools.combinations(range(n), size):
                if is_regular(adj, subset):
                    count += 1
        return count
    
    def add_vertex_greedy(adj, n, new_vertex_idx):
        """
        Try all 2^n possible edge patterns for a new vertex connected to
        existing vertices 0..n-1. Return the pattern that minimizes the
        number of new regular induced subgraphs of size >= 6.
        
        adj is an n x n adjacency matrix (list of lists of 0/1).
        new_vertex_idx = n (the new vertex index).
        """
        best_pattern = None
        best_count = float('inf')
        
        # For efficiency, we'll check all 2^n patterns
        # But for n up to ~20, 2^20 = 1M which might be slow.
        # Let's limit: for n > 12, we'll use a heuristic subset of patterns.
        
        if n <= 10:
            patterns = range(2 ** n)
        else:
            # Sample a subset of patterns for larger n
            patterns = [random.randint(0, 2 ** n - 1) for _ in range(500)]
            patterns.append(0)  # always try no edges
            patterns.append(2 ** n - 1)  # always try all edges
        
        for pattern in patterns:
            # Build the new adjacency matrix
            new_n = n + 1
            new_adj = [row[:] for row in adj]
            new_adj.append([0] * new_n)
            for i in range(n):
                new_adj[i].append(0)
            
            for i in range(n):
                if (pattern >> i) & 1:
                    new_adj[new_vertex_idx][i] = 1
                    new_adj[i][new_vertex_idx] = 1
            
            # Count regular subgraphs of size >= 6 in the new graph
            # Optimization: only check subsets that include the new vertex
            # plus subsets that don't (but those were already counted before)
            # Actually, we need to count ALL regular subgraphs in the new graph.
            # But we can shortcut: any regular subgraph not containing the new vertex
            # was already present in the old graph. So we only need to check subsets
            # that contain the new vertex.
            
            count = 0
            for size in range(6, new_n + 1):
                # Subsets of size 'size' that include the new vertex (index n)
                # Choose size-1 from the existing n vertices
                for subset_old in itertools.combinations(range(n), size - 1):
                    subset = list(subset_old) + [n]
                    if is_regular(new_adj, subset):
                        count += 1
            
            if count < best_count:
                best_count = count
                best_pattern = pattern
            
            # Early termination if we find a pattern with 0 new regular subgraphs
            if best_count == 0:
                break
        
        return best_pattern, best_count
    
    # Start with a small base and grow
    # Try different starting points
    
    best_n = 5
    best_edges = []
    
    for attempt in range(50):
        # Start with random small graph
        start_n = random.randint(4, 8)
        adj = [[0] * start_n for _ in range(start_n)]
        
        # Randomly initialize edges
        for i in range(start_n):
            for j in range(i + 1, start_n):
                if random.random() < 0.3:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Check if initial graph is valid
        if count_regular_subgraphs(adj, start_n) > 0:
            continue
        
        n = start_n
        current_adj = [row[:] for row in adj]
        
        # Try to add vertices
        for new_idx in range(n, 22):
            if new_idx >= 22:
                break
            
            pattern, new_count = add_vertex_greedy(current_adj, n, new_idx)
            
            if new_count > 0:
                # This pattern creates new regular subgraphs; try to see if any pattern works
                # If no pattern works with 0, break
                break
            
            # Add the vertex with the chosen pattern
            new_n = n + 1
            new_adj = [row[:] for row in current_adj]
            new_adj.append([0] * new_n)
            for i in range(n):
                new_adj[i].append(0)
            
            for i in range(n):
                if (pattern >> i) & 1:
                    new_adj[new_idx][i] = 1
                    new_adj[i][new_idx] = 1
            
            current_adj = new_adj
            n = new_n
        
        if n > best_n:
            best_n = n
            # Extract edges
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if current_adj[i][j]:
                        best_edges.append([i, j])
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
