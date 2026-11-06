import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs, maximizing cardinality.
    
    This implementation uses a maximum clique algorithm (Tomita's algorithm)
    on the graph of numbers in [1, 255] that share a factor with 255.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Identify candidates
    # Candidates must share a factor with 255 (3, 5, 17)
    candidates = [x for x in range(1, 256) if x != 255 and (x % 3 == 0 or x % 5 == 0 or x % 17 == 0)]
    n = len(candidates)
    
    # Map candidates to indices
    # candidates[i] is the number
    # We need to build the graph where edge (i, j) exists if gcd(candidates[i], candidates[j]) > 1
    
    # 2. Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        ci = candidates[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Tomita's Algorithm for Maximum Clique
    best_clique = []
    best_size = 0
    
    # Greedy coloring of the graph to get upper bounds
    # We process vertices in a specific order. 
    # Standard Tomita: 
    # 1. Color the graph.
    # 2. Process vertices in reverse order of color assignment.
    
    # Simple greedy coloring function for a set of vertices P
    def color_sort(P):
        # P is a bitmask or list of vertices
        # Returns a list of (vertex, color) sorted by color
        # We use a simple greedy coloring: assign smallest available color
        # This is not optimal coloring, but provides an upper bound for the clique in P
        # Actually, Tomita uses a specific ordering based on colors.
        
        # Let's implement the standard expand routine with coloring
        # We need a function to get the color of each vertex in P
        # and sort them.
        
        pass

    # A more direct implementation of Tomita's algorithm using bitmasks
    # Source: Adapted from standard implementations
    
    def expand(P, size):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if size > best_size:
                best_size = size
                # We need to reconstruct the clique. 
                # For simplicity and speed, we might just return the size 
                # and then find a clique of that size separately, 
                # or track the path.
                # Tracking path is easier if we pass the current clique.
                # However, for n=126, recursion depth is small.
                # Let's track the clique.
                pass
            return

        # Greedy coloring of P
        # We want to color P such that vertices with higher colors are processed first?
        # No, Tomita processes vertices in reverse order of the greedy coloring 
        # of the subgraph induced by P.
        # Specifically, if we color P with colors 1..k, and let P_i be the set of 
        # vertices with color i, we process P_k, then P_{k-1}, etc.
        # The upper bound for a clique starting with a vertex v of color c is size + c.
        
        # Let's implement a helper to color P and return vertices in order
        # and their colors.
        
        # Simple greedy coloring:
        # 1. Sort vertices in P by degree in P (descending) - heuristic
        # 2. Assign colors greedily.
        
        # To keep it simple and fast, I'll use a standard bit-mask based Tomita implementation.
        # Reference: "An Algorithm for the Maximum Clique Problem" by Tomita et al.
        
        # I will implement a simplified version that works well for n < 200.
        
        # 1. Get list of vertices in P
        # 2. Color them
        # 3. Sort by color descending
        # 4. Iterate
        
        # Efficient coloring:
        # We can use a simple loop.
        # Given P (bitmask), we extract vertices.
        # This might be slow if done naively in Python.
        # n=126 is small enough that list operations are fine.
        
        # Let's use a list of vertices for P
        # But passing lists is slower than bitmasks.
        # I'll stick to bitmasks for P, but convert to list for coloring/sorting.
        
        # Optimization: Pre-compute colors? No, P changes.
        
        # Let's try a different approach: Bron-Kerbosch with Pivot.
        # It's simpler to implement correctly.
        # With n=126, Bron-Kerbosch with pivot is likely fast enough.
        
        pass

    # Let's implement Bron-Kerbosch with Pivot
    # R: current clique
    # P: candidates
    # X: excluded
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # Choose pivot u from P U X
        # u should maximize |N(u) ∩ P|
        # If P is empty, we can't choose pivot from P, but X might not be empty.
        # Standard: pick u from P ∪ X.
        
        if P or X:
            # Convert P and X to lists for easier manipulation
            # Or use bitmasks
            # Let's use lists for clarity, n is small.
            P_list = [i for i in range(n) if (P >> i) & 1]
            X_list = [i for i in range(n) if (X >> i) & 1]
            candidates_pivot = P_list + X_list
            
            if not candidates_pivot:
                return
                
            # Find u with max degree in P
            max_deg = -1
            pivot = None
            for u in candidates_pivot:
                # Count neighbors in P
                deg = 0
                # adj[u] is bitmask
                # P is bitmask
                # Intersection: adj[u] & P
                # Count bits
                deg = bin(adj[u] & P).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # Branch on P \ N(pivot)
            # If pivot is None, branch on all P
            if pivot is not None:
                # P \ N(pivot)
                # N(pivot) is adj[pivot]
                # We want vertices in P that are NOT in adj[pivot]
                # Note: adj[pivot] does not include pivot itself.
                # So P \ N(pivot) includes pivot if pivot in P?
                # Yes, usually.
                
                # Let's iterate through vertices in P
                # For each v in P \ N(pivot):
                
                # Create list of vertices to process
                # P_mask = P
                # N_pivot_mask = adj[pivot]
                # To_process_mask = P_mask & ~N_pivot_mask
                
                # However, we must remove v from P and add to X as we go.
                
                # Let's just iterate P_list
                # But we need to exclude neighbors of pivot.
                
                # Re-calculate P_list excluding neighbors of pivot
                # This is O(n)
                
                # Actually, standard implementation iterates P, and for each v,
                # checks if v is in P \ N(pivot).
                
                # Let's do it this way:
                # Get list of v in P such that v is not neighbor of pivot
                # (and v != pivot, though pivot not in adj[pivot])
                
                verts_to_process = []
                for v in P_list:
                    # Check if v is neighbor of pivot
                    # If not, process v
                    # Note: if v == pivot, it is not a neighbor of itself, so it is processed.
                    if not ((adj[pivot] >> v) & 1):
                        verts_to_process.append(v)
            else:
                verts_to_process = P_list
                
            for v in verts_to_process:
                # Branch on v
                # New R = R + v
                # New P = P ∩ N(v)
                # New X = X ∩ N(v)
                
                new_R = R + [v]
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                bron_kerbosch(new_R, new_P, new_X)
                
                # Remove v from P, add to X
                P = P & ~(1 << v)
                X = X | (1 << v)
                
                if time.time() > deadline:
                    return

    # Initialize
    initial_P = (1 << n) - 1  # All vertices
    initial_X = 0
    bron_kerbosch([], initial_P, initial_X)
    
    # best_clique contains indices of the maximum clique in the candidate graph
    # The actual numbers are candidates[i]
    result_numbers = [candidates[i] for i in best_clique]
    
    # Add 255
    result_numbers.append(255)
    
    # Verify validity (optional, for safety)
    # for i in range(len(result_numbers)):
    #     for j in range(i+1, len(result_numbers)):
    #         if math.gcd(result_numbers[i], result_numbers[j]) == 1:
    #             # Fallback to multiples of 3
    #             return [x for x in range(3, 256, 3)]
    
    return result_numbers
