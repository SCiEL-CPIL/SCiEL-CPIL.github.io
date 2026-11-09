import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 60.0  # Keep well under 90s limit
    
    best_edges = None
    best_score = float('inf')
    
    # Vertices 0..11
    # Bipartite split: U = 0..5, V = 6..11
    # We want 3-regular bipartite graphs.
    
    def get_cycle_lengths(edges):
        # Build adjacency list
        adj = {i: set() for i in range(12)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all simple cycles
        # Since it's bipartite, we can just DFS
        # We want to find lengths of all simple cycles
        lengths = set()
        
        # To avoid duplicates, we can enforce that the smallest vertex in the cycle is the start
        # And the second vertex is smaller than the last?
        # Standard approach:
        # For each start node s, DFS to find cycles back to s.
        # Only count cycles where s is the smallest vertex in the cycle.
        
        for s in range(12):
            # DFS
            # Stack: (current, path)
            stack = [(s, [s])]
            while stack:
                curr, path = stack.pop()
                for neighbor in adj[curr]:
                    if neighbor == s and len(path) > 2:
                        # Found a cycle
                        # Check if s is the smallest vertex in the path
                        if min(path) == s:
                            lengths.add(len(path))
                    elif neighbor > s:
                        # Only visit nodes greater than s to ensure s is min
                        # And avoid visiting s again until we close the loop
                        if neighbor not in path:
                            stack.append((neighbor, path + [neighbor]))
        return lengths

    def evaluate(edges):
        # Check validity
        # 1. Simple, undirected
        # 2. Degree >= 3
        # 3. Triangle-free (guaranteed by bipartite)
        # 4. Contains a cycle
        
        # Degree check
        deg = [0]*12
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d < 3 for d in deg):
            return float('inf')
            
        # Cycle check
        lengths = get_cycle_lengths(edges)
        if not lengths:
            return float('inf')
            
        return len(lengths)

    # Generate random 3-regular bipartite graphs
    # Use a simple rejection sampling or configuration model
    # Or just iterate through some known structures if random is slow
    
    # Let's try random permutations of a base 3-regular bipartite graph
    # Base: 3 disjoint K_{3,3}? No, that's 9 vertices.
    # Base: One 3-regular bipartite graph on 12 vertices.
    # Example: The 3x4 grid? No.
    
    # Let's generate random 6x6 matrices with 3 ones per row/col.
    
    def generate_random_bipartite():
        # Start with a valid matrix
        # Row 0: 0,1,2
        # Row 1: 1,2,3
        # ...
        # Let's just use a known valid one and shuffle columns
        # Or use a simple random generation with repair
        
        # Simpler: Use a list of possible 3-regular bipartite graphs?
        # There are many.
        
        # Let's try to construct one:
        # U: 0,1,2,3,4,5
        # V: 6,7,8,9,10,11
        
        # Try to make it a "circular" graph
        # u_i connected to v_i, v_{i+1}, v_{i+2} (mod 6)
        edges = []
        for i in range(6):
            for j in [0, 1, 2]:
                u = i
                v = (i + j) % 6 + 6
                edges.append((u, v))
        return edges

    # This specific graph (circulant) might have specific cycle lengths.
    # Let's evaluate it first.
    base_edges = generate_random_bipartite()
    score = evaluate(base_edges)
    if score < best_score:
        best_score = score
        best_edges = base_edges
        
    # If score is 1, we are done.
    if best_score == 1:
        return [[u, v] for u, v in best_edges]

    # If not, try random perturbations or other structures.
    # Given the time limit, we might just return the best found.
    # The circulant graph likely has 4 and 6 cycles.
    
    # Let's check if we can find a graph with only 4-cycles.
    # This requires no 6-cycles.
    
    # For now, return the best found.
    # The seed program had score 1.0.
    # If seed value was 3, and we find value 2, score is 1.5.
    # If we find value 1, score is 3.0.
    
    # To be safe, we can run a small search.
    import random
    random.seed(42)
    
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Generate a random 3-regular bipartite graph
        # Use a simple method: shuffle the neighbor lists
        # Start with the circulant one
        edges = generate_random_bipartite()
        
        # Randomly swap edges to maintain regularity?
        # Or just generate a new random 6x6 (0,1) matrix with 3 ones/row/col.
        
        # Let's try a random permutation of columns for the base pattern
        # Base pattern rows:
        # 0: 0,1,2
        # 1: 1,2,3
        # 2: 2,3,4
        # 3: 3,4,5
        # 4: 4,5,0
        # 5: 5,0,1
        
        # This is a fixed structure.
        # Let's try to randomize the connections.
        
        # Algorithm:
        # 1. Create 6 lists for U, each with 3 random V indices.
        # 2. Ensure each V index appears exactly 3 times.
        
        # This is a perfect matching problem in a 3-partite sense?
        # We can use a simple random greedy with backtracking or just rejection.
        
        # Given the small size, rejection is fine.
        while True:
            v_counts = [0]*6
            u_neighbors = []
            for i in range(6):
                # Pick 3 distinct V's
                candidates = list(range(6))
                random.shuffle(candidates)
                # Try to pick 3 that don't exceed count 3
                # This is tricky.
                pass
            
            # Simpler: Generate random 6x6 matrix with 3 ones/row.
            # Then check if cols have 3 ones.
            # If not, retry.
            
            mat = [[0]*6 for _ in range(6)]
            for i in range(6):
                cols = list(range(6))
                random.shuffle(cols)
                for j in cols[:3]:
                    mat[i][j] = 1
            
            # Check cols
            valid = True
            for j in range(6):
                if sum(mat[i][j] for i in range(6)) != 3:
                    valid = False
                    break
            
            if valid:
                edges = []
                for i in range(6):
                    for j in range(6):
                        if mat[i][j]:
                            edges.append((i, j+6))
                break
        
        score = evaluate(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            
            if best_score == 1:
                break

    return [[u, v] for u, v in best_edges]
