"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    # Target n=50 is the known maximum. We will try to find a valid coloring for n=50.
    # If that fails within time, we fall back to smaller n.
    
    def find_coloring(n, time_limit=110.0):
        """
        Uses Min-Conflicts with Simulated Annealing to find a valid 4-coloring of K_n.
        Returns list of colors in lexicographic order or None.
        """
        start_time = time.time()
        
        # Precompute edge indices and triangle neighbors
        # edges[i, j] for i < j
        # We store colors in a 2D array C[n][n] for O(1) access, symmetric.
        # colors[i][j] is the color of edge (i, j).
        
        # List of edges in lexicographic order for output
        edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                edge_list.append((i, j))
        
        num_edges = len(edge_list)
        
        # Map (i, j) to index in edge_list for output, but we work with matrix C
        # C is n x n
        C = np.zeros((n, n), dtype=np.int8)
        
        # Initialize randomly
        random.seed(42)
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, 3)
                C[i, j] = c
                C[j, i] = c
        
        # Precompute for each edge (u, v), the list of other vertices w
        # such that (u, v, w) forms a triangle.
        # Actually, we can just iterate w from 0 to n-1 excluding u, v.
        
        def count_conflicts():
            """Count total number of monochromatic triangles."""
            conflicts = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = C[i, j]
                    for k in range(j+1, n):
                        if C[i, k] == c and C[j, k] == c:
                            conflicts += 1
            return conflicts
        
        # Optimization: Instead of counting all conflicts from scratch, 
        # we can maintain the count or just evaluate the change in conflicts for a specific edge change.
        
        def delta_conflicts(u, v, new_color):
            """
            Calculate the change in number of monochromatic triangles if edge (u,v) 
            is changed to new_color.
            """
            old_color = C[u, v]
            if old_color == new_color:
                return 0
            
            delta = 0
            # Triangles containing (u, v) are (u, v, w) for w != u, v
            for w in range(n):
                if w == u or w == v:
                    continue
                cuw = C[u, w]
                cvw = C[v, w]
                
                # Check if (u, v, w) was a conflict before
                if old_color == cuw and old_color == cvw:
                    delta -= 1 # We removed a conflict
                
                # Check if (u, v, w) becomes a conflict after
                if new_color == cuw and new_color == cvw:
                    delta += 1 # We added a conflict
            
            return delta

        # Simulated Annealing parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        max_iter = 500000 # Adjust based on time
        
        current_conflicts = count_conflicts()
        best_conflicts = current_conflicts
        best_C = C.copy()
        
        # If we start with 0 conflicts, we are done
        if current_conflicts == 0:
            return C
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            old_color = C[u, v]
            # Try all 4 colors for this edge
            best_new_color = old_color
            best_delta = 0
            
            for c in range(4):
                d = delta_conflicts(u, v, c)
                if d < best_delta:
                    best_delta = d
                    best_new_color = c
            
            # Accept if it improves, or with probability exp(-delta/T)
            if best_delta < 0 or (best_delta > 0 and random.random() < np.exp(-best_delta / T)):
                # Apply change
                C[u, v] = best_new_color
                C[v, u] = best_new_color
                current_conflicts += best_delta
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_C = C.copy()
                    
                    if current_conflicts == 0:
                        # Found a valid coloring
                        break
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Optionally restart or increase T slightly to escape local minima
                # For now, keep cooling floor
        
        if best_conflicts == 0:
            return best_C
        return None

    # Try n=50 first
    # We need to be careful with time. 
    # Min-Conflicts is stochastic. We can run it for a few seconds.
    
    result = None
    # Try n=50
    start = time.time()
    C = find_coloring(50, time_limit=100.0)
    if C is not None:
        # Convert matrix to list
        colors = []
        for i in range(50):
            for j in range(i+1, 50):
                colors.append(int(C[i, j]))
        return {'n': 50, 'colors': colors}
    
    # If n=50 failed, try n=49
    if time.time() - start < 110.0:
        C = find_coloring(49, time_limit=105.0)
        if C is not None:
            colors = []
            for i in range(49):
                for j in range(i+1, 49):
                    colors.append(int(C[i, j]))
            return {'n': 49, 'colors': colors}
            
    # Fallback: Known valid coloring for n=32 using finite field GF(32) or similar?
    # Actually, let's use a simple backtracking for small n as a guaranteed fallback
    # Or just return n=1 if all else fails? No, n=32 is easy to find with the old method or simple logic.
    # Let's implement a quick backtracking for n=32 just in case, or use a known pattern.
    # A known 4-coloring for n=32 exists.
    # Let's try n=32 with the local search too, it's very fast.
    if time.time() - start < 115.0:
        C = find_coloring(32, time_limit=110.0 - (time.time() - start))
        if C is not None:
            colors = []
            for i in range(32):
                for j in range(i+1, 32):
                    colors.append(int(C[i, j]))
            return {'n': 32, 'colors': colors}
            
    # Absolute fallback
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
