import time
import random

N = 20
C = 20
DEADLINE = 120.0
START = time.time()

def elapsed():
    return time.time() - START

def construct():
    best_edges = []
    
    # Bipartite sets
    L = list(range(10))
    R = list(range(10, 20))
    
    # Try to construct a valid complete bipartite graph K_{10,10}
    # with rainbow 4-cycles
    for attempt in range(500):
        if elapsed() > DEADLINE - 5:
            break
        
        # Random order of cells
        cells = [(i, j) for i in range(10) for j in range(10)]
        random.shuffle(cells)
        
        M = [[-1]*10 for _ in range(10)]
        success = True
        
        for i, j in cells:
            # Try random colors
            colors = list(range(C))
            random.shuffle(colors)
            
            placed = False
            for c in colors:
                # Check if placing c at (i,j) creates a non-rainbow 4-cycle
                # A 4-cycle is formed if there exist i2 != i and j2 != j
                # such that edges (i2,j), (i,j2), (i2,j2) already exist.
                # The cycle is i-j-i2-j2-i.
                # Edges: (i,j) [new], (i2,j) [M[i2][j]], (i2,j2) [M[i2][j2]], (i,j2) [M[i][j2]]
                valid = True
                for i2 in range(10):
                    if i2 == i:
                        continue
                    if M[i2][j] == -1:
                        continue
                    for j2 in range(10):
                        if j2 == j:
                            continue
                        if M[i][j2] == -1 or M[i2][j2] == -1:
                            continue
                        
                        # All 4 edges exist
                        cols = {c, M[i2][j], M[i][j2], M[i2][j2]}
                        if len(cols) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    M[i][j] = c
                    placed = True
                    break
            
            if not placed:
                success = False
                break
        
        if success:
            # Convert M to edge list
            edges = []
            for i in range(10):
                for j in range(10):
                    if M[i][j] != -1:
                        edges.append([L[i], R[j], M[i][j]])
            
            if len(edges) > len(best_edges):
                best_edges = edges
            
            # If we found a complete bipartite graph, it's 100 edges.
            # This is likely the max for this structure.
            # We can return immediately as 100 is a very good score.
            if len(edges) >= 100:
                return edges

    # If no full bipartite found, try to return the best partial one
    if best_edges:
        return best_edges
    
    # Fallback: Simple star
    star = [[0, v, 0] for v in range(1, N)]
    return star
