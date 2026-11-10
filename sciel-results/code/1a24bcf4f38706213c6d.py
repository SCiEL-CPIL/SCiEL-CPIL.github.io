import random
import time

# Precompute all K4 edge index combinations
# Vertices 0..15
# Edges (i, j) with i < j
# Index of edge (i, j) is i * 15 - i * (i + 1) // 2 + (j - i - 1) ? 
# Let's just generate a list of (i, j) and map to index.

def get_edge_index(i, j):
    if i > j: i, j = j, i
    # Number of edges starting with k < i is sum_{k=0}^{i-1} (15-k) = 15i - i(i-1)/2
    # Edges starting with i: (i, i+1) is 0th, (i, j) is (j-i-1)th
    return 15*i - i*(i-1)//2 + (j - i - 1)

# Generate all K4s
k4s = []
verts = range(16)
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                # Edges: ab, ac, ad, bc, bd, cd
                idxs = [
                    get_edge_index(a, b),
                    get_edge_index(a, c),
                    get_edge_index(a, d),
                    get_edge_index(b, c),
                    get_edge_index(b, d),
                    get_edge_index(c, d)
                ]
                k4s.append(tuple(idxs))

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Try colors from 14 to 17
    for num_colors in [14, 15, 16, 17]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize random coloring
        coloring = [random.randint(0, num_colors - 1) for _ in range(120)]
        
        # Local Search
        for _ in range(20000): # Max iterations per color count
            if time.time() - start_time > time_limit:
                break
                
            # Check for violation
            viol = None
            for k4 in k4s:
                c_set = set()
                for idx in k4:
                    c_set.add(coloring[idx])
                if len(c_set) < 5:
                    viol = k4
                    break
            
            if viol is None:
                return coloring
            
            # Pick a random edge in the violating K4 and change its color
            idx = random.choice(viol)
            coloring[idx] = random.randint(0, num_colors - 1)
            
            # Heuristic: If we are stuck, maybe try a specific color?
            # For now, random is fine.
            
        # If we exit the loop without finding a solution, try next num_colors
    
    # Fallback: Use the seed's valid coloring (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
