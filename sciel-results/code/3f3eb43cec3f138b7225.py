import time
import random
import numpy as np

def _build_adj(colors, n):
    """Build adjacency matrix from colors list."""
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
    return adj

def _edge_index(i, j, n):
    """Get edge index for pair (i,j) with i<j."""
    # Number of edges before row i
    # Edges in rows 0..i-1: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    base = i * (n - 1) - i * (i - 1) // 2
    return base + (j - i - 1)

def _has_mono_k222_fast(adj, n):
    """Check if adjacency matrix contains a monochromatic K2,2,2 using bitmasks."""
    # For each vertex, precompute bitmask of neighbors for each color
    # neigh_mask[v][c] = bitmask of vertices u where adj[v][u] == c
    neigh_mask = [[0, 0] for _ in range(n)]
    for v in range(n):
        m0 = 0
        m1 = 0
        for u in range(n):
            if u == v:
                continue
            if adj[v][u] == 0:
                m0 |= (1 << u)
            else:
                m1 |= (1 << u)
        neigh_mask[v][0] = m0
        neigh_mask[v][1] = m1
    
    # For each pair (a,b) and color c, check if there exist two disjoint pairs
    # in the common c-neighborhood that are also connected by color c
    for a in range(n):
        for b in range(a+1, n):
            for c in (0, 1):
                if adj[a][b] != c:
                    continue
                # Common neighbors with color c
                common = neigh_mask[a][c] & neigh_mask[b][c]
                # Remove a and b
                common &= ~(1 << a)
                common &= ~(1 << b)
                if common == 0:
                    continue
                # Count bits
                if bin(common).count('1') < 4:
                    continue
                # Find two disjoint edges in the induced subgraph on common with color c
                # Get list of vertices in common
                verts = []
                mask = common
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    mask ^= lsb
                m = len(verts)
                # Find all edges in verts with color c
                edges = []
                for i in range(m):
                    vi = verts[i]
                    # For each other vertex in verts, check if edge has color c
                    for j in range(i+1, m):
                        vj = verts[j]
                        if adj[vi][vj] == c:
                            edges.append((i, j))
                # Check for two disjoint edges
                for i in range(len(edges)):
                    a1, b1 = edges[i]
                    for j in range(i+1, len(edges)):
                        a2, b2 = edges[j]
                        if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                            return True
    return False

def _count_mono_k222_fast(adj, n):
    """Count the number of monochromatic K2,2,2 subgraphs using bitmasks."""
    neigh_mask = [[0, 0] for _ in range(n)]
    for v in range(n):
        m0 = 0
        m1 = 0
        for u in range(n):
            if u == v:
                continue
            if adj[v][u] == 0:
                m0 |= (1 << u)
            else:
                m1 |= (1 << u)
        neigh_mask[v][0] = m0
        neigh_mask[v][1] = m1
    
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in (0, 1):
                if adj[a][b] != c:
                    continue
                common = neigh_mask[a][c] & neigh_mask[b][c]
                common &= ~(1 << a)
                common &= ~(1 << b)
                if bin(common).count('1') < 4:
                    continue
                verts = []
                mask = common
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    mask ^= lsb
                m = len(verts)
                edges = []
                for i in range(m):
                    vi = verts[i]
                    for j in range(i+1, m):
                        vj = verts[j]
                        if adj[vi][vj] == c:
                            edges.append((i, j))
                for i in range(len(edges)):
                    a1, b1 = edges[i]
                    for j in range(i+1, len(edges)):
                        a2, b2 = edges[j]
                        if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                            count += 1
    return count

def _paley_colors(n):
    """Generate Paley coloring for prime n."""
    squares = set()
    for x in range(1, n // 2 + 1):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s limit
    
    n_target = 30
    num_edges = n_target * (n_target - 1) // 2
    
    # Fallback: n=29 Paley coloring
    best_n = 29
    best_colors_29 = _paley_colors(29)
    
    # Precompute edge index mapping
    edge_to_pair = []
    for i in range(n_target):
        for j in range(i+1, n_target):
            edge_to_pair.append((i, j))
    
    random.seed(42)
    
    # Simulated annealing parameters
    T_start = 2.0
    T_end = 0.01
    cooling_rate = 0.999
    
    max_restarts = 30
    max_steps_per_restart = 20000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: use Paley-like structure with random perturbations
        # For restart 0, use a structured initialization
        if restart == 0:
            # Try to extend Paley n=29 by adding vertex 29 with balanced connections
            colors = _paley_colors(29)
            # Add edges from vertex 29 to vertices 0..28
            for i in range(29):
                # Random but somewhat balanced
                colors.append(random.randint(0, 1))
        elif restart == 1:
            # All zeros then randomize
            colors = [0] * num_edges
            for _ in range(num_edges // 2):
                colors[random.randint(0, num_edges - 1)] = 1
        else:
            colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        adj = _build_adj(colors, n_target)
        
        # Initial energy
        energy = _count_mono_k222_fast(adj, n_target)
        if energy == 0:
            return {'n': n_target, 'colors': colors[:]}
        
        best_energy = energy
        best_adj = [row[:] for row in adj]
        
        T = T_start
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            # Pick random edge to flip
            edge_idx = random.randint(0, num_edges - 1)
            i, j = edge_to_pair[edge_idx]
            
            old_color = adj[i][j]
            new_color = 1 - old_color
            
            # Flip
            adj[i][j] = new_color
            adj[j][i] = new_color
            colors[edge_idx] = new_color
            
            # Compute new energy
            new_energy = _count_mono_k222_fast(adj, n_target)
            
            # Accept or reject
            delta = new_energy - energy
            if delta <= 0:
                # Accept improvement
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_adj = [row[:] for row in adj]
                    if energy == 0:
                        return {'n': n_target, 'colors': colors[:]}
            else:
                # Metropolis acceptance
                if T > 0:
                    prob = 1.0 / (1.0 + np.exp(delta / T))
                    if random.random() < prob:
                        energy = new_energy
                    else:
                        # Reject: flip back
                        adj[i][j] = old_color
                        adj[j][i] = old_color
                        colors[edge_idx] = old_color
            
            # Cool down
            T *= cooling_rate
        
        # After each restart, check best found
        if best_energy == 0:
            # Extract colors from best_adj
            final_colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    final_colors.append(best_adj[i][j])
            return {'n': n_target, 'colors': final_colors}
    
    # If we didn't find n=30, return n=29
    return {'n': best_n, 'colors': best_colors_29}
