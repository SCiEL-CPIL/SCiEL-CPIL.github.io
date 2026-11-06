import time
import random
import numpy as np

def _build_k32():
    """Build the 32-vertex 4-coloring from GF(16) cubic residues."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return np.array(colors, dtype=np.uint8)

_K32 = _build_k32()

def _edge_index(i, j, n):
    """Index of edge (i,j) in the flat colors array (i<j)."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def _validate_fast(colors, n):
    """Fast validation: no color class contains a triangle."""
    if n <= 3:
        return True
    # Build adjacency bitmasks for each color
    # Using Python integers for bitmasks is fast for n <= 62
    adj = [0] * (4 * n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1

    for c in range(4):
        base = c * n
        for i in range(n):
            nbrs = adj[base + i]
            temp = nbrs
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                temp ^= lsb
                if adj[base + j] & nbrs:
                    return False
    return True

def _count_triangles(colors, n):
    """Count total monochromatic triangles."""
    if n < 3:
        return 0
    adj = [0] * (4 * n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1
    count = 0
    for c in range(4):
        base = c * n
        for i in range(n):
            nbrs = adj[base + i]
            temp = nbrs
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                temp ^= lsb
                if i < j:
                    common = adj[base + j] & nbrs
                    # Count edges between i and j's neighbors that are also i's neighbors
                    # Actually, if k is in common, (i,j,k) is a triangle.
                    # We count each triangle 3 times (once per vertex).
                    count += common.bit_count()
    return count // 3

def _local_search(colors, n, deadline, start_temp=1.0):
    """
    Simulated Annealing to minimize monochromatic triangles.
    """
    current_colors = colors.copy()
    current_n = n
    current_energy = _count_triangles(current_colors, current_n)
    best_colors = current_colors.copy()
    best_n = current_n
    best_energy = current_energy
    
    temp = start_temp
    cooling = 0.999
    
    # Precompute edge indices for fast access? 
    # We will use a list of (i, j, idx) for all edges.
    edges = []
    idx = 0
    for i in range(current_n):
        for j in range(i + 1, current_n):
            edges.append((i, j, idx))
            idx += 1
            
    step = 0
    while time.time() < deadline - 5:
        if current_energy == 0:
            # Try to add a vertex
            new_colors, new_n = _try_add_vertex_fast(current_colors, current_n, deadline)
            if new_n > current_n:
                current_n = new_n
                current_colors = new_colors
                current_energy = _count_triangles(current_colors, current_n)
                if current_energy == 0:
                    best_colors = current_colors.copy()
                    best_n = current_n
                    best_energy = 0
                    # Restart search with larger n
                    edges = []
                    idx = 0
                    for i in range(current_n):
                        for j in range(i + 1, current_n):
                            edges.append((i, j, idx))
                            idx += 1
                    temp = start_temp
                else:
                    # If energy > 0, we might want to cool down or restart
                    pass
            else:
                # Failed to add vertex, continue minimizing energy
                pass
        
        if current_energy > 0:
            # Pick a random edge
            i, j, idx = random.choice(edges)
            old_c = current_colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Calculate delta energy
            # This is expensive to do exactly. We can approximate or just recalc.
            # For n=32-50, recalc is O(n^2) ~ 2500 ops. Fast enough.
            current_colors[idx] = new_c
            new_energy = _count_triangles(current_colors, current_n)
            
            delta = new_energy - current_energy
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = current_colors.copy()
                    best_n = current_n
            else:
                current_colors[idx] = old_c
            
            temp *= cooling
            step += 1
        else:
            # Energy is 0, try to add vertex more aggressively
            new_colors, new_n = _try_add_vertex_fast(current_colors, current_n, deadline)
            if new_n > current_n:
                current_n = new_n
                current_colors = new_colors
                current_energy = _count_triangles(current_colors, current_n)
                if current_energy == 0:
                    best_colors = current_colors.copy()
                    best_n = current_n
                    best_energy = 0
                    edges = []
                    idx = 0
                    for i in range(current_n):
                        for j in range(i + 1, current_n):
                            edges.append((i, j, idx))
                            idx += 1
                    temp = start_temp
                else:
                    # Accept if it's not too bad? Or revert?
                    # If we accept non-zero energy, we can search from there.
                    if current_energy < 5: # Small penalty
                        best_colors = current_colors.copy()
                        best_n = current_n
                        best_energy = current_energy
                        edges = []
                        idx = 0
                        for i in range(current_n):
                            for j in range(i + 1, current_n):
                                edges.append((i, j, idx))
                                idx += 1
                        temp = start_temp
            else:
                # Stuck, cool down
                temp *= cooling

    return best_colors, best_n

def _try_add_vertex_fast(colors, n, deadline):
    """
    Try to add a vertex using greedy with random restarts.
    """
    # Build adjacency lists for fast conflict checking
    adj = [[] for _ in range(4)]
    for c in range(4):
        adj[c] = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].append(j)
            adj[c][j].append(i)
            idx += 1
            
    max_attempts = 50
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        assignment = [-1] * n
        conflict = [set() for _ in range(4)]
        current_sets = [set() for _ in range(4)]
        
        success = True
        for v in order:
            valid_colors = []
            for c in range(4):
                if v not in conflict[c]:
                    valid_colors.append(c)
            
            if not valid_colors:
                success = False
                break
            
            # Heuristic: pick color with fewest conflicts or fewest current members
            c = min(valid_colors, key=lambda x: len(current_sets[x]))
            
            assignment[v] = c
            current_sets[c].add(v)
            
            for u in adj[c][v]:
                if assignment[u] == -1:
                    conflict[c].add(u)
        
        if success:
            new_colors = np.concatenate([colors, np.array(assignment, dtype=np.uint8)])
            return new_colors, n + 1
            
    return None, n

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 32
    best_colors = _K32.copy()
    
    # Run local search
    found_colors, found_n = _local_search(best_colors, best_n, deadline)
    
    if found_n > best_n:
        best_n = found_n
        best_colors = found_colors
        
    # Final validation
    if time.time() < deadline:
        if not _validate_fast(best_colors, best_n):
            # Fallback to seed if validation fails (should not happen)
            best_n = 32
            best_colors = _K32.copy()
            
    return {'n': best_n, 'colors': best_colors.tolist()}
