import time
import random
import sys
import math
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Fallback: n=5 (trivially valid, no C6 possible)
    def get_fallback(n):
        m = n * (n - 1) // 2
        colors = [0] * m
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    def verify_coloring(n, colors):
        """Verify that no color class contains a C6. Returns True if valid."""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        
        # Build adjacency bitmasks for each color
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 3:
                    return False
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for C6 in each color class
        for c in range(4):
            for start in range(n):
                # Find all paths of length 5 from start, check if endpoint connects back to start
                # Use iterative DFS with bitmasks
                # State: (node, visited_mask, depth)
                stack = [(start, 1 << start, 0)]
                found = False
                while stack and not found:
                    node, visited, depth = stack.pop()
                    if depth == 5:
                        if adj[c][node] & (1 << start):
                            found = True
                            break
                        continue
                    
                    neighbors = adj[c][node]
                    while neighbors:
                        lsb = neighbors & -neighbors
                        nb = lsb.bit_length() - 1
                        if not (visited & lsb):
                            stack.append((nb, visited | lsb, depth + 1))
                        neighbors ^= lsb
                
                if found:
                    return False
        
        return True
    
    def precompute_cycle_info(n):
        """Precompute all C6 cycles and edge-to-cycle mapping."""
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        m = n * (n - 1) // 2
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        vertices = list(range(n))
        
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            seen_cycles = set()
            
            for perm in itertools.permutations(others):
                # Avoid reverse duplicates
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                key = tuple(sorted(cycle_edges))
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycle_id = len(cycles)
                    cycles.append(cycle_edges)
                    for e in cycle_edges:
                        edge_to_cycles[e].append(cycle_id)
        
        return cycles, edge_to_cycles
    
    def count_bad_cycles(colors, cycles):
        """Count monochromatic C6 cycles."""
        bad = 0
        for ce in cycles:
            c0 = colors[ce[0]]
            mono = True
            for e in ce[1:]:
                if colors[e] != c0:
                    mono = False
                    break
            if mono:
                bad += 1
        return bad
    
    def spectral_initial_coloring(n, seed=None):
        """Generate initial coloring using spectral graph theory approach."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**31 - 1))
        
        # Generate a random regular graph
        degree = max(4, min(6, n - 1))
        if degree % 2 == 1 and n % 2 == 1:
            degree -= 1
        
        # Create adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            neighbors = list(range(n))
            neighbors.remove(i)
            random.shuffle(neighbors)
            for j in neighbors[:degree]:
                A[i][j] = 1
                A[j][i] = 1
        
        # Ensure regularity by adjusting
        for i in range(n):
            deg = int(A[i].sum())
            if deg != degree:
                # Adjust
                if deg > degree:
                    neighbors = [j for j in range(n) if j != i and A[i][j] == 1]
                    random.shuffle(neighbors)
                    for j in neighbors[:deg - degree]:
                        A[i][j] = 0
                        A[j][i] = 0
                elif deg < degree:
                    neighbors = [j for j in range(n) if j != i and A[i][j] == 0]
                    random.shuffle(neighbors)
                    for j in neighbors[:degree - deg]:
                        A[i][j] = 1
                        A[j][i] = 1
        
        # Compute Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            # Fallback to cyclic coloring
            m = n * (n - 1) // 2
            colors = [0] * m
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    colors[idx] = (j - i) % 4
                    idx += 1
            return colors
        
        # Use the 4 eigenvectors with smallest non-zero eigenvalues
        # Skip the first (constant) eigenvector
        if eigenvalues[0] < 1e-10:
            vecs = eigenvectors[:, 1:5]
        else:
            vecs = eigenvectors[:, :4]
        
        # Assign colors based on quadrant signs
        m = n * (n - 1) // 2
        colors = [0] * m
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                vi = vecs[i]
                vj = vecs[j]
                # Compute sign pattern
                s = 0
                for k in range(min(4, vecs.shape[1])):
                    if vi[k] > 0 and vj[k] > 0:
                        s += 1
                    elif vi[k] < 0 and vj[k] < 0:
                        s += 1
                color = s % 4
                colors[idx] = color
                idx += 1
        
        return colors
    
    def simulated_annealing(n, time_budget, initial_colors=None):
        """Run simulated annealing to minimize monochromatic C6 count."""
        if time.time() > deadline - 5:
            return None
        
        try:
            cycles, edge_to_cycles = precompute_cycle_info(n)
        except Exception:
            return None
        
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Initialize
        if initial_colors is None:
            colors = spectral_initial_coloring(n)
        else:
            colors = initial_colors[:]
        
        # Count initial bad cycles
        bad_count = count_bad_cycles(colors, cycles)
        if bad_count == 0:
            return {'n': n, 'colors': colors}
        
        best_colors = colors[:]
        best_bad = bad_count
        
        # Simulated annealing parameters
        T = 5.0
        cool_rate = 0.9995
        start_time_sa = time.time()
        
        while time.time() < start_time_sa + time_budget and time.time() < deadline - 5:
            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]
            
            # Pick new color (different from old)
            other_colors = [c for c in range(4) if c != c_old]
            c_new = random.choice(other_colors)
            
            # Calculate delta in bad cycle count
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                
                # Check if was monochromatic before
                was_mono = True
                for ed in ce:
                    if ed != e and colors[ed] != c_old:
                        was_mono = False
                        break
                
                # Check if will be monochromatic after
                will_be_mono = True
                for ed in ce:
                    if ed != e and colors[ed] != c_new:
                        will_be_mono = False
                        break
                
                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            else:
                if delta / T < 50:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                colors[e] = c_new
                bad_count += delta
                
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}
            
            T *= cool_rate
        
        if best_bad == 0:
            return {'n': n, 'colors': best_colors}
        return None
    
    # Try to find valid colorings for decreasing n
    targets = [17, 16, 15, 14, 13, 12, 11, 10]
    
    time_allocations = {
        17: 55,
        16: 30,
        15: 15,
        14: 8,
        13: 5,
        12: 3,
        11: 2,
        10: 1,
    }
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        alloc = time_allocations.get(n, 5)
        remaining = deadline - 10 - time.time()
        alloc = min(alloc, remaining)
        
        if alloc < 1:
            break
        
        # Try multiple initializations
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            init_colors = spectral_initial_coloring(n, seed=seed)
            result = simulated_annealing(n, alloc / 5, init_colors)
            
            if result and result['n'] == n:
                if verify_coloring(n, result['colors']):
                    best_result = result
                    found = True
                    break
        
        if found:
            break
    
    # If we didn't find anything better, try quick random search for small n
    if best_result['n'] <= 5:
        for n in [8, 7, 6]:
            if time.time() > deadline - 5:
                break
            for _ in range(20):
                m = n * (n - 1) // 2
                colors = [random.randint(0, 3) for _ in range(m)]
                if verify_coloring(n, colors):
                    best_result = {'n': n, 'colors': colors}
                    break
            if best_result['n'] >= n:
                break
    
    return best_result
