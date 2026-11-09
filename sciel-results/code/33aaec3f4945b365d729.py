import time
import random
import numpy as np

def _paley_colors(n):
    """Generate Paley graph coloring for prime n."""
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

def _count_mono_k222(adj):
    """
    Count monochromatic K2,2,2 subgraphs in the graph defined by adj matrix.
    adj is n x n, values 0 or 1.
    """
    n = adj.shape[0]
    total = 0
    for c in (0, 1):
        A = (adj == c).astype(np.int8)
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    continue
                # Common neighbors of u and v in color c
                mask = A[u] & A[v]
                mask[u] = 0
                mask[v] = 0
                S_idx = np.where(mask)[0]
                m = len(S_idx)
                if m < 4:
                    continue
                
                # Subgraph induced by S_idx
                M = A[S_idx][:, S_idx]
                
                # Degrees in M
                degs = M.sum(axis=1)
                
                # Number of edges in M
                E_count = M.sum() // 2
                if E_count < 2:
                    continue
                
                # Find edges in M (upper triangle)
                i_idx, j_idx = np.where(np.triu(M, k=1) == 1)
                if len(i_idx) == 0:
                    continue
                
                vals = E_count - degs[i_idx] - degs[j_idx]
                sum_val = np.sum(vals)
                total += sum_val // 2
    return total

def _sa_search(adj_init, deadline):
    """
    Run Simulated Annealing to minimize energy.
    Returns (adj, energy) of best found.
    """
    n = adj_init.shape[0]
    adj = adj_init.copy()
    current_energy = _count_mono_k222(adj)
    
    if current_energy == 0:
        return adj, 0
        
    T_start = 10.0
    T_end = 0.05
    steps = 3000
    
    if steps > 0:
        alpha = (T_end / T_start) ** (1.0 / steps)
    else:
        alpha = 1.0
        
    T = T_start
    best_adj = adj.copy()
    best_energy = current_energy
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Pick random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        while i == j:
            j = random.randint(0, n - 1)
            
        if i > j:
            i, j = j, i
            
        old_color = adj[i, j]
        new_color = 1 - old_color
        
        # Apply flip
        adj[i, j] = new_color
        adj[j, i] = new_color
        
        new_energy = _count_mono_k222(adj)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj.copy()
                if best_energy == 0:
                    return best_adj, 0
        else:
            # Revert
            adj[i, j] = old_color
            adj[j, i] = old_color
            
        # Cool
        T *= alpha
        
    return best_adj, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    n_target = 30
    colors_29 = _paley_colors(29)
    
    # Initialization 1: Based on Paley 29 + random vertex
    adj_paley_29 = np.zeros((29, 29), dtype=np.int8)
    idx = 0
    for i in range(29):
        for j in range(i+1, 29):
            c = colors_29[idx]
            adj_paley_29[i, j] = c
            adj_paley_29[j, i] = c
            idx += 1
            
    adj_init_1 = np.zeros((n_target, n_target), dtype=np.int8)
    adj_init_1[:29, :29] = adj_paley_29
    # Randomly connect vertex 29 to 0..28
    for i in range(29):
        c = random.randint(0, 1)
        adj_init_1[i, 29] = c
        adj_init_1[29, i] = c
        
    random.seed(42)
    
    # Run SA
    adj_res, energy_res = _sa_search(adj_init_1, deadline)
    if energy_res == 0:
        colors = []
        for i in range(n_target):
            for j in range(i+1, n_target):
                colors.append(adj_res[i, j])
        return {'n': n_target, 'colors': colors}
        
    # If time remains, try random initialization
    if time.time() < deadline - 5:
        adj_init_2 = np.random.randint(0, 2, (n_target, n_target))
        adj_init_2 = np.triu(adj_init_2, 1)
        adj_init_2 = adj_init_2 + adj_init_2.T
        np.fill_diagonal(adj_init_2, 0)
        
        adj_res2, energy_res2 = _sa_search(adj_init_2, deadline)
        if energy_res2 == 0:
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj_res2[i, j])
            return {'n': n_target, 'colors': colors}
            
    # Fallback
    return {'n': 29, 'colors': colors_29}
