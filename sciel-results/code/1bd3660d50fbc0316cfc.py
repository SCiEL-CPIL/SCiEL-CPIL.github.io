import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Precompute adjacency structure
    # For each edge, store (u, v) for quick matrix building
    edge_uv = [(u, v) for u, v in edges]
    
    def build_matrix(signs):
        A = np.zeros((n, n), dtype=float)
        for i in range(e_count):
            u, v = edge_uv[i]
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def compute_radius(signs):
        A = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Identify edge types
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # Generate diverse initial candidates
    candidate_signs_list = []
    rng = np.random.default_rng(42)
    
    # Structural patterns
    # Pattern 1: All +1
    s = np.ones(e_count, dtype=int)
    candidate_signs_list.append(s)
    
    # Pattern 2: All -1
    s = np.ones(e_count, dtype=int)
    s[:] = -1
    candidate_signs_list.append(s)
    
    # Pattern 3: Matching -1, cycles +1
    s = np.ones(e_count, dtype=int)
    for edge_idx in matching_edges:
        s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 4: Alternating matching
    s = np.ones(e_count, dtype=int)
    for i, edge_idx in enumerate(matching_edges):
        if i % 2 == 0:
            s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 5: Top cycle alternating
    s = np.ones(e_count, dtype=int)
    for i, edge_idx in enumerate(top_cycle_edges):
        if i % 2 == 0:
            s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 6: Bottom cycle alternating
    s = np.ones(e_count, dtype=int)
    for i, edge_idx in enumerate(bottom_cycle_edges):
        if i % 2 == 0:
            s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 7: All cycles -1, matching +1
    s = np.ones(e_count, dtype=int)
    for edge_idx in top_cycle_edges + bottom_cycle_edges:
        s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 8: Checkerboard on all edges (by vertex parity)
    s = np.ones(e_count, dtype=int)
    for i, (u, v) in enumerate(edges):
        if (u + v) % 2 == 0:
            s[i] = -1
    candidate_signs_list.append(s)
    
    # Random candidates
    for _ in range(20):
        s = rng.choice([-1, 1], size=e_count).astype(int)
        candidate_signs_list.append(s)
    
    # Evaluate initial candidates
    best_radius = float('inf')
    best_signs = np.ones(e_count, dtype=int)
    
    for s in candidate_signs_list:
        if time.time() > deadline:
            break
        try:
            r = compute_radius(s)
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
        except Exception:
            pass
    
    # Local search with multiple move types
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    def try_flip(signs, indices):
        """Try flipping signs at given indices, return new radius or None if invalid."""
        new_signs = signs.copy()
        for idx in indices:
            new_signs[idx] = -new_signs[idx]
        try:
            return compute_radius(new_signs)
        except Exception:
            return None
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # 1-flip moves
        for edge_idx in range(e_count):
            if time.time() > deadline:
                break
            r = try_flip(current_signs, [edge_idx])
            if r is not None and r < current_radius - 1e-12:
                current_signs[edge_idx] = -current_signs[edge_idx]
                current_radius = r
                best_signs = current_signs.copy()
                best_radius = r
                improved = True
                no_improve = 0
                break
        
        if improved:
            continue
        
        # 2-flip moves
        for i in range(e_count):
            if time.time() > deadline:
                break
            for j in range(i + 1, e_count):
                if time.time() > deadline:
                    break
                r = try_flip(current_signs, [i, j])
                if r is not None and r < current_radius - 1e-12:
                    current_signs[i] = -current_signs[i]
                    current_signs[j] = -current_signs[j]
                    current_radius = r
                    best_signs = current_signs.copy()
                    best_radius = r
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # 3-flip moves (sampled to avoid O(e^3))
        if no_improve % 3 == 0:
            num_attempts = min(50, (e_count * (e_count-1) * (e_count-2)) // 6)
            for _ in range(num_attempts):
                if time.time() > deadline:
                    break
                indices = rng.choice(e_count, size=3, replace=False)
                r = try_flip(current_signs, list(indices))
                if r is not None and r < current_radius - 1e-12:
                    for idx in indices:
                        current_signs[idx] = -current_signs[idx]
                    current_radius = r
                    best_signs = current_signs.copy()
                    best_radius = r
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            
            # Random restart from perturbed best
            if no_improve >= 10:
                perturbed = best_signs.copy()
                num_flips = rng.integers(1, 4)
                flip_indices = rng.choice(e_count, size=num_flips, replace=False)
                for idx in flip_indices:
                    perturbed[idx] = -perturbed[idx]
                r = compute_radius(perturbed)
                if r < current_radius:
                    current_signs = perturbed
                    current_radius = r
                    no_improve = 0
                else:
                    no_improve = 0  # Reset to continue searching
    
    return best_signs
