import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Exploit the prism graph's cyclic symmetry to reduce search space.
    2. For each edge type (top cycle, bottom cycle, matching), parameterize signs
       as a function of position: s_type(i) = sign_type * (-1)^(i mod period)
    3. This reduces the search from 2^(3m) to a much smaller combinatorial space.
    4. Use exhaustive search for small m, then simulated annealing for larger m.
    """
    
    deadline = time.time() + 85.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Identify edge types:
    # Top cycle: edges between vertices 0..m-1 (consecutive)
    # Bottom cycle: edges between vertices m..2m-1 (consecutive)
    # Matching: edges (i, i+m)
    
    # Precompute which edge is which type
    top_cycle_edges = []  # (edge_idx, position)
    bottom_cycle_edges = []  # (edge_idx, position)
    matching_edges = []  # (edge_idx, position)
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle edge
            pos = min(u, v)
            top_cycle_edges.append((i, pos))
        elif u >= m and v >= m:
            # Bottom cycle edge
            pos = min(u - m, v - m)
            bottom_cycle_edges.append((i, pos))
        else:
            # Matching edge
            pos = min(u, v) if min(u, v) < m else min(u - m, v - m)
            matching_edges.append((i, pos))
    
    # Verify we have the right structure
    assert len(top_cycle_edges) == m
    assert len(bottom_cycle_edges) == m
    assert len(matching_edges) == m
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        This is equivalent to the signing NOT being a coboundary.
        """
        f = [-1] * n
        f[0] = 0
        
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        
        return False

    def get_radius(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            val = float(signs[i])
            S[u, v] = val
            S[v, u] = val
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    def generate_signs_from_pattern(top_sign, bottom_sign, match_sign, top_period, bottom_period, match_period):
        """
        Generate signs from a symmetric pattern.
        top_period, bottom_period, match_period: 1 or 2 (whether to alternate)
        """
        signs = np.ones(num_edges, dtype=np.int8)
        
        for edge_idx, pos in top_cycle_edges:
            if top_period == 1:
                signs[edge_idx] = top_sign
            else:
                signs[edge_idx] = top_sign * ((-1) ** pos)
        
        for edge_idx, pos in bottom_cycle_edges:
            if bottom_period == 1:
                signs[edge_idx] = bottom_sign
            else:
                signs[edge_idx] = bottom_sign * ((-1) ** pos)
        
        for edge_idx, pos in matching_edges:
            if match_period == 1:
                signs[edge_idx] = match_sign
            else:
                signs[edge_idx] = match_sign * ((-1) ** pos)
        
        return signs
    
    best_signs = None
    best_radius = np.inf
    
    # Strategy 1: Exhaustive search over symmetric patterns
    # Parameters:
    # top_sign: +1 or -1
    # bottom_sign: +1 or -1  
    # match_sign: +1 or -1
    # top_period: 1 (constant) or 2 (alternating)
    # bottom_period: 1 or 2
    # match_period: 1 or 2
    
    # Total: 2^3 * 2^3 = 64 patterns
    
    for top_sign in [-1, 1]:
        for bottom_sign in [-1, 1]:
            for match_sign in [-1, 1]:
                for top_period in [1, 2]:
                    for bottom_period in [1, 2]:
                        for match_period in [1, 2]:
                            if time.time() > deadline:
                                break
                            
                            signs = generate_signs_from_pattern(
                                top_sign, bottom_sign, match_sign,
                                top_period, bottom_period, match_period
                            )
                            
                            if not is_lift_connected(signs):
                                continue
                            
                            radius = get_radius(signs)
                            
                            if radius < best_radius:
                                best_radius = radius
                                best_signs = signs.copy()
    
    # Strategy 2: More general patterns - allow different phases
    # For each cycle, the sign pattern can be:
    # - Constant: all +1 or all -1
    # - Alternating: (-1)^i or (-1)^(i+1)
    # But also consider shifted patterns
    
    # Actually, for a cycle of length m, the possible sign patterns that respect
    # the cyclic symmetry are limited. Let's try a broader search.
    
    # Strategy 3: Simulated annealing with symmetry-aware initialization
    
    rng = np.random.default_rng(3)
    
    # Start from the best symmetric pattern found
    if best_signs is not None:
        signs = best_signs.copy()
    else:
        # Random start
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        while not is_lift_connected(signs):
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
    
    current_radius = get_radius(signs)
    if current_radius < best_radius:
        best_radius = current_radius
        best_signs = signs.copy()
    
    # SA Parameters
    T0 = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    steps_per_temp = 50
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    T = T0
    while T > T_end and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            # Pick random edge to flip
            idx = rng.integers(0, num_edges)
            old_sign = signs[idx]
            new_sign = -old_sign
            
            signs[idx] = new_sign
            
            if not is_lift_connected(signs):
                signs[idx] = old_sign
                continue
            
            S = np.zeros((n, n), dtype=np.float64)
            S[u_arr, v_arr] = signs.astype(float)
            S[v_arr, u_arr] = signs.astype(float)
            new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
            
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = signs.copy()
            else:
                signs[idx] = old_sign
        
        T *= cooling_rate
        
        # Local search polish
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                old_sign = signs[i]
                signs[i] = -old_sign
                if not is_lift_connected(signs):
                    signs[i] = old_sign
                    continue
                S = np.zeros((n, n), dtype=np.float64)
                S[u_arr, v_arr] = signs.astype(float)
                S[v_arr, u_arr] = signs.astype(float)
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
                    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1

    return best_signs
