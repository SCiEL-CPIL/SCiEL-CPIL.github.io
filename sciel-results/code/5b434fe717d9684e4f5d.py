"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute edge indices
    edge_idx_map = {}
    for i, (u, v) in enumerate(edges):
        edge_idx_map[(u, v)] = i
        edge_idx_map[(v, u)] = i
        
    def spectral_radius(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift_fast(signs):
        """
        Returns True if the 2-lift is connected (signed graph is unbalanced).
        """
        labels = [-1] * n
        labels[0] = 0
        queue = [0]
        
        while queue:
            u = queue.pop()
            for i, (a, b) in enumerate(edges):
                if a == u:
                    v = b
                elif b == u:
                    v = a
                else:
                    continue
                
                s = signs[i]
                # x_u + x_v = 1 if s == -1 else 0
                # x_v = x_u XOR (1 if s==-1 else 0)
                target_x_v = labels[u] ^ (1 if s == -1 else 0)
                
                if labels[v] == -1:
                    labels[v] = target_x_v
                    queue.append(v)
                else:
                    if labels[v] != target_x_v:
                        return True # Contradiction found -> Unbalanced -> Connected
        
        # If no contradiction, it is Balanced -> Disconnected
        return False

    def is_valid_lift(signs):
        try:
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, float('inf')

    # Initial candidates
    best_signs = None
    best_radius = float('inf')
    
    guesses = []
    
    # Guess 1: All -1
    guesses.append(np.ones(num_edges, dtype=int) * -1)
    
    # Guess 2: Matching -1, Cycles +1
    s = np.ones(num_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            s[i] = -1
    guesses.append(s)
    
    # Guess 3: Matching +1, Cycles -1
    s = np.ones(num_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            s[i] = 1
        else:
            s[i] = -1
    guesses.append(s)

    # Guess 4: Alternating on cycles
    for top_start in [-1, 1]:
        for bot_start in [-1, 1]:
            s = np.ones(num_edges, dtype=int)
            for i, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    s[i] = -1
                elif u < m and v < m:
                    if v == (u + 1) % m:
                        pos = u
                    else:
                        pos = u
                    s[i] = top_start * ((-1) ** pos)
                elif u >= m and v >= m:
                    u_local = u - m
                    v_local = v - m
                    if v_local == (u_local + 1) % m:
                        pos = u_local
                    else:
                        pos = u_local
                    s[i] = bot_start * ((-1) ** pos)
            guesses.append(s)

    # Evaluate initial guesses
    for s in guesses:
        if is_connected_lift_fast(s):
            rad = spectral_radius(s)
            if rad < best_radius:
                best_radius = rad
                best_signs = s.copy()

    if best_signs is None:
        # Fallback: Find any valid one
        for s in guesses:
             if is_connected_lift_fast(s):
                 best_signs = s
                 best_radius = spectral_radius(s)
                 break
        if best_signs is None:
             # Should not happen, but if it does, return all -1
             best_signs = np.ones(num_edges, dtype=int) * -1
             best_radius = spectral_radius(best_signs)

    # Simulated Annealing
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    T_start = 0.5
    T_end = 0.01
    cooling_rate = 0.998
    T = T_start
    max_iterations = 20000 # More iterations with fast check
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_iterations):
        if T < T_end:
            break
            
        idx = rng.integers(0, num_edges)
        new_signs = current_signs.copy()
        new_signs[idx] *= -1
        
        if not is_connected_lift_fast(new_signs):
            T *= cooling_rate
            continue
            
        new_radius = spectral_radius(new_signs)
        
        delta = new_radius - current_radius
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_signs = new_signs
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        
        T *= cooling_rate

    # Local Search Polish
    improved = True
    max_iter = 500
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(num_edges):
            trial = best_signs.copy()
            trial[i] *= -1
            if not is_connected_lift_fast(trial):
                continue
            rad = spectral_radius(trial)
            if rad < best_radius:
                best_radius = rad
                best_signs = trial
                improved = True
                
    # Final check
    valid, final_rad = is_valid_lift(best_signs)
    if not valid:
        # If somehow invalid, fallback to a known valid pattern
        for s in guesses:
            if is_connected_lift_fast(s):
                best_signs = s
                break

    return best_signs
# EVOLVE-BLOCK-END
