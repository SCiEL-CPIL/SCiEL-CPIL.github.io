import numpy as np
import time
def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    n = graph['n']
    
    # Pre-compute indices for faster matrix construction
    u_idx = np.array([e[0] for e in edges])
    v_idx = np.array([e[1] for e in edges])
    
    def compute_radius_fast(signs):
        S = np.zeros((n, n))
        S[u_idx, v_idx] = signs
        S[v_idx, u_idx] = signs
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    start_time = time.time()
    deadline = start_time + 55
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    if m <= 5:
        for mask in range(2 ** num_edges):
            if time.time() > deadline:
                break
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(num_edges)])
            r = compute_radius_fast(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    else:
        # Simulated Annealing with Local Search Polish
        for restart in range(30):
            if time.time() > deadline:
                break
            
            current_signs = rng.choice([-1, 1], size=num_edges)
            current_radius = compute_radius_fast(current_signs)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
                
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            
            while T > min_T and time.time() < deadline:
                i = rng.integers(0, num_edges)
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                new_radius = compute_radius_fast(current_signs)
                
                delta = new_radius - current_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
                    
                T *= cooling
                
            # Polish with local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(num_edges):
                    old_sign = current_signs[i]
                    current_signs[i] = -old_sign
                    new_radius = compute_radius_fast(current_signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = current_signs.copy()
                    else:
                        current_signs[i] = old_sign

    return best_signs
