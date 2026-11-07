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
    
    def compute_radius_fast(signs, S=None):
        if S is None:
            S = np.zeros((n, n))
        else:
            S[:] = 0
        S[u_idx, v_idx] = signs
        S[v_idx, u_idx] = signs
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals)), S

    def flip_and_eval(signs, i, S):
        """Flip sign i, return new radius and updated S"""
        old = signs[i]
        signs[i] = -old
        u, v = edges[i]
        S[u, v] = signs[i]
        S[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(S)
        r = np.max(np.abs(eigvals))
        return r, S

    def flip_back(signs, i, S, old):
        """Restore sign i"""
        signs[i] = old
        u, v = edges[i]
        S[u, v] = old
        S[v, u] = old

    start_time = time.time()
    deadline = start_time + 55
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    if m <= 5:
        # Brute force for small m
        S = np.zeros((n, n))
        for mask in range(2 ** num_edges):
            if time.time() > deadline:
                break
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(num_edges)])
            S[:] = 0
            S[u_idx, v_idx] = signs
            S[v_idx, u_idx] = signs
            eigvals = np.linalg.eigvalsh(S)
            r = np.max(np.abs(eigvals))
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    else:
        # Simulated Annealing with Local Search Polish
        S = np.zeros((n, n))
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            current_signs = rng.choice([-1, 1], size=num_edges)
            S[:] = 0
            S[u_idx, v_idx] = current_signs
            S[v_idx, u_idx] = current_signs
            eigvals = np.linalg.eigvalsh(S)
            current_radius = np.max(np.abs(eigvals))
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
                
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            
            while T > min_T and time.time() < deadline:
                i = rng.integers(0, num_edges)
                old_sign = current_signs[i]
                
                # Flip and evaluate
                current_signs[i] = -old_sign
                u, v = edges[i]
                S[u, v] = current_signs[i]
                S[v, u] = current_signs[i]
                eigvals = np.linalg.eigvalsh(S)
                new_radius = np.max(np.abs(eigvals))
                
                delta = new_radius - current_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
                    S[u, v] = old_sign
                    S[v, u] = old_sign
                    
                T *= cooling
                
            # Polish with local search - single flips
            improved = True
            polish_pass = 0
            while improved and time.time() < deadline:
                improved = False
                polish_pass += 1
                for i in range(num_edges):
                    old_sign = current_signs[i]
                    current_signs[i] = -old_sign
                    u, v = edges[i]
                    S[u, v] = current_signs[i]
                    S[v, u] = current_signs[i]
                    eigvals = np.linalg.eigvalsh(S)
                    new_radius = np.max(np.abs(eigvals))
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = current_signs.copy()
                    else:
                        current_signs[i] = old_sign
                        S[u, v] = old_sign
                        S[v, u] = old_sign
                
                # Add 2-flip moves after first pass
                if polish_pass >= 2:
                    for i in range(num_edges):
                        if time.time() > deadline:
                            break
                        for j in range(i+1, num_edges):
                            old_i = current_signs[i]
                            old_j = current_signs[j]
                            current_signs[i] = -old_i
                            current_signs[j] = -old_j
                            ui, vi = edges[i]
                            uj, vj = edges[j]
                            S[ui, vi] = current_signs[i]
                            S[vi, ui] = current_signs[i]
                            S[uj, vj] = current_signs[j]
                            S[vj, uj] = current_signs[j]
                            eigvals = np.linalg.eigvalsh(S)
                            new_radius = np.max(np.abs(eigvals))
                            if new_radius < current_radius:
                                current_radius = new_radius
                                improved = True
                                if current_radius < best_radius:
                                    best_radius = current_radius
                                    best_signs = current_signs.copy()
                            else:
                                current_signs[i] = old_i
                                current_signs[j] = old_j
                                S[ui, vi] = old_i
                                S[vi, ui] = old_i
                                S[uj, vj] = old_j
                                S[vj, uj] = old_j

    # Final high-precision polish on best_signs
    if best_signs is not None and time.time() < deadline:
        S = np.zeros((n, n))
        S[u_idx, v_idx] = best_signs
        S[v_idx, u_idx] = best_signs
        eigvals = np.linalg.eigvalsh(S)
        final_radius = np.max(np.abs(eigvals))
        
        # Try single flips one more time
        for i in range(num_edges):
            if time.time() > deadline:
                break
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            u, v = edges[i]
            S[u, v] = best_signs[i]
            S[v, u] = best_signs[i]
            eigvals = np.linalg.eigvalsh(S)
            new_radius = np.max(np.abs(eigvals))
            if new_radius < final_radius:
                final_radius = new_radius
                best_radius = new_radius
            else:
                best_signs[i] = old_sign
                S[u, v] = old_sign
                S[v, u] = old_sign

    return best_signs
