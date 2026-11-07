import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        num_lift_vertices = 2 * n
        parent = list(range(num_lift_vertices))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                union(u * 2, v * 2)
                union(u * 2 + 1, v * 2 + 1)
            else:
                union(u * 2, v * 2 + 1)
                union(u * 2 + 1, v * 2)
        root = find(0)
        for i in range(1, num_lift_vertices):
            if find(i) != root:
                return False
        return True
    
    rng = np.random.default_rng(3)
    
    # Strategy: The spectral radius is minimized when the signed graph has
    # eigenvalues as close to zero as possible. For a 3-regular signed graph,
    # the minimum possible spectral radius is related to the structure.
    
    # Key: for each cycle, having an odd number of -1 signs (product = -1)
    # shifts the eigenvalues away from ±2.
    
    # Let's try structured initial guesses:
    # 1. All +1
    # 2. One -1 in bottom cycle, rest +1
    # 3. One -1 in top cycle, rest +1
    # 4. Matching all -1
    # etc.
    
    best_signs = None
    best_radius = float('inf')
    
    def evaluate(signs):
        nonlocal best_signs, best_radius
        if not is_connected_lift(signs):
            return
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # Random starts
    for start in range(500):
        if time.time() > deadline - 15:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        evaluate(signs)
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = spectral_radius(best_signs)
    
    # Local search: flip pairs of signs to maintain connectivity
    improved = True
    while improved and (time.time() < deadline - 8):
        improved = False
        for i in range(num_edges):
            if time.time() > deadline - 8:
                break
            for j in range(i + 1, num_edges):
                trial = best_signs.copy()
                trial[i] *= -1
                trial[j] *= -1
                if is_connected_lift(trial):
                    r = spectral_radius(trial)
                    if r < best_radius - 1e-12:
                        best_signs = trial
                        best_radius = r
                        improved = True
                        break
            if time.time() > deadline - 8:
                break
    
    # Simulated annealing
    if time.time() < deadline - 10:
        current_signs = best_signs.copy()
        current_radius = best_radius
        T = 0.03
        cooling = 0.9995
        for iteration in range(200000):
            if time.time() > deadline - 5:
                break
            i = rng.integers(0, num_edges)
            trial = current_signs.copy()
            trial[i] *= -1
            if not is_connected_lift(trial):
                continue
            r = spectral_radius(trial)
            delta = r - current_radius
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_signs = trial
                current_radius = r
                if r < best_radius:
                    best_signs = trial
                    best_radius = r
            T *= cooling
    
    return best_signs
