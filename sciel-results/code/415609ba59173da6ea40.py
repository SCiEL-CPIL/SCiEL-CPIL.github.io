import time
import numpy as np
from collections import deque
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (4 calls)
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Pre-compute adjacency list for fast connectivity checks
    # adj[u] = list of (v, edge_index)
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    def is_connected_lift(signs):
        """
        Check if the 2-lift is connected.
        The lift is connected iff the sign assignment is "frustrated" (not gauge-equivalent to all +1).
        We check this by trying to assign a label to each vertex such that label[u] * label[v] = sign(edge).
        If such an assignment exists for all vertices, the lift is disconnected (two copies).
        If a contradiction is found, the lift is connected.
        """
        label = [0] * n
        label[0] = 1
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, i in adj[u]:
                s = signs[i]
                expected_v = s * label[u]
                
                if label[v] == 0:
                    label[v] = expected_v
                    queue.append(v)
                elif label[v] != expected_v:
                    # Contradiction found -> Lift is connected
                    return True
        
        # No contradiction -> Lift is disconnected (two components)
        return False

    def compute_radius(signs):
        """Compute spectral radius of the signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            S[u, v] = s
            S[v, u] = s
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def evaluate(signs):
        """
        Evaluate a sign vector.
        Returns (radius, is_valid).
        """
        # Fast connectivity check
        if not is_connected_lift(signs):
            return np.inf, False
        
        # Check 3-regularity (always true for prism base, but good to be safe? 
        # Actually, the base graph is fixed. The lift is always 3-regular if the base is 3-regular.
        # Prism graphs are 3-regular. So we just need connectivity.
        
        # Compute radius
        try:
            r = compute_radius(signs)
            return r, True
        except:
            return np.inf, False

    # --- Strategy ---
    # 1. Generate a diverse set of initial candidates.
    #    - Random samples
    #    - Structured patterns (alternating on cycles)
    # 2. Use local search (1-flip, then 2-flip) to improve.
    # 3. Use a "smooth" surrogate or just robust local search? 
    #    The directive suggests smooth surrogates, but for binary signs, 
    #    local search with connectivity constraints is standard.
    #    Let's stick to robust local search with multiple restarts.
    
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Helper to update best
    def try_update(signs):
        nonlocal best_signs, best_radius
        r, valid = evaluate(signs)
        if valid and r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            return True
        return False

    # Generate initial candidates
    candidates = []
    
    # 1. Random samples
    for _ in range(50):
        candidates.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))
    
    # 2. Structured patterns
    # Prism graph edges: 
    # Typically: 
    # 0..m-1: cycle 1 edges (0,1), (1,2)...(m-1,0)
    # m..2m-1: cycle 2 edges (m, m+1)...(2m-1, m)
    # 2m..3m-1: matching edges (i, i+m)
    # We don't know exact order, but we can try patterns based on parity of edge index 
    # or vertex indices.
    
    # Pattern: Alternate signs on all edges
    for phase in [0, 1]:
        signs = np.ones(num_edges, dtype=np.int8)
        for i in range(num_edges):
            if (i + phase) % 2 == 0:
                signs[i] = -1
        candidates.append(signs)
        
    # Pattern: Signs based on vertex parity
    # If edge connects (u,v), sign = 1 if u%2 == v%2 else -1 ?
    # This makes the lift bipartite?
    for phase in [0, 1]:
        signs = np.ones(num_edges, dtype=np.int8)
        for i, (u, v) in enumerate(edges):
            if ((u + phase) % 2) != ((v + phase) % 2):
                signs[i] = -1
        candidates.append(signs)
        
    # Pattern: Random with bias
    for _ in range(20):
        p = rng.uniform(0.2, 0.8)
        signs = np.where(rng.random(num_edges) < p, 1, -1).astype(np.int8)
        candidates.append(signs)

    # Evaluate all candidates
    for c in candidates:
        try_update(c)
        
    if best_signs is None:
        # Fallback: find ANY valid sign
        # Try single -1 flips from all +1 (which is disconnected)
        # All +1 is disconnected. Flipping one edge might make it connected?
        # If we flip one edge, is it connected?
        # In a 2-lift, if the base is connected, the lift is connected iff the product of signs 
        # around some cycle is -1.
        # If we have all +1, all cycle products are +1.
        # If we flip one edge, cycles containing that edge have product -1.
        # Does every cycle contain that edge? No.
        # So we need to ensure at least one cycle has product -1.
        # Actually, the condition for connectivity is that the "twist" is not trivial.
        # Let's just try random until we find one.
        for _ in range(1000):
            s = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            if try_update(s):
                break
        
        if best_signs is None:
            # Last resort: return a pattern that is likely connected
            # Alternate signs on the matching edges?
            signs = np.ones(num_edges, dtype=np.int8)
            # Assume last m edges are matching
            for i in range(n, n + n//2):
                if i < num_edges:
                    signs[i] = -1
            return signs

    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Local Search
    # 1-flip moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order
        order = np.random.permutation(num_edges)
        for i in order:
            if time.time() >= deadline:
                break
            new_signs = current_signs.copy()
            new_signs[i] *= -1
            r, valid = evaluate(new_signs)
            if valid and r < current_radius - 1e-9:
                current_signs = new_signs
                current_radius = r
                best_signs = new_signs.copy()
                best_radius = r
                improved = True
                # Don't break, keep searching for more improvements in this pass
                # Or break to restart? Hill climbing usually accepts first improvement or best improvement.
                # Let's accept first improvement and continue the pass.
        
    # 2-flip moves if stuck
    if time.time() < deadline:
        two_flip_improved = True
        while two_flip_improved and time.time() < deadline:
            two_flip_improved = False
            order = np.random.permutation(num_edges)
            for i in order:
                if time.time() >= deadline:
                    break
                for j in order:
                    if i == j:
                        continue
                    if time.time() >= deadline:
                        break
                    new_signs = current_signs.copy()
                    new_signs[i] *= -1
                    new_signs[j] *= -1
                    r, valid = evaluate(new_signs)
                    if valid and r < current_radius - 1e-9:
                        current_signs = new_signs
                        current_radius = r
                        best_signs = new_signs.copy()
                        best_radius = r
                        two_flip_improved = True
                        break
                if two_flip_improved:
                    break
    
    # Random restarts if time remains
    if time.time() < deadline:
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            # Generate a new random start
            new_start = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            
            # Quick local search from this start
            s = new_start
            r, valid = evaluate(s)
            if not valid:
                continue
                
            cur_r = r
            # Short hill climb
            for _ in range(100): # Limit steps per restart
                if time.time() >= deadline:
                    break
                improved_inner = False
                for i in range(num_edges):
                    if time.time() >= deadline:
                        break
                    ns = s.copy()
                    ns[i] *= -1
                    nr, v = evaluate(ns)
                    if v and nr < cur_r - 1e-9:
                        s = ns
                        cur_r = nr
                        improved_inner = True
                        break
                if not improved_inner:
                    break
                    
            if cur_r < best_radius:
                best_radius = cur_r
                best_signs = s.copy()
    
    return best_signs
