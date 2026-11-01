import time
import numpy as np

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    num_edges = 15

    def check_girth_fast(adj, n_vertices, girth_limit):
        """Check if graph has girth >= girth_limit using BFS from each vertex."""
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                d = dist[node]
                if d >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = d + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        cycle_len = d + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True

    # Precompute Thue-Morse sequence
    # t(n) = popcount(n) % 2
    # We need a long sequence to allow for many shifts
    TM_LENGTH = 10000
    TM_SEQ = [bin(i).count('1') % 2 for i in range(TM_LENGTH)]

    def generate_voltages_thue_morse(m, trial_id):
        """
        Generate a voltage assignment for the 15 edges using Thue-Morse sequence.
        Uses 16-bit chunks for better dispersion.
        """
        voltages = []
        # Shift start index based on trial_id to get different patterns
        # Each edge uses 16 bits, so we consume 15*16 = 240 bits per trial
        # We can overlap or just shift. Shifting is cleaner.
        start_idx = (trial_id * 240) % TM_LENGTH
        
        for i in range(num_edges):
            idx = (start_idx + i * 16) % TM_LENGTH
            val = 0
            for j in range(16):
                val = (val << 1) | TM_SEQ[(idx + j) % TM_LENGTH]
            voltages.append(val % m)
        return voltages

    def build_and_check(m, voltages, girth_limit):
        """Build lift, check girth, return edges if valid."""
        n_vertices = n_base * m
        adj = [[] for _ in range(n_vertices)]
        
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i]
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        if check_girth_fast(adj, n_vertices, girth_limit):
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                w = voltages[i]
                for s in range(m):
                    u_id = u * m + s
                    v_id = v * m + (s + w) % m
                    if u_id < v_id:
                        edges.append((u_id, v_id))
                    else:
                        edges.append((v_id, u_id))
            return n_vertices, edges
        return None

    start_time = time.time()
    
    # Moore bound lower bound for n is 512.
    # n = 10 * m, so m >= 52.
    m_min = 52
    
    # Search for smallest m
    max_m_search = 200
    
    for m in range(m_min, m_min + max_m_search):
        if time.time() - start_time > 110:
            break
            
        # Try Thue-Morse derived patterns
        # Increase trials for smaller m where solution is more likely
        num_tries = 3000 if m < 60 else 1000
        
        for t in range(num_tries):
            if time.time() - start_time > 105:
                break
            
            voltages = generate_voltages_thue_morse(m, t)
            result = build_and_check(m, voltages, g)
            if result:
                return result
                
    # Fallback
    m = m_min + max_m_search
    n_vertices = n_base * m
    rng = np.random.default_rng(42)
    voltages = rng.integers(0, m, size=num_edges).tolist()
    edges = []
    for i, (u, v) in enumerate(petersen_edges):
        w = voltages[i]
        for s in range(m):
            u_id = u * m + s
            v_id = v * m + (s + w) % m
            if u_id < v_id:
                edges.append((u_id, v_id))
            else:
                edges.append((v_id, u_id))
    return n_vertices, edges
