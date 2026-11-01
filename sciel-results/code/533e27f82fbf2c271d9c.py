import time
import random
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

    def thue_morse(n):
        """Compute the Thue-Morse sequence value at position n."""
        # Count bits in binary representation
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count % 2

    def thue_morse_array(length, offset=0, stride=1):
        """Generate Thue-Morse values for indices offset, offset+stride, ..."""
        return [thue_morse(offset + i * stride) for i in range(length)]

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
            
        # Try Thue-Morse based patterns with different strides and offsets
        # Thue-Morse has optimal low-dispersion, should minimize short cycle collisions
        
        # Strategy: use Thue-Morse to generate base sequence, then scale to [0, m)
        # Try multiple strides to get different dispersion patterns
        
        found = False
        for stride in [1, 2, 3, 5, 7, 11, 13]:
            if time.time() - start_time > 105:
                break
            for offset in range(0, 10):
                if time.time() - start_time > 105:
                    break
                # Generate Thue-Morse values and map to [0, m)
                tm_vals = thue_morse_array(num_edges, offset=offset, stride=stride)
                # Scale to [0, m) using a simple hash-like mapping
                # Use the Thue-Morse bit pattern to create a pseudo-random but deterministic mapping
                voltages = []
                for i in range(num_edges):
                    # Combine Thue-Morse value with position for better dispersion
                    # Use a simple linear congruential approach seeded by Thue-Morse
                    seed_val = (tm_vals[i] * 2**20 + i * stride + offset) % (m * 3 + 1)
                    voltages.append(seed_val % m)
                
                result = build_and_check(m, voltages, g)
                if result:
                    return result
        
        # If Thue-Morse patterns didn't work, try pure Thue-Morse scaled
        # Map Thue-Morse bits directly: use multiple bits for higher dispersion
        for bit_start in range(0, 8):
            if time.time() - start_time > 105:
                break
            voltages = []
            for i in range(num_edges):
                # Use 4 consecutive Thue-Morse bits starting at bit_start + i
                val = 0
                for b in range(4):
                    val = (val << 1) | thue_morse(bit_start + i * 4 + b)
                voltages.append(val % m)
            
            result = build_and_check(m, voltages, g)
            if result:
                return result
        
        # Fallback: try a few random patterns as backup
        num_tries = 100
        for t in range(num_tries):
            if time.time() - start_time > 105:
                break
            seed_val = (m * 100000 + t)
            rng = np.random.default_rng(seed_val)
            voltages = rng.integers(0, m, size=num_edges).tolist()
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
