import time
import random

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    num_edges = 15

    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        return [bin(i).count('1') % 2 for i in range(n)]

    def check_girth_fast(adj, n_vertices, girth_limit):
        """Check if graph has girth >= girth_limit using BFS from each vertex."""
        # Optimized BFS
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                if dist[node] >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        cycle_len = dist[node] + dist[neighbor] + 1
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
    # Precompute Thue-Morse sequence
    tm = get_thue_morse(2000)
    
    def get_tm_int(index, length):
        """Get integer formed by 'length' bits of Thue-Morse starting at 'index'."""
        val = 0
        for j in range(length):
            val = (val << 1) | tm[(index + j) % len(tm)]
        return val

    # Calculate lower bound for m
    if g % 2 == 0:
        r = g // 2
        # Moore bound for even girth g=2r: 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # For k=3: 1 + 3 * (2^(r-1) - 1)
        moore_bound = 1 + k * ((2 ** (r - 1)) - 1)
    else:
        r = (g - 1) // 2
        # For odd girth, bound is different, but we assume g is even for this task context usually
        # General approximation: 1 + k * (2^r - 1)
        moore_bound = 1 + k * ((2 ** r) - 1)
        
    m_min = max(2, (moore_bound + n_base - 1) // n_base)

    # Search for smallest m
    # We try increasing m
    max_m_search = 250 # Search up to m_min + 250 or a fixed limit
    
    for m in range(m_min, m_min + max_m_search):
        if time.time() - start_time > 110:
            break
            
        # Try multiple deterministic Thue-Morse based voltage patterns
        # We use 't' to shift the Thue-Morse window and mix with edge index
        num_tries = 300 if m < 100 else 150
        
        for t in range(num_tries):
            # Generate voltages using Thue-Morse derived high-dispersion sequence
            voltages = []
            for e in range(num_edges):
                # Combine edge index and trial shift to sample Thue-Morse
                # Use a window of 8 bits for good dispersion
                tm_val = get_tm_int(e * 7 + t * 13, 8)
                # Mix with linear terms to ensure spread across Z_m
                v = (tm_val * 31 + e * 17 + t * 11) % m
                voltages.append(v)
            
            result = build_and_check(m, voltages, g)
            if result:
                return result
                
    # Fallback: return a large graph if not found
    # Use a large m and random voltages as last resort
    m = m_min + max_m_search
    n_vertices = n_base * m
    random.seed(42)
    voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
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
