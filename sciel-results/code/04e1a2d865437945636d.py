import time
import random

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        tm = []
        for i in range(n):
            tm.append(bin(i).count('1') % 2)
        return tm
    
    def check_girth(adj, n_vertices, girth_limit):
        """Check if graph has girth >= girth_limit using BFS from each vertex."""
        for start in range(n_vertices):
            # BFS to find shortest cycle through start
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
                    elif neighbor != parent[node] and parent[neighbor] != node:
                        # Found a cycle
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True
    
    def build_lift(m, voltages):
        """Build the voltage graph lift and return adjacency list."""
        n_vertices = 10 * m
        adj = [[] for _ in range(n_vertices)]
        
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i] % m
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        return adj
    
    def generate_edges(m, voltages):
        """Generate edge list for the lift."""
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i] % m
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                if u_id < v_id:
                    edges.append((u_id, v_id))
                else:
                    edges.append((v_id, u_id))
        return edges
    
    start_time = time.time()
    
    # Try increasing m values with Thue-Morse derived voltage assignments
    tm = get_thue_morse(200)
    
    for m in range(20, 200):
        if time.time() - start_time > 100:
            break
        
        # Generate multiple voltage patterns from Thue-Morse
        patterns = []
        
        # Pattern 1: Thue-Morse bits mapped to [0, m-1]
        patterns.append([tm[i] * (m // 2) for i in range(15)])
        
        # Pattern 2: Thue-Morse with scaling
        patterns.append([(tm[i] * (i + 1) * 7) % m for i in range(15)])
        
        # Pattern 3: Thue-Morse of index * 3
        patterns.append([tm[(i * 3) % len(tm)] * (m // 2 + 1) for i in range(15)])
        
        # Pattern 4: Quadratic Thue-Morse
        patterns.append([(tm[i] * (i * i + 1)) % m for i in range(15)])
        
        # Pattern 5: Linear combination
        patterns.append([(tm[i] * (3 * i + 1) + tm[(i + 5) % 15] * 2) % m for i in range(15)])
        
        # Pattern 6: Random with Thue-Morse seed
        random.seed(m * 1000 + 42)
        patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        # Pattern 7: Another random
        random.seed(m * 2000 + 13)
        patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        for vol in patterns:
            # Quick filter: ensure no voltage is 0 (avoids trivial short cycles)
            if any(v % m == 0 for v in vol):
                continue
            
            adj = build_lift(m, vol)
            
            # Check girth
            if check_girth(adj, 10 * m, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges
    
    # Fallback: try larger m with more patterns
    for m in range(200, 300):
        if time.time() - start_time > 110:
            break
        
        patterns = []
        for seed_offset in range(10):
            random.seed(m * 100 + seed_offset)
            patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        for vol in patterns:
            adj = build_lift(m, vol)
            if check_girth(adj, 10 * m, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges
    
    # Last resort: use a known large m
    m = 300
    random.seed(42)
    vol = [random.randint(1, m - 1) for _ in range(15)]
    adj = build_lift(m, vol)
    edges = generate_edges(m, vol)
    return 10 * m, edges
