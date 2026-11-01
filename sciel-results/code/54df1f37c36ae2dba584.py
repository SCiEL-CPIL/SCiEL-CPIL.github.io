import time
import random
from collections import deque

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        tm = [0] * n
        for i in range(1, n):
            tm[i] = bin(i).count('1') % 2
        return tm

    def check_girth_lift(m, voltages, girth_limit):
        """
        Check if the lift of Petersen graph with given voltages has girth >= girth_limit.
        """
        n_vertices = 10 * m
        adj = [[] for _ in range(n_vertices)]
        
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i] % m
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        # BFS from each vertex
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                d = dist[node]
                if d >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = d + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node] and parent[neighbor] != node:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True

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
    
    # Precompute Thue-Morse for a sufficient length
    tm = get_thue_morse(100)
    
    # We want to find smallest m.
    # Moore bound for n(3, 16) is 512.
    # n = 10 * m. So m >= 52.
    # Let's start from m=50 upwards.
    
    for m in range(50, 200):
        if time.time() - start_time > 110:
            break
            
        # Generate candidate voltage patterns
        candidates = []
        
        # Pattern 1: Thue-Morse scaled
        candidates.append([(tm[i] * (m // 2) + 1) % m for i in range(15)])
        
        # Pattern 2: Thue-Morse with index mixing
        candidates.append([((tm[i] * (i + 1)) + 1) % m for i in range(15)])
        
        # Pattern 3: Thue-Morse of shifted index
        candidates.append([(tm[(i+5) % 100] * (m // 2) + i) % m for i in range(15)])
        
        # Pattern 4: Random with fixed seed based on m
        random.seed(m * 12345)
        candidates.append([random.randint(1, m-1) for _ in range(15)])
        
        # Pattern 5: Random with different seed
        random.seed(m * 54321)
        candidates.append([random.randint(1, m-1) for _ in range(15)])
        
        # Pattern 6: Linear sequence
        candidates.append([(i * 7 + 3) % m for i in range(15)])
        
        # Pattern 7: Quadratic
        candidates.append([(i*i + i + 1) % m for i in range(15)])

        for vol in candidates:
            if check_girth_lift(m, vol, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges

    # Fallback if not found in range
    m = 94
    for seed in range(100):
        if time.time() - start_time > 115:
            break
        random.seed(seed)
        vol = [random.randint(1, m-1) for _ in range(15)]
        if check_girth_lift(m, vol, g):
            edges = generate_edges(m, vol)
            return 10 * m, edges
            
    # Last resort
    m = 300
    random.seed(42)
    vol = [random.randint(1, m-1) for _ in range(15)]
    edges = generate_edges(m, vol)
    return 10 * m, edges
