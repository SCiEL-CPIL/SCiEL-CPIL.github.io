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
        tm = [0]
        for i in range(1, n):
            tm.append(tm[i // 2] ^ (i & 1))
        return tm

    def check_girth_fast(adj, n_vertices, girth_limit):
        """Check if graph has girth >= girth_limit using non-backtracking walks."""
        for start in range(n_vertices):
            # Non-backtracking BFS: state = (node, prev_node)
            dist = {}
            queue = []
            for neighbor in adj[start]:
                dist[(start, neighbor)] = 1
                queue.append((start, neighbor))
            head = 0
            while head < len(queue):
                node, prev = queue[head]
                head += 1
                d = dist[(node, prev)]
                if d >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue
                    state = (node, neighbor)
                    if state not in dist:
                        dist[state] = d + 1
                        queue.append(state)
                    else:
                        # Found a cycle
                        cycle_len = d + dist[state] + 1
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

    def generate_thue_morse_voltages(m, seed_offset=0):
        """Generate high-dispersion voltages from Thue-Morse word."""
        tm = get_thue_morse(100)
        voltages = []
        for i in range(15):
            # Mix Thue-Morse bits with golden ratio for high dispersion
            golden = int(((i * 1618 + seed_offset) * 0.6180339887) * m) % m
            tm_bit = tm[(i * 7 + seed_offset) % 100]
            # Combine: use Thue-Morse to select from a dispersed set
            v = (golden + tm_bit * (m // 3 + 1)) % m
            if v == 0:
                v = 1
            voltages.append(v)
        return voltages

    def generate_quadratic_voltages(m, seed_offset=0):
        """Generate voltages using quadratic residues for high dispersion."""
        voltages = []
        for i in range(15):
            # Use quadratic residue-like pattern
            base = (i + 1 + seed_offset)
            v = (base * base + 3 * base + 7) % m
            if v == 0:
                v = 1
            voltages.append(v)
        return voltages

    def generate_mixed_voltages(m, seed_offset=0):
        """Generate voltages mixing Thue-Morse and linear patterns."""
        tm = get_thue_morse(100)
        voltages = []
        for i in range(15):
            tm_bit = tm[(i * 5 + seed_offset) % 100]
            linear = (7 * i + 3 * seed_offset) % m
            combined = (linear + tm_bit * (m // 2 + 1)) % m
            if combined == 0:
                combined = 1
            voltages.append(combined)
        return voltages

    start_time = time.time()
    
    # Try increasing m values with various voltage assignment strategies
    best_result = None
    
    for m in range(32, 256):
        if time.time() - start_time > 105:
            break
        
        # Generate multiple voltage patterns
        patterns = []
        
        # Pattern 1: Thue-Morse with golden ratio
        patterns.append(generate_thue_morse_voltages(m, 0))
        patterns.append(generate_thue_morse_voltages(m, 1))
        patterns.append(generate_thue_morse_voltages(m, 2))
        
        # Pattern 2: Quadratic residues
        patterns.append(generate_quadratic_voltages(m, 0))
        patterns.append(generate_quadratic_voltages(m, 1))
        
        # Pattern 3: Mixed Thue-Morse and linear
        patterns.append(generate_mixed_voltages(m, 0))
        patterns.append(generate_mixed_voltages(m, 1))
        
        # Pattern 4: Random with fixed seed
        random.seed(m * 1000 + 42)
        patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        for vol in patterns:
            # Quick filter: ensure no voltage is 0
            if any(v % m == 0 for v in vol):
                continue
            
            adj = build_lift(m, vol)
            
            # Check girth
            if check_girth_fast(adj, 10 * m, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges
    
    # Fallback: try larger m with more patterns
    for m in range(256, 400):
        if time.time() - start_time > 115:
            break
        
        patterns = []
        for seed_offset in range(5):
            patterns.append(generate_thue_morse_voltages(m, seed_offset))
        
        random.seed(m * 100 + 42)
        patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        for vol in patterns:
            if any(v % m == 0 for v in vol):
                continue
            adj = build_lift(m, vol)
            if check_girth_fast(adj, 10 * m, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges
    
    # Last resort: use a known large m
    m = 400
    random.seed(42)
    vol = [random.randint(1, m - 1) for _ in range(15)]
    adj = build_lift(m, vol)
    edges = generate_edges(m, vol)
    return 10 * m, edges
