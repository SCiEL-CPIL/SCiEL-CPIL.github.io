import itertools
import random
import time

def construct(k, g):
    # Petersen Graph Definition
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def get_thue_morse(n):
        tm = [0]
        for i in range(1, n):
            tm.append(bin(i).count('1') % 2)
        return tm

    def check_girth(adj, n_vertices, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        for start in range(n_vertices):
            # BFS to find shortest cycle through start
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] >= min_girth - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent[node]:
                        continue
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    else:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def build_lift(voltages, m):
        """Build the lifted graph from voltage assignment."""
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

    def verify_and_return(voltages, m):
        """Verify the lift has girth >= g and return edges if valid."""
        adj = build_lift(voltages, m)
        if not check_girth(adj, 10 * m, g):
            return None
        
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
        
        return 10 * m, edges

    start_time = time.time()
    tm_sequence = get_thue_morse(200)
    
    # Try increasing m values, searching for valid voltage assignments
    for m in range(8, 100):
        if time.time() - start_time > 115:
            break
        
        # Generate diverse voltage patterns
        candidates = []
        
        # Pattern 1: Direct Thue-Morse scaled
        candidates.append([tm_sequence[i] * (m // 2 + 1) % m for i in range(15)])
        
        # Pattern 2: Thue-Morse with linear offset
        candidates.append([(tm_sequence[i] * (i + 1) * (m // 2 + 1)) % m for i in range(15)])
        
        # Pattern 3: Thue-Morse with quadratic term
        candidates.append([(tm_sequence[i] * (i * i + 1) * (m // 2 + 1)) % m for i in range(15)])
        
        # Pattern 4: Randomized Thue-Morse
        random.seed(m * 7 + 3)
        candidates.append([(tm_sequence[i] * random.randint(1, m - 1)) % m for i in range(15)])
        
        # Pattern 5: Pure random
        candidates.append([random.randint(0, m - 1) for _ in range(15)])
        
        # Pattern 6: Thue-Morse with alternating signs
        candidates.append([(tm_sequence[i] * (i % 2 + 1) * (m // 2 + 1)) % m for i in range(15)])
        
        # Try each candidate
        for vol in candidates:
            result = verify_and_return(vol, m)
            if result:
                return result
        
        # If simple patterns fail, try local search for this m
        if time.time() - start_time < 100:
            # Start from best random initialization
            random.seed(m)
            best_vol = [random.randint(0, m - 1) for _ in range(15)]
            
            # Local search: try to improve girth by changing individual voltages
            for iteration in range(50):
                if time.time() - start_time > 110:
                    break
                
                improved = False
                for i in range(15):
                    for new_val in range(m):
                        if new_val == best_vol[i]:
                            continue
                        test_vol = best_vol.copy()
                        test_vol[i] = new_val
                        result = verify_and_return(test_vol, m)
                        if result:
                            return result
                        # Keep track of best improvement (we'd need girth metric here)
                        # For now, just accept if it doesn't make things worse
                        # This is a simplification; a proper local search would track girth
                        if not improved:
                            best_vol[i] = new_val
                            improved = True
                    if improved:
                        break

    # Fallback: L-U-W construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
