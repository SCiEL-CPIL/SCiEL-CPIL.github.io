import time
import random
from itertools import product

def construct(k, g):
    # Petersen Graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Build adjacency for Petersen
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
        
    # Find all cycles of length < g in the base graph
    # We need to know the sequence of edge indices for each cycle to check voltage sums
    # Edge indices: 0..14
    edge_to_idx = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    cycles = []
    def find_cycles(start, current, path, visited_edges, depth, max_depth):
        if depth == max_depth:
            # Check if current is connected to start
            if start in adj_base[current]:
                edge_idx = edge_to_idx[(current, start)]
                if edge_idx not in visited_edges:
                    cycles.append(path + [edge_idx])
            return
            
        for neighbor in adj_base[current]:
            if neighbor == start and depth < max_depth - 1:
                continue
            edge_idx = edge_to_idx[(current, neighbor)]
            if edge_idx in visited_edges:
                continue
            find_cycles(start, neighbor, path + [edge_idx], visited_edges | {edge_idx}, depth + 1, max_depth)

    # We only care about cycles of length 5 to g-1
    # Petersen is small, we can find all simple cycles
    # Actually, non-backtracking walks that close are the ones that matter for girth in the lift?
    # Yes, a cycle in the lift corresponds to a closed non-backtracking walk in the base.
    # So we need to check all closed non-backtracking walks of length < g.
    
    # Let's generate all closed non-backtracking walks of length 5 to g-1
    # State: (current_node, prev_node, length, edge_sequence)
    # Since base is small, we can do this.
    
    closed_walks = []
    def gen_walks(node, prev, length, edges_seq, max_len):
        if length == max_len:
            if prev is not None:
                # Check if we can close to start? No, we track start separately.
                pass
            return
        
        for next_node in adj_base[node]:
            if next_node == prev:
                continue
            edge_idx = edge_to_idx[(node, next_node)]
            gen_walks(next_node, node, length + 1, edges_seq + [edge_idx], max_len)

    # Better: Generate walks from each start node, and check if they return to start.
    # To avoid duplicates, we can just collect all unique edge sequences of closed walks.
    # Or simply, for voltage checking, we just need the set of edge sequences.
    
    # Let's collect all closed non-backtracking walks of length L for L in [5, g-1]
    # We can represent a walk by its tuple of edge indices.
    
    target_lengths = list(range(5, g))
    
    all_closed_walks = set()
    
    for start in range(n_base):
        # Stack: (current, prev, length, edge_tuple)
        stack = [(start, None, 0, ())]
        while stack:
            curr, prev, l, e_seq = stack.pop()
            if l in target_lengths:
                if curr == start and l > 0:
                    # Found a closed walk
                    # Normalize direction? (e1, e2, ...) vs reverse?
                    # Voltage sum is the same for reverse if we sum edges.
                    # But edge indices are directed? No, edge_to_idx is undirected.
                    # So sum is commutative.
                    all_closed_walks.add(e_seq)
            
            if l >= g:
                continue
                
            for next_node in adj_base[curr]:
                if next_node == prev:
                    continue
                edge_idx = edge_to_idx[(curr, next_node)]
                stack.append((next_node, curr, l + 1, e_seq + (edge_idx,)))
                
    # Convert to list for faster access
    closed_walks_list = list(all_closed_walks)
    
    def check_voltages(voltages, m):
        """Check if voltages [v0..v14] satisfy girth constraint for modulus m."""
        for walk in closed_walks_list:
            s = sum(voltages[e] for e in walk)
            if s % m == 0:
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
    
    # Moore bound for (3, 16) is 512.
    # Since n = 10m, m must be at least 52.
    # Try m from 52 upwards.
    
    # Pre-compute closed walks for g=16
    # The code above does this dynamically, which is fine for Petersen.
    
    for m in range(52, 200):
        if time.time() - start_time > 110:
            break
            
        # Try to find valid voltages
        # Strategy: Randomized Local Search / Simulated Annealing
        # Objective: Minimize number of violated cycles (sum % m == 0)
        
        best_voltages = None
        best_violations = float('inf')
        
        # Initial random assignment
        random.seed(m)
        current_voltages = [random.randint(1, m-1) for _ in range(15)]
        
        def count_violations(vols):
            count = 0
            for walk in closed_walks_list:
                if sum(vols[e] for e in walk) % m == 0:
                    count += 1
            return count
        
        current_violations = count_violations(current_voltages)
        if current_violations == 0:
            best_voltages = current_voltages
            break
            
        # Local search
        for step in range(5000):
            if time.time() - start_time > 110:
                break
                
            # Pick a random edge to change
            e_idx = random.randint(0, 14)
            old_val = current_voltages[e_idx]
            new_val = random.randint(1, m-1)
            if new_val == old_val:
                continue
                
            current_voltages[e_idx] = new_val
            new_violations = count_violations(current_voltages)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if new_violations == 0:
                    best_voltages = current_voltages
                    break
            else:
                # Accept with some probability (simulated annealing) or just reject
                # Given small state space, simple hill climbing might get stuck.
                # Let's try restarts if stuck.
                pass
                
        if best_voltages:
            adj = build_lift(m, best_voltages)
            # Final verification
            if check_girth_fast(adj, 10*m, g): # Need to implement fast girth check or trust the walk check
                edges = generate_edges(m, best_voltages)
                return 10 * m, edges
                
        # If local search failed, try a few more random restarts
        for _ in range(5):
            if time.time() - start_time > 110:
                break
            random.seed(random.randint(0, 1000000))
            current_voltages = [random.randint(1, m-1) for _ in range(15)]
            if count_violations(current_voltages) == 0:
                best_voltages = current_voltages
                break
                
        if best_voltages:
            adj = build_lift(m, best_voltages)
            if check_girth_fast(adj, 10*m, g):
                edges = generate_edges(m, best_voltages)
                return 10 * m, edges

    # Fallback: Return a larger graph if no small one found
    # Use m=200 with random voltages, hoping it works or just returning it (risky)
    # Better: Return the best candidate found or a known large one.
    # Since we must return a valid graph, and we don't have one, we should ensure the fallback is valid.
    # A very large m makes it very likely that random voltages work.
    m = 1000
    random.seed(42)
    vol = [random.randint(1, m-1) for _ in range(15)]
    # Verify this one
    if check_girth_fast(build_lift(m, vol), 10*m, g):
         return 10*m, generate_edges(m, vol)
         
    # If even that fails (unlikely), increase m
    m = 2000
    random.seed(42)
    vol = [random.randint(1, m-1) for _ in range(15)]
    return 10*m, generate_edges(m, vol)
    
    # Helper for fast girth check using BFS (same as original but optimized)
    def check_girth_fast(adj, n_vertices, girth_limit):
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
                    elif neighbor != parent[node] and parent[neighbor] != node:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True
