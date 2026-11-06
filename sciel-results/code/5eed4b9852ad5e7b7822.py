import time
import random
import math

N = 20
TOTAL_EDGES = N * (N - 1) // 2  # 190
DEADLINE_MARGIN = 10.0

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Precomputation
    # Map edge index to (u, v)
    edge_to_uv = []
    uv_to_edge = {}
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            edge_to_uv.append((u, v))
            uv_to_edge[(u, v)] = idx
            uv_to_edge[(v, u)] = idx
            idx += 1
            
    # Helper to calculate penalty from counts
    def calc_penalty(counts):
        pen = 0
        for u in range(N):
            for v in range(u + 1, N):
                c = counts[u][v]
                if c > 2:
                    pen += c - 2
        return pen
        
    # Simulated Annealing with Bitmasks
    def run_sa(seed, time_limit):
        rng = random.Random(seed)
        
        # Initialize with a random valid graph or greedy
        # Let's use a greedy initialization to get a good start
        adj = [0] * N
        counts = [[0] * N for _ in range(N)]
        
        # Greedy initialization
        edges = list(range(TOTAL_EDGES))
        rng.shuffle(edges)
        
        for e in edges:
            u, v = edge_to_uv[e]
            # Check if adding (u,v) is valid
            # 1. c[u][v] <= 2 (already checked if we maintain counts, but initially 0)
            if counts[u][v] > 2:
                continue
            
            # Check affected pairs
            valid = True
            # Pairs (u, w) for w in N(v)
            # N(v) is adj[v]
            w_mask = adj[v]
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                if w == u:
                    continue
                # c[u][w] will increase by 1
                if counts[u][w] >= 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Pairs (v, w) for w in N(u)
            w_mask = adj[u]
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                if w == v:
                    continue
                # c[v][w] will increase by 1
                if counts[v][w] >= 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Update counts
            w_mask = adj[v] & ~(1 << u) # N(v) \ {u}
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                counts[u][w] += 1
                counts[w][u] += 1
                
            w_mask = adj[u] & ~(1 << v) # N(u) \ {v}
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                counts[v][w] += 1
                counts[w][v] += 1
        
        current_edges = sum(bin(adj[u]).count('1') for u in range(N)) // 2
        current_penalty = calc_penalty(counts)
        
        best_edges = current_edges if current_penalty == 0 else 0
        best_adj = adj[:]
        best_counts = [row[:] for row in counts]
        
        T = 10.0
        cooling_rate = 0.999
        lambda_pen = 100.0
        
        while time.time() < time_limit and T > 0.01:
            # Choose move type: 0=Add, 1=Remove, 2=Swap
            move_type = rng.random()
            if move_type < 0.4:
                move_type = 0 # Add
            elif move_type < 0.7:
                move_type = 1 # Remove
            else:
                move_type = 2 # Swap
                
            if move_type == 0: # Add
                # Pick a random non-existent edge
                e = rng.randint(0, TOTAL_EDGES - 1)
                u, v = edge_to_uv[e]
                if (adj[u] >> v) & 1:
                    continue
                
                # Calculate delta
                delta_penalty = 0
                # Affected: (u, w) for w in N(v) \ {u}
                w_mask = adj[v] & ~(1 << u)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    if counts[u][w] >= 2:
                        delta_penalty += 1
                # Affected: (v, w) for w in N(u) \ {v}
                w_mask = adj[u] & ~(1 << v)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    if counts[v][w] >= 2:
                        delta_penalty += 1
                
                delta_E = -1 + delta_penalty * lambda_pen
                
                if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                    # Accept
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    current_penalty += delta_penalty
                    
                    # Update counts
                    w_mask = adj[v] & ~(1 << u)
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        counts[u][w] += 1
                        counts[w][u] += 1
                        
                    w_mask = adj[u] & ~(1 << v)
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        counts[v][w] += 1
                        counts[w][v] += 1
                        
                    if current_penalty == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_counts = [row[:] for row in counts]
                        
            elif move_type == 1: # Remove
                # Pick a random existing edge
                e = rng.randint(0, TOTAL_EDGES - 1)
                u, v = edge_to_uv[e]
                if not ((adj[u] >> v) & 1):
                    continue
                
                # Calculate delta
                delta_penalty = 0
                # Affected: (u, w) for w in N(v) \ {u}
                w_mask = adj[v] & ~(1 << u)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    if counts[u][w] >= 3:
                        delta_penalty -= 1
                # Affected: (v, w) for w in N(u) \ {v}
                w_mask = adj[u] & ~(1 << v)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    if counts[v][w] >= 3:
                        delta_penalty -= 1
                        
                delta_E = 1 + delta_penalty * lambda_pen
                
                if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                    # Accept
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    current_penalty += delta_penalty
                    
                    # Update counts
                    w_mask = adj[v] & ~(1 << u)
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        counts[u][w] -= 1
                        counts[w][u] -= 1
                        
                    w_mask = adj[u] & ~(1 << v)
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        counts[v][w] -= 1
                        counts[w][v] -= 1
                        
                    if current_penalty == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_counts = [row[:] for row in counts]
                        
            else: # Swap
                # Pick two distinct edges, one existing, one non-existing
                e1 = rng.randint(0, TOTAL_EDGES - 1)
                e2 = rng.randint(0, TOTAL_EDGES - 1)
                if e1 == e2:
                    continue
                u1, v1 = edge_to_uv[e1]
                u2, v2 = edge_to_uv[e2]
                
                if not ((adj[u1] >> v1) & 1) or ((adj[u2] >> v2) & 1):
                    continue
                    
                # This is complex. For now, skip swap or implement carefully.
                # Given time constraints, Add/Remove might be sufficient with many restarts.
                # Let's just stick to Add/Remove for simplicity and robustness.
                pass
                
            T *= cooling_rate
            
        return best_edges, best_adj

    best_overall_edges = 0
    best_overall_adj = [0] * N
    
    seed = 42
    while time.time() < deadline:
        # Run SA for 5 seconds
        run_time = min(5.0, deadline - time.time())
        if run_time < 1.0:
            break
        m, adj = run_sa(seed, time.time() + run_time)
        if m > best_overall_edges:
            best_overall_edges = m
            best_overall_adj = adj
        seed += 1
        
    # Convert adj to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_overall_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
