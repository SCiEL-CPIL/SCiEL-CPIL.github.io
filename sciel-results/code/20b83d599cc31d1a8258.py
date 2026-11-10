import time
import random
import numpy as np

N = 31
M = 91  # number of edges

def count_c4(adj_list):
    """Count 4-cycles in a graph given adjacency list."""
    n = len(adj_list)
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        adj_set[i] = set(adj_list[i])
    
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj_set[u] & adj_set[v])
            if common >= 2:
                c4_count += common * (common - 1) // 2
    return c4_count

def polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5) which has 31 vertices and 90 edges,
    with very few C4s.
    """
    points = []
    reps = {}
    idx = 0
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = pow(a, 4, 5)
                    rep = (1, b * inv_a % 5, c * inv_a % 5)
                elif b != 0:
                    inv_b = pow(b, 4, 5)
                    rep = (0, 1, c * inv_b % 5)
                else:
                    rep = (0, 0, 1)
                
                if rep not in reps:
                    reps[rep] = idx
                    points.append(rep)
                    idx += 1
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    edges = []
    for i in range(31):
        for j in range(i + 1, 31):
            a, b, c = points[i]
            d, e, f = points[j]
            val = (a * e + b * d + c * f) % 5
            if val == 0:
                edges.append((i, j))
    
    return edges

def delta_add_edge(u, v, adj_set, edge_set):
    """Count new C4s created by adding edge (u,v)."""
    new_c4s = 0
    for a in adj_set[v]:
        for b in adj_set[u]:
            if a == b:
                continue
            if (min(a, b), max(a, b)) in edge_set:
                new_c4s += 1
    return new_c4s

def count_c4s_containing_edge(u, v, adj_set, edge_set):
    """Count C4s that contain edge (u,v)."""
    count = 0
    for a in adj_set[u]:
        if a == v:
            continue
        for b in adj_set[v]:
            if b == u or b == a:
                continue
            if (min(a, b), max(a, b)) in edge_set:
                count += 1
    return count

def local_search_improve(edges, time_budget, rng):
    """
    Improve edge set via local search: try swapping edges to reduce C4 count.
    Uses fast delta computation with simulated annealing for escape.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_edges = list(edges)
    current_edge_set = set(current_edges)
    current_adj = [set() for _ in range(N)]
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
    
    current_c4s = count_c4([list(s) for s in current_adj])
    best_c4s = current_c4s
    best_edges = list(current_edges)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Simulated annealing parameters
    temperature = 5.0
    cooling_rate = 0.995
    min_temperature = 0.1
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline and temperature > min_temperature:
        made_progress = False
        no_improve_in_pass = 0
        
        # Shuffle edges for diversity
        shuffled_indices = list(range(len(current_edges)))
        rng.shuffle(shuffled_indices)
        
        for idx in shuffled_indices:
            if time.time() > deadline:
                break
            
            u1, v1 = current_edges[idx]
            u1n = min(u1, v1)
            v1n = max(u1, v1)
            
            # Count C4s containing edge (u1,v1)
            lost = count_c4s_containing_edge(u1, v1, current_adj, current_edge_set)
            
            # Try to find a non-edge that, when added, creates fewer C4s than lost
            best_gain = float('inf')
            best_edge = None
            
            # Sample a subset of non-edges for speed, but also try all if time permits
            non_edges = [(u2, v2) for u2, v2 in all_edges if (u2, v2) not in current_edge_set]
            
            # If we have time, try all; otherwise sample
            if len(non_edges) < 200 or rng.random() < 0.3:
                candidates = non_edges
            else:
                candidates = [non_edges[rng.randint(0, len(non_edges)-1)] for _ in range(200)]
            
            for u2, v2 in candidates:
                gain = delta_add_edge(u2, v2, current_adj, current_edge_set)
                if gain < best_gain:
                    best_gain = gain
                    best_edge = (u2, v2)
            
            if best_edge is not None:
                u2, v2 = best_edge
                u2n = min(u2, v2)
                v2n = max(u2, v2)
                
                delta = best_gain - lost
                
                # Acceptance criterion: simulated annealing
                if delta < 0:
                    accept = True
                else:
                    if temperature > min_temperature:
                        accept = rng.random() < np.exp(-delta / temperature)
                    else:
                        accept = False
                
                if accept:
                    # Perform swap
                    current_edges[idx] = (u2, v2)
                    current_edge_set.discard((u1n, v1n))
                    current_edge_set.add((u2n, v2n))
                    current_adj[u1].discard(v1)
                    current_adj[v1].discard(u1)
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                    
                    current_c4s += delta
                    made_progress = True
                    
                    if current_c4s < best_c4s:
                        best_c4s = current_c4s
                        best_edges = list(current_edges)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    break  # Restart scan after accepting a move
        
        if not made_progress:
            no_improve_count += 5
            if no_improve_count > max_no_improve:
                # Random perturbation to escape local minimum
                for _ in range(5):
                    idx = rng.randint(0, len(current_edges) - 1)
                    u1, v1 = current_edges[idx]
                    u1n, v1n = min(u1, v1), max(u1, v1)
                    
                    non_edges = [(u, v) for u, v in all_edges if (u, v) not in current_edge_set]
                    if non_edges:
                        u2, v2 = non_edges[rng.randint(0, len(non_edges)-1)]
                        u2n, v2n = min(u2, v2), max(u2, v2)
                        
                        lost = count_c4s_containing_edge(u1, v1, current_adj, current_edge_set)
                        gain = delta_add_edge(u2, v2, current_adj, current_edge_set)
                        
                        current_edges[idx] = (u2, v2)
                        current_edge_set.discard((u1n, v1n))
                        current_edge_set.add((u2n, v2n))
                        current_adj[u1].discard(v1)
                        current_adj[v1].discard(u1)
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
                        current_c4s += gain - lost
                        
                        if current_c4s < best_c4s:
                            best_c4s = current_c4s
                            best_edges = list(current_edges)
                
                no_improve_count = 0
                temperature = max(temperature * cooling_rate, min_temperature)
            else:
                temperature = max(temperature * cooling_rate, min_temperature)
        else:
            temperature = max(temperature * cooling_rate, min_temperature)
    
    return best_edges

def construct():
    """
    Main entry point: construct a graph with 91 edges on 31 vertices minimizing C4 count.
    Strategy: Start with polarity graph of PG(2,5) (90 edges, very few C4s),
    then greedily add the best edge, then do local search with simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Step 1: Construct polarity graph
    pol_edges = polarity_graph_q5()
    
    # Step 2: If we have exactly 90 edges, greedily add 1 more
    if len(pol_edges) == 90:
        edge_set = set(pol_edges)
        adj = [set() for _ in range(N)]
        for u, v in pol_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Greedily add the edge that minimizes new C4s
        best_edge = None
        best_new = float('inf')
        
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in edge_set:
                    continue
                new_c4s = delta_add_edge(u, v, adj, edge_set)
                if new_c4s < best_new:
                    best_new = new_c4s
                    best_edge = (u, v)
        
        if best_edge:
            final_edges = list(pol_edges) + [best_edge]
        else:
            final_edges = list(pol_edges)
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) not in edge_set:
                        final_edges.append((u, v))
                        break
                if len(final_edges) > len(pol_edges):
                    break
    elif len(pol_edges) > 90:
        final_edges = list(pol_edges[:90])
    else:
        final_edges = list(pol_edges)
        edge_set = set(final_edges)
        for u in range(N):
            for v in range(u + 1, N):
                if len(final_edges) >= 91:
                    break
                if (u, v) not in edge_set:
                    final_edges.append((u, v))
                    edge_set.add((u, v))
            if len(final_edges) >= 91:
                break
    
    # Ensure exactly 91 edges
    if len(final_edges) != 91:
        edge_set = set(final_edges)
        if len(final_edges) < 91:
            for u in range(N):
                for v in range(u + 1, N):
                    if len(final_edges) >= 91:
                        break
                    if (u, v) not in edge_set:
                        final_edges.append((u, v))
                        edge_set.add((u, v))
                if len(final_edges) >= 91:
                    break
        else:
            final_edges = final_edges[:91]
    
    # Step 3: Local search with simulated annealing to improve
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        improved_edges = local_search_improve(final_edges, remaining_time, rng)
        if improved_edges and len(improved_edges) == 91:
            final_edges = improved_edges
    
    # Convert to required format: list of [u, v]
    result = [[u, v] for u, v in final_edges]
    
    # Final validation
    if len(result) != 91:
        result = []
        edge_set = set()
        for u in range(N):
            for v in range(u + 1, N):
                if len(result) >= 91:
                    break
                result.append([u, v])
                edge_set.add((u, v))
            if len(result) >= 91:
                break
    
    return result
