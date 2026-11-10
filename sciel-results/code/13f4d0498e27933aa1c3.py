"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np
from collections import defaultdict

N = 31
E = 91
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def count_c4(adj_matrix):
    """Count the number of 4-cycles (subgraphs) in the graph."""
    # A 4-cycle is a-b-c-d-a with distinct vertices.
    # Count = (1/8) * sum over unordered pairs {u,v} of C(common_neighbors(u,v), 2)
    # Wait, that's not quite right. Let me think again.
    # 
    # Number of 4-cycles = (1/8) * sum_{u < v} C(deg_common(u,v), 2) * ... 
    # Actually, let me use the standard formula:
    # #C4 = (1/8) * sum_{u,v} C(N(u) ∩ N(v), 2)
    # where the sum is over all ordered pairs (u,v) with u ≠ v.
    # But we need to be careful about overcounting.
    #
    # Actually, the cleanest way: for each pair of vertices (u,v), let c(u,v) = |N(u) ∩ N(v)|.
    # The number of 4-cycles containing edge... no, let's just do it directly.
    #
    # A 4-cycle is determined by choosing 4 vertices and checking if they form a cycle.
    # For n=31, C(31,4) = 31465, which is very manageable.
    
    n = len(adj_matrix)
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Check all 3 ways to form a 4-cycle on these 4 vertices
                    # The 3 possible 4-cycles on vertices {a,b,c,d} are:
                    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
                    if (adj_matrix[a][b] and adj_matrix[b][c] and adj_matrix[c][d] and adj_matrix[d][a]):
                        count += 1
                    if (adj_matrix[a][b] and adj_matrix[b][d] and adj_matrix[d][c] and adj_matrix[c][a]):
                        count += 1
                    if (adj_matrix[a][c] and adj_matrix[c][b] and adj_matrix[b][d] and adj_matrix[d][a]):
                        count += 1
    return count


def count_c4_fast(adj_matrix):
    """Faster C4 count using the formula: #C4 = (1/8) * sum_{u<v} C(|N(u)∩N(v)|, 2) * 2
    Wait, let me derive this properly.
    
    Each 4-cycle a-b-c-d-a has two pairs of opposite vertices: {a,c} and {b,d}.
    For pair {a,c}, the common neighbors include b and d (and possibly others).
    So for each unordered pair {u,v}, the number of 4-cycles where u,v are opposite
    is C(|N(u)∩N(v)|, 2).
    
    Each 4-cycle is counted exactly once per pair of opposite vertices, and there are
    2 pairs of opposite vertices per 4-cycle. So total C4 = (1/2) * sum_{u<v} C(|N(u)∩N(v)|, 2).
    """
    n = len(adj_matrix)
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j]:
                adj[i].add(j)
                adj[j].add(i)
    
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj[u] & adj[v])
            if common >= 2:
                total += common * (common - 1) // 2
    return total // 2


def initial_graph():
    """Create an initial graph with 91 edges, designed to minimize C4."""
    # Start with a random graph, then try to improve
    adj = [[0] * N for _ in range(N)]
    edges = set()
    
    # Use a structured approach: start with a high-girth-like structure
    # For n=31, we can try to build something close to a polarity graph
    
    # Start with a random seed
    rng = random.Random(42)
    
    # Generate random edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs)
    
    for i, j in all_pairs[:E]:
        adj[i][j] = 1
        adj[j][i] = 1
        edges.add((i, j))
    
    return adj, edges


def local_search(adj, edges, time_limit):
    """Improve the graph using local search (edge swaps)."""
    best_adj = [row[:] for row in adj]
    best_edges = set(edges)
    best_c4 = count_c4_fast(best_adj)
    
    start_time = time.time()
    rng = random.Random(123)
    
    while time.time() - start_time < time_limit:
        # Try a random edge swap: remove edge (u,v), add edge (w,x)
        if len(edges) < 2:
            break
            
        # Pick a random edge to remove
        edge_list = list(edges)
        u, v = rng.choice(edge_list)
        
        # Pick a random non-edge to add
        # Sample until we find a non-edge
        w = rng.randint(0, N - 1)
        x = rng.randint(0, N - 1)
        if w == x:
            continue
        if w > x:
            w, x = x, w
        if (w, x) in edges:
            continue
        if (u, v) == (w, x):
            continue
        
        # Check if this swap keeps the graph valid (it does by construction)
        # Remove edge (u,v), add edge (w,x)
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((u, v))
        
        adj[w][x] = 1
        adj[x][w] = 1
        edges.add((w, x))
        
        new_c4 = count_c4_fast(adj)
        
        if new_c4 < best_c4:
            best_c4 = new_c4
            best_adj = [row[:] for row in adj]
            best_edges = set(edges)
        else:
            # Revert
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add((u, v))
            
            adj[w][x] = 0
            adj[x][w] = 0
            edges.discard((w, x))
        
        if time.time() - start_time > time_limit:
            break
    
    return best_adj, best_edges, best_c4


def simulated_annealing(adj, edges, time_limit):
    """Use simulated annealing to find a better graph."""
    best_adj = [row[:] for row in adj]
    best_edges = set(edges)
    best_c4 = count_c4_fast(best_adj)
    
    start_time = time.time()
    rng = random.Random(456)
    
    current_c4 = best_c4
    temperature = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Try a random edge swap
        if len(edges) < 2:
            break
            
        edge_list = list(edges)
        u, v = rng.choice(edge_list)
        
        w = rng.randint(0, N - 1)
        x = rng.randint(0, N - 1)
        if w == x:
            continue
        if w > x:
            w, x = x, w
        if (w, x) in edges:
            continue
        if (u, v) == (w, x):
            continue
        
        # Perform swap
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((u, v))
        
        adj[w][x] = 1
        adj[x][w] = 1
        edges.add((w, x))
        
        new_c4 = count_c4_fast(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-10)):
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = [row[:] for row in adj]
                best_edges = set(edges)
        else:
            # Revert
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add((u, v))
            
            adj[w][x] = 0
            adj[x][w] = 0
            edges.discard((w, x))
        
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
    
    return best_adj, best_edges, best_c4


def construct():
    start_time = time.time()
    
    # Phase 1: Generate multiple initial graphs and pick the best
    candidates = []
    num_initial = 5
    
    for seed in range(num_initial):
        if time.time() - start_time > DEADLINE - 60:
            break
        rng = random.Random(seed * 1000 + 42)
        adj = [[0] * N for _ in range(N)]
        edges = set()
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_pairs)
        
        for i, j in all_pairs[:E]:
            adj[i][j] = 1
            adj[j][i] = 1
            edges.add((i, j))
        
        c4 = count_c4_fast(adj)
        candidates.append((c4, adj, edges))
    
    # Sort by C4 count
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on the best candidates
    remaining_time = DEADLINE - (time.time() - start_time)
    time_per_candidate = remaining_time / max(1, min(len(candidates), 3))
    
    best_adj, best_edges, best_c4 = candidates[0][1], candidates[0][2], candidates[0][0]
    
    for i in range(min(3, len(candidates))):
        if time.time() - start_time > DEADLINE - 10:
            break
        c4, adj, edges = candidates[i]
        adj_copy = [row[:] for row in adj]
        edges_copy = set(edges)
        
        # Run local search
        ls_adj, ls_edges, ls_c4 = local_search(adj_copy, edges_copy, min(time_per_candidate, DEADLINE - (time.time() - start_time) - 5))
        
        if ls_c4 < best_c4:
            best_c4 = ls_c4
            best_adj = ls_adj
            best_edges = ls_edges
    
    # Phase 3: Simulated annealing from the best result
    if time.time() - start_time < DEADLINE - 15:
        sa_time = min(30, DEADLINE - (time.time() - start_time) - 5)
        sa_adj = [row[:] for row in best_adj]
        sa_edges = set(best_edges)
        
        sa_adj, sa_edges, sa_c4 = simulated_annealing(sa_adj, sa_edges, sa_time)
        
        if sa_c4 < best_c4:
            best_c4 = sa_c4
            best_adj = sa_adj
            best_edges = sa_edges
    
    # Convert to list of edges
    result = []
    for u, v in sorted(best_edges):
        result.append([u, v])
    
    # Ensure exactly 91 edges
    assert len(result) == E, f"Expected {E} edges, got {len(result)}"
    assert len(set(tuple(e) for e in result)) == E, "Duplicate edges found"
    
    return result
