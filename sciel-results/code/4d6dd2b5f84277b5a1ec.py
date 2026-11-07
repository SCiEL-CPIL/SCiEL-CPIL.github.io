"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    deadline = time.time() + 55  # safety margin before 60s
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def random_edges(n, m):
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        return [list(e) for e in all_edges[:m]]
    
    def local_search(values, edges, max_iters=5000):
        n = P['n']
        m = P['m']
        all_edges = list(combinations(range(n), 2))
        edge_set = set(tuple(e) for e in edges)
        current_score = evaluate(values, edges)
        best_score = current_score
        best_edges = [list(e) for e in edges]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random edge in current set
            idx = random.randint(0, m - 1)
            old_edge = tuple(edges[idx])
            # Pick a random edge not in current set
            new_edge = random.choice(all_edges)
            if new_edge in edge_set:
                continue
            
            # Swap
            edges[idx] = list(new_edge)
            edge_set.remove(old_edge)
            edge_set.add(new_edge)
            
            new_score = evaluate(values, edges)
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in edges]
            else:
                # Revert
                edges[idx] = list(old_edge)
                edge_set.remove(new_edge)
                edge_set.add(old_edge)
        
        return best_edges, best_score
    
    # Start with values 1..14
    values = list(range(1, 15))
    
    # Initial edges: greedily pick edges with smallest product diversity
    # Use the sum-15,16,17,... structure
    # Build candidate edges sorted by product
    all_edges = list(combinations(range(14), 2))
    # Sort by product
    all_edges_sorted = sorted(all_edges, key=lambda e: values[e[0]] * values[e[1]])
    
    # Greedy: pick edges that minimize new distinct sums and products
    edges = []
    edge_set = set()
    sums = set()
    prods = set()
    for e in all_edges_sorted:
        i, j = e
        s = values[i] + values[j]
        p = values[i] * values[j]
        edges.append([i, j])
        edge_set.add(e)
        sums.add(s)
        prods.add(p)
        if len(edges) == 35:
            break
    
    best_score = evaluate(values, edges)
    best_edges = [list(e) for e in edges]
    
    # Local search
    edges_ls, score_ls = local_search(values, edges, max_iters=20000)
    if score_ls < best_score:
        best_score = score_ls
        best_edges = edges_ls
    
    # Try a few random restarts if time permits
    restarts = 0
    while time.time() < deadline - 5:
        edges_r = random_edges(14, 35)
        edges_r_ls, score_r = local_search(values, edges_r, max_iters=5000)
        if score_r < best_score:
            best_score = score_r
            best_edges = edges_r_ls
        restarts += 1
        if restarts > 50:
            break
    
    # Ensure types are correct
    final_values = [int(v) for v in best_edges and values]  # just values
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
