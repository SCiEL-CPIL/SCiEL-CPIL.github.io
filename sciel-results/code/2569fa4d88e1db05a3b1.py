import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all possible edges and triples
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    
    def get_score(adj_set):
        # adj_set is a set of (u, v) tuples with u < v
        deg = [0] * n
        for u, v in adj_set:
            deg[u] += 1
            deg[v] += 1
            
        max_sum = 0
        for i, j, k in all_triples:
            if (i, j) in adj_set and (j, k) in adj_set and (i, k) in adj_set:
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    def random_graph():
        edges = random.sample(all_edges, target_edges)
        return set(tuple(sorted(e)) for e in edges)

    def optimize(initial_adj_set):
        current_adj_set = set(initial_adj_set)
        current_score = get_score(current_adj_set)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        max_iters = 2000 # Reduced to allow more restarts
        
        best_local_score = current_score
        best_local_adj = set(current_adj_set)
        
        for _ in range(max_iters):
            if T < min_T:
                break
                
            # Pick a random edge to remove
            if not current_adj_set:
                break
            e_rem = random.choice(list(current_adj_set))
            
            # Pick a random edge to add
            while True:
                e_add = random.choice(all_edges)
                e_add = tuple(sorted(e_add))
                if e_add not in current_adj_set:
                    break
            
            # Create new set
            new_adj_set = set(current_adj_set)
            new_adj_set.remove(e_rem)
            new_adj_set.add(e_add)
            
            # Calculate new score
            new_score = get_score(new_adj_set)
            
            # Acceptance
            if new_score < current_score:
                current_adj_set = new_adj_set
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_adj = set(current_adj_set)
            else:
                delta = new_score - current_score
                if delta == 0:
                    current_adj_set = new_adj_set
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_adj_set = new_adj_set
                        current_score = new_score
            
            T *= cooling
        
        return best_local_adj, best_local_score

    best_global_adj = None
    best_global_score = float('inf')
    
    start_time = time.time()
    time_limit = 120.0 # Keep margin
    
    while time.time() - start_time < time_limit:
        initial_adj_set = random_graph()
        local_adj, local_score = optimize(initial_adj_set)
        
        if local_score < best_global_score:
            best_global_score = local_score
            best_global_adj = local_adj
            if best_global_score <= 23:
                break

    if best_global_adj is None:
        # Fallback to seed
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    else:
        edges = [list(e) for e in best_global_adj]
        
    return {'n': 16, 'edges': edges}
