import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    values = list(range(1, 15))
    n = 14
    
    all_edges = list(itertools.combinations(range(n), 2))
    edge_data = []
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data.append((u, v, s, p))
        
    num_edges = len(all_edges)
    required_edges = 35
    
    def evaluate(edge_set):
        sums = set()
        products = set()
        for idx in edge_set:
            _, _, s, p = edge_data[idx]
            sums.add(s)
            products.add(p)
        return max(len(sums), len(products))
    
    best_score = float('inf')
    best_edges = set()
    rng = random.Random(42)
    
    # Try a few different initial configurations
    # 1. Lowest sums
    # 2. Lowest products
    # 3. Random
    # 4. Mixed
    
    initial_strategies = []
    
    # Strategy 1: Lowest sums
    idxs = list(range(num_edges))
    sorted_by_sum = sorted(idxs, key=lambda i: (edge_data[i][2], edge_data[i][3]))
    initial_strategies.append(set(sorted_by_sum[:required_edges]))
    
    # Strategy 2: Lowest products
    sorted_by_prod = sorted(idxs, key=lambda i: (edge_data[i][3], edge_data[i][2]))
    initial_strategies.append(set(sorted_by_prod[:required_edges]))
    
    # Strategy 3: Random
    initial_strategies.append(set(rng.sample(idxs, required_edges)))
    
    # Strategy 4: Low "cost" (sum + product)
    sorted_by_cost = sorted(idxs, key=lambda i: (edge_data[i][2] + edge_data[i][3]))
    initial_strategies.append(set(sorted_by_cost[:required_edges]))
    
    for init_edges in initial_strategies:
        if time.time() > deadline:
            break
            
        current_edges = set(init_edges)
        current_score = evaluate(current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(current_edges)
            
        # Local Search
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            if not current_edges:
                break
            remove_idx = rng.choice(list(current_edges))
            add_idx = rng.randint(0, num_edges - 1)
            if add_idx == remove_idx:
                continue
                
            new_edges = set(current_edges)
            new_edges.remove(remove_idx)
            new_edges.add(add_idx)
            
            new_score = evaluate(new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
            else:
                no_improve_count += 1
                # Occasional kick to escape local minima
                if no_improve_count % 100 == 0:
                    # Random swap regardless of score
                    current_edges = new_edges
                    current_score = new_score
                    no_improve_count = 0

    final_edges = []
    for idx in best_edges:
        u, v, _, _ = edge_data[idx]
        final_edges.append([u, v])
        
    return {'values': values, 'edges': final_edges}
