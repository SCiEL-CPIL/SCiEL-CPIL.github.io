import random
import itertools
import time

def construct():
    """
    Constructs a graph with 14 vertices and 35 edges to minimize the max of distinct sums and products.
    """
    n = 14
    m = 35
    
    # Candidate value sets
    # We want distinct positive integers <= 10000.
    # Small numbers often yield more product collisions (e.g., 1*x, 2*x).
    # Consecutive numbers yield limited sum ranges.
    
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    candidates.append(list(range(5, n + 5)))
    candidates.append(list(range(10, n + 10)))
    
    # 2. Numbers with many common factors (e.g., multiples of 6)
    # 6, 12, 18, ...
    candidates.append([6 * i for i in range(1, n + 1)])
    candidates.append([12 * i for i in range(1, n + 1)])
    
    # 3. Mixed
    random.seed(42)
    for _ in range(5):
        candidates.append(random.sample(range(1, 1000), n))

    best_score = float('inf')
    best_edges = None
    best_values = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def local_search(values, max_time=10):
        # Initialize with a random set of 35 edges
        all_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(all_edges)
        current_edges = set(all_edges[:m])
        
        current_score = evaluate(values, current_edges)
        
        start_time = time.time()
        no_improvement_count = 0
        max_no_improvement = 5000
        
        while time.time() - start_time < max_time:
            # Generate a move: swap one edge in current with one not in current
            if not current_edges:
                break
            edge_in = random.choice(list(current_edges))
            # Pick a random edge not in current
            # To be efficient, we can just pick a random edge from all_edges and check if it's in current
            # Since m=35 and total=91, probability of collision is ~38%.
            # We can also maintain a list of available edges.
            
            # Let's just pick random pairs until we find one not in current
            while True:
                edge_out = tuple(sorted(random.sample(range(n), 2)))
                if edge_out not in current_edges:
                    break
            
            new_edges = set(current_edges)
            new_edges.remove(edge_in)
            new_edges.add(edge_out)
            
            new_score = evaluate(values, new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break
        
        return current_score, list(current_edges)

    for values in candidates:
        # Ensure values are distinct and <= 10000
        if len(set(values)) != n or max(values) > 10000:
            continue
            
        # Run local search
        score, edges = local_search(values, max_time=5)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = values
            
        # Early stopping if we hit a theoretical lower bound?
        # Min sums: 2 (if all pairs sum to same? impossible for distinct values > 0)
        # Min products: 1 (if all pairs product to same? impossible)
        # For 1..14, min sums is around 13-15, min products is larger.
        # If best_score is very low, we can stop.
        if best_score <= 15:
            break

    # Fallback if local search failed or wasn't run (shouldn't happen)
    if best_edges is None:
        best_values = list(range(1, n + 1))
        all_edges = list(itertools.combinations(range(n), 2))
        best_edges = all_edges[:m]

    # Convert to required format
    # values: list of ints
    # edges: list of lists [i, j]
    
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
