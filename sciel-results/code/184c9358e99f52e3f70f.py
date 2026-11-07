import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    def get_score(values, edges):
        """Calculate the raw score: max(distinct_sums, distinct_prods)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edge_selection(values, max_edges=35):
        """
        Greedily select edges to minimize the distinct sums and products.
        We prioritize edges that produce sums/products already seen.
        """
        selected_edges = []
        seen_sums = set()
        seen_prods = set()
        
        # Pre-calculate all possible edge sums and products
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Heuristic: Sort edges by how "common" their sum and product are?
        # Or just iterate and pick the one that adds the least new distinct values.
        # Since m=35 is small relative to 91, we can do a simple greedy.
        
        # To make it more robust, we can try a few different greedy orders.
        # Order 1: Random
        # Order 2: Smallest sums first
        # Order 3: Smallest products first
        
        best_edges = None
        best_score = float('inf')
        
        for strategy in ['random', 'sum_asc', 'prod_asc']:
            local_edges = []
            local_sums = set()
            local_prods = set()
            
            if strategy == 'random':
                candidates = edge_data[:]
                random.shuffle(candidates)
            elif strategy == 'sum_asc':
                candidates = sorted(edge_data, key=lambda x: x[2])
            else: # prod_asc
                candidates = sorted(edge_data, key=lambda x: x[3])
            
            for u, v, s, p in candidates:
                if len(local_edges) >= max_edges:
                    break
                
                new_sums = 0 if s in local_sums else 1
                new_prods = 0 if p in local_prods else 1
                
                # We want to minimize the increase in max(len(sums), len(prods))
                # Simple heuristic: prefer edges that don't add new distinct values
                # If both add new, prefer the one that adds to the smaller set?
                
                # Let's just add it if it doesn't increase the max too much,
                # or if we need more edges.
                # A better heuristic: 
                # Cost = max(len(local_sums) + new_sums, len(local_prods) + new_prods)
                # But we are picking one by one.
                
                # Simple greedy: Add edge if it minimizes the immediate increase in the union size?
                # Actually, just adding all 35 in the sorted order is a valid strategy.
                # The "greedy" part is that we stop at 35.
                
                local_edges.append((u, v))
                local_sums.add(s)
                local_prods.add(p)
            
            score = max(len(local_sums), len(local_prods))
            if score < best_score:
                best_score = score
                best_edges = local_edges
        
        return best_edges, best_score

    def generate_value_candidates():
        """
        Generate diverse value sets designed to create collisions.
        """
        candidates = []
        
        # 1. Geometric Progression (Powers of 2)
        # Sums: 2^a + 2^b. Products: 2^(a+b).
        # Distinct sums are unique for each pair (a,b) if base is large enough?
        # 1+2=3, 1+4=5, 2+4=6...
        # Products: 2, 4, 8...
        # This might not be great for collisions.
        
        # 2. Arithmetic Progression
        # v_i = 1 + i * d
        # Sums: 2 + (i+j)d. Products: (1+id)(1+jd) = 1 + (i+j)d + ijd^2.
        
        # 3. Clusters of identical values (but they must be distinct!)
        # We need 14 DISTINCT integers.
        # So we can't have identical values.
        # But we can have values that are very close, e.g., 10, 11, 12...
        
        # 4. Divisors of a highly composite number
        # If all values divide N, then products divide N^2? Not necessarily helpful for distinct counts.
        
        # 5. Random small integers
        # 6. Random geometric
        # 7. Random arithmetic
        
        # Strategy: Generate a batch of candidate value sets, run greedy edge selection on each,
        # and keep the best.
        
        # Candidate Set 1: Random 1..100
        for _ in range(20):
            candidates.append(random.sample(range(1, 101), n))
            
        # Candidate Set 2: Random 1..50
        for _ in range(20):
            candidates.append(random.sample(range(1, 51), n))
            
        # Candidate Set 3: Geometric-ish (Powers of 2 with noise)
        for _ in range(10):
            vals = []
            used = set()
            for i in range(n):
                base = 2 ** i
                # Add small noise to make distinct and break pure power structure?
                # Or just use powers of 2?
                # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
                # Sum collisions: 1+8=9, 3+6? No, 3 is not a power of 2.
                # 1+2=3, 1+4=5, 2+4=6.
                # 1+8=9, 4+5? No.
                # With powers of 2, all sums are unique?
                # 2^a + 2^b. If a<b, the binary representation has bits a and b set.
                # So all sums are unique.
                # Products: 2^(a+b). All products are unique.
                # So powers of 2 are BAD.
                pass
            
        # Candidate Set 4: Arithmetic Progression
        for start in range(1, 10):
            for d in range(1, 10):
                vals = [start + i * d for i in range(n)]
                if max(vals) <= 10000 and len(set(vals)) == n:
                    candidates.append(vals)
                    
        # Candidate Set 5: Randomly clustered
        # Split 14 into 3 groups, assign close values to each group
        for _ in range(20):
            groups = [5, 5, 4]
            vals = []
            used = set()
            for g in groups:
                base = random.randint(10, 100)
                for _ in range(g):
                    v = base + random.randint(-5, 5)
                    while v in used or v <= 0:
                        v = random.randint(1, 200)
                    vals.append(v)
                    used.add(v)
            candidates.append(vals)
            
        return candidates

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    random.seed(42)
    
    # Generate candidates
    candidates = generate_value_candidates()
    
    # Shuffle to mix strategies
    random.shuffle(candidates)
    
    # Process each candidate
    # We have ~70 candidates.
    # Greedy edge selection is fast (91 edges, sort, pick 35).
    # We can afford to process many.
    
    for vals in candidates:
        if time.time() > deadline:
            break
            
        # Ensure distinct and valid
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        edges, score = greedy_edge_selection(vals)
        
        if score < best_global_score:
            best_global_score = score
            best_global_values = list(vals)
            best_global_edges = edges
            
        # Optional: Local search on the best found so far?
        # If we have time, we can try to improve the best candidate further.
        
    # If we have time, run a local search on the best candidate found
    if time.time() < deadline - 10.0 and best_global_values is not None:
        # Local Search: Swap values or swap edges
        current_values = list(best_global_values)
        current_edges = set(best_global_edges)
        current_score = best_global_score
        
        # Simple SA for a short time
        T = 1.0
        for i in range(5000):
            if time.time() > deadline - 5.0:
                break
                
            T = T * 0.99
            move_type = random.random()
            
            new_values = current_values
            new_edges = current_edges
            
            if move_type < 0.5:
                # Edge Swap
                if len(current_edges) > 0:
                    e_out = random.choice(list(current_edges))
                    e_in = random.choice(all_edges)
                    if e_in not in current_edges:
                        new_edges = (current_edges - {e_out}) | {e_in}
            else:
                # Value Swap
                i1, i2 = random.sample(range(n), 2)
                new_values = list(current_values)
                new_values[i1], new_values[i2] = new_values[i2], new_values[i1]
                
            new_score = get_score(new_values, new_edges)
            
            if new_score <= current_score or random.random() < math.exp(-(new_score - current_score) / max(T, 1e-9)):
                current_values = new_values
                current_edges = new_edges
                current_score = new_score
                
                if current_score < best_global_score:
                    best_global_score = current_score
                    best_global_values = current_values
                    best_global_edges = current_edges
                    
    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = list(all_edges[:35])
        best_global_score = get_score(best_global_values, best_global_edges)
        
    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
