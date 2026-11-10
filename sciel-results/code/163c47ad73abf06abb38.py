import time
import random
from itertools import combinations

def construct():
    # 1. Define Values
    # Using multiples of 6 to encourage product collisions
    values = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
    n = 14
    
    # Precompute all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
            
    # 2. Greedy Initialization
    # We want to pick 35 edges.
    # Start with empty set
    current_edges = set()
    current_sums = set()
    current_prods = set()
    
    # To make greedy efficient, we can just iterate through all edges and pick the "best" one
    # "Best" means it adds the fewest new distinct sums/prods.
    # Tie-breaker: prefer higher frequency in the global pool? Or just random among ties.
    
    # A better greedy: 
    # 1. Calculate global frequency of each sum and product.
    sum_freq = {}
    prod_freq = {}
    for _, _, s, p in all_edges:
        sum_freq[s] = sum_freq.get(s, 0) + 1
        prod_freq[p] = prod_freq.get(p, 0) + 1
        
    # Sort edges by a score that favors "common" sums and products
    # Score = sum_freq[s] * prod_freq[p]
    scored_edges = []
    for i, j, s, p in all_edges:
        score = sum_freq[s] * prod_freq[p]
        scored_edges.append((score, i, j, s, p))
        
    # Sort by score descending
    scored_edges.sort(key=lambda x: -x[0])
    
    # Pick top 35? This might be too correlated.
    # Let's try a more dynamic greedy.
    
    def get_score(edges_set):
        sums = set()
        prods = set()
        for i, j, s, p in edges_set:
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))
        
    # Greedy Step-by-Step
    selected = []
    remaining = list(all_edges)
    
    for _ in range(35):
        best_edge = None
        best_score = float('inf')
        
        # To speed up, we can just check the top candidates from the static sort
        # But dynamic is better. 35 * 91 is small.
        for e in remaining:
            i, j, s, p = e
            # Tentatively add
            new_sums = current_sums | {s}
            new_prods = current_prods | {p}
            new_score = max(len(new_sums), len(new_prods))
            
            if new_score < best_score:
                best_score = new_score
                best_edge = e
                best_sums = new_sums
                best_prods = new_prods
            elif new_score == best_score:
                # Tie break: prefer edges with higher static score?
                # Or just random
                pass
                
        if best_edge:
            selected.append(best_edge)
            current_sums = best_sums
            current_prods = best_prods
            remaining.remove(best_edge)
            
    # 3. Local Search
    # Try to swap edges to improve score
    # Represent state as a set of edge indices (0 to 90)
    
    edge_index_map = {e: idx for idx, e in enumerate(all_edges)}
    
    current_state = set(edge_index_map[e] for e in selected)
    
    def calc_score_from_state(state):
        sums = set()
        prods = set()
        for idx in state:
            _, _, s, p = all_edges[idx]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))
        
    current_score = calc_score_from_state(current_state)
    
    # Local Search Loop
    start_time = time.time()
    time_limit = 45.0 # Leave margin for 60s limit
    
    best_state = current_state
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        # Pick a random edge in current state
        if not current_state:
            break
        e_in = random.choice(list(current_state))
        
        # Pick a random edge NOT in current state
        # There are 91 total, 35 in state. 56 not in state.
        # We can maintain a list of available edges or just pick random and check
        
        e_out = None
        while e_out is None or e_out in current_state:
            e_out = random.randint(0, 90)
            
        new_state = set(current_state)
        new_state.remove(e_in)
        new_state.add(e_out)
        
        new_score = calc_score_from_state(new_state)
        
        # Accept if better or equal (to allow movement)
        # Or use Simulated Annealing logic
        if new_score <= current_score:
            current_state = new_state
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_state = set(current_state)
                
    # Reconstruct result from best_state
    final_edges = []
    for idx in best_state:
        i, j, _, _ = all_edges[idx]
        final_edges.append([i, j])
        
    return {
        'values': values,
        'edges': final_edges
    }
