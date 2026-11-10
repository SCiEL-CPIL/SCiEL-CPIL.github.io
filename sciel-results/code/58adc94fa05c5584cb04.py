"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))

    # Helper to find the best 35 edges for a given set of values
    # We want to minimize max(|sums|, |prods|)
    # Strategy: Start with all edges, remove the "worst" ones? 
    # Or start with a random set and hill-climb?
    # Hill-climbing on edges is fast because n=14, m=35.
    
    def optimize_edges_for_values(values, time_limit=2.0):
        all_edges = list(combinations(range(n), 2))
        # Initialize with random edges
        current_edges = set(random.sample(all_edges, m))
        current_score = _raw_value(values, current_edges)
        best_edges = set(current_edges)
        best_score = current_score
        
        # Local search: swap one edge for another
        # To speed up, we can precompute sum/prod for all pairs
        sum_cache = {}
        prod_cache = {}
        for i, j in all_edges:
            sum_cache[(i,j)] = values[i] + values[j]
            prod_cache[(i,j)] = values[i] * values[j]
            
        def eval_edges(edge_set):
            sums = set()
            prods = set()
            for e in edge_set:
                sums.add(sum_cache[e])
                prods.add(prod_cache[e])
            return max(len(sums), len(prods))

        t_start = time.time()
        no_improve = 0
        while time.time() - t_start < time_limit:
            # Pick a random edge to remove
            e_out = random.choice(list(current_edges))
            # Pick a random edge to add (not in current)
            # Efficient way: pick random from all_edges until it's not in current_edges
            e_in = random.choice(all_edges)
            while e_in in current_edges:
                e_in = random.choice(all_edges)
                
            new_edges = set(current_edges)
            new_edges.remove(e_out)
            new_edges.add(e_in)
            
            new_score = eval_edges(new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
            else:
                # Accept with small probability to escape local minima?
                # Or just strict hill climb for speed?
                # Given small n, strict hill climb is usually sufficient for edges.
                # Let's use a small prob for noise to avoid getting stuck.
                if random.random() < 0.05:
                    current_edges = new_edges
                    current_score = new_score
                else:
                    no_improve += 1
                    if no_improve > 100:
                        # Restart edges locally?
                        current_edges = set(random.sample(all_edges, m))
                        current_score = eval_edges(current_edges)
                        no_improve = 0
                        
        return best_edges, best_score

    # Strategy for Values:
    # Generate diverse candidate value sets that are likely to have collisions.
    # 1. Arithmetic Progressions (AP): a, a+d, a+2d...
    # 2. Geometric Progressions (GP): a, a*r, a*r^2... (integers)
    # 3. Mixed: Small base, geometric growth.
    # 4. Random sets with small variance.
    
    candidate_value_sets = []
    
    # 1. APs with small d
    for d in range(1, 20):
        for a in range(1, 100):
            vals = [a + i*d for i in range(n)]
            if max(vals) <= maximum:
                # Ensure distinct (d>0 ensures this)
                candidate_value_sets.append(vals)
                
    # 2. GPs with small integer ratio
    for r in range(2, 10):
        for a in range(1, 10):
            vals = []
            valid = True
            for i in range(n):
                v = a * (r ** i)
                if v > maximum:
                    valid = False
                    break
                vals.append(v)
            if valid:
                candidate_value_sets.append(vals)
                
    # 3. "Quasi-GP" or structured randoms: 
    #    Values like 1, 2, 4, 8... but with some perturbations?
    #    Or values from a specific set like {1, 2, 3, 4, 6, 12, ...} (highly composite)
    #    Let's try sets based on divisors of highly composite numbers?
    #    Or just random sets but biased towards small numbers.
    
    # 4. Random small values
    for _ in range(50):
        vals = random.sample(range(1, 200), n)
        candidate_value_sets.append(vals)
        
    # 5. Specific known good structures?
    #    e.g. 1, 2, 4, 8, 16... is bad for sums (all distinct) but good for products (distinct).
    #    We want BOTH to collide.
    #    If values are close, sums collide. If values are small, products collide.
    #    So small values are key.
    
    # Deduplicate candidates
    unique_candidates = set()
    final_candidates = []
    for vals in candidate_value_sets:
        # Sort to canonicalize? No, indices matter for edges, but the set of values is what matters for sum/prod sets.
        # Actually, the *set* of values determines the possible sums and products.
        # The *indices* determine which pairs are edges.
        # So we can just use the list as is, but ensure we don't evaluate identical multisets repeatedly if possible.
        # Since n is small, just evaluate a diverse set.
        if tuple(sorted(vals)) not in unique_candidates:
            unique_candidates.add(tuple(sorted(vals)))
            final_candidates.append(vals)
            
    # Shuffle candidates to be fair
    random.shuffle(final_candidates)
    
    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    # Allocate time per candidate
    # We have ~85 seconds.
    # Let's evaluate candidates until time runs out.
    # We prioritize candidates with smaller max value, as they allow more product collisions.
    final_candidates.sort(key=lambda x: max(x))
    
    for vals in final_candidates:
        if time.time() > deadline - 10: # Stop 10s early to be safe
            break
            
        # Optimize edges for this value set
        # Give more time to promising candidates?
        # Let's give 2 seconds per candidate, or scale based on remaining time.
        # If many candidates, give less time to each.
        candidates_remaining = len(final_candidates)
        time_per_candidate = min(2.0, (deadline - 10 - time.time()) / max(1, candidates_remaining))
        
        if time_per_candidate < 0.1:
            # If time is tight, just do a quick greedy or skip?
            # Let's just do a very quick optimization
            time_per_candidate = 0.1
            
        best_edges, score = optimize_edges_for_values(vals, time_limit=time_per_candidate)
        
        if score < best_global_score:
            best_global_score = score
            best_global_values = vals.copy()
            best_global_edges = best_edges
            # Early stop if we hit a very low score?
            # Theoretical min for 35 edges?
            # If score is 20, that's very good.
            if best_global_score <= 20:
                break

    # If we didn't find anything (shouldn't happen), fallback
    if best_global_values is None:
        best_global_values = list(range(1, n+1))
        best_global_edges = set(random.sample(list(combinations(range(n), 2)), m))
        best_global_score = _raw_value(best_global_values, best_global_edges)

    # Final local refinement on the best found solution
    # Try to improve values slightly around the best found
    if time.time() < deadline - 5:
        current_vals = best_global_values.copy()
        current_edges = set(best_global_edges)
        current_score = best_global_score
        
        # Try small perturbations to values
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, n-1)
            old_val = current_vals[idx]
            # Try nearby values
            new_val = old_val + random.randint(-10, 10)
            if 1 <= new_val <= maximum and new_val not in current_vals:
                current_vals[idx] = new_val
                # Re-optimize edges quickly? Or just evaluate current edges?
                # Evaluating current edges is faster. If it improves, then re-optimize edges.
                new_score = _raw_value(current_vals, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    # Now re-optimize edges for this new value set
                    time_left = deadline - 5 - time.time()
                    if time_left > 1.0:
                        new_edges, new_score = optimize_edges_for_values(current_vals, time_limit=min(1.0, time_left/2))
                        if new_score < current_score:
                            current_edges = new_edges
                            current_score = new_score
                            best_global_score = current_score
                            best_global_values = current_vals.copy()
                            best_global_edges = new_edges
                else:
                    current_vals[idx] = old_val # Revert
    
    # Return best
    edges_list = [list(e) for e in best_global_edges]
    edges_list.sort()
    
    return {
        'values': list(best_global_values),
        'edges': edges_list
    }
# EVOLVE-BLOCK-END
