"""Rewritten program: simulated annealing on value assignments to minimize distinct sums/products."""
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    N = 14
    M = 35
    MAX_VAL = 10000

    # Use a structured graph: near-complete graph with 35 edges
    # We'll pick edges that are "friendly" to sum/product collisions
    # Start with a cycle + chords pattern, then optimize values
    
    def make_edges():
        """Create 35 edges on 14 vertices - use a structured pattern"""
        edges = []
        seen = set()
        # First add a 14-cycle
        for i in range(N):
            j = (i + 1) % N
            key = (min(i,j), max(i,j))
            edges.append([i, j])
            seen.add(key)
        # Add chords: connect vertices that are 2 apart
        for i in range(N):
            j = (i + 2) % N
            key = (min(i,j), max(i,j))
            if key not in seen:
                edges.append([i, j])
                seen.add(key)
        # Add more chords: distance 3
        for i in range(N):
            j = (i + 3) % N
            key = (min(i,j), max(i,j))
            if key not in seen:
                edges.append([i, j])
                seen.add(key)
        # Add remaining edges randomly until we have 35
        random.shuffle(range(N*N))
        for i in range(N):
            for j in range(i+1, N):
                if len(edges) >= M:
                    break
                key = (i, j)
                if key not in seen:
                    edges.append([i, j])
                    seen.add(key)
            if len(edges) >= M:
                break
        return edges[:M]
    
    edges = make_edges()
    
    def compute_score(values):
        """Compute raw value = max(distinct_sums, distinct_products)"""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def compute_score_detailed(values):
        """Compute both counts for tie-breaking"""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return (len(sums), len(prods))
    
    # Initial values: try to create collisions
    # Strategy: use values that are multiples of a small number
    # If values are all multiples of k, sums are multiples of k, products are multiples of k^2
    # This doesn't directly help with distinct count unless we engineer collisions
    
    # Better strategy: use pairs of values that sum to the same thing
    # e.g., values like [a, b, c, d] where a+b = c+d = S
    # Then edges (a,b) and (c,d) have the same sum
    
    # For products: a*b = c*d = P
    # We need values where multiple pairs have the same product
    
    # Let's try: use values from a set where many pairs share sum and product
    # Classic: if we use {x, y, z, w} with x*w = y*z and x+w = y+z, that's a nice collision
    
    # Simpler: use values that are all close together, so sums cluster
    # But we need distinct positive integers <= 10000
    
    # Start with a good initial guess
    random.seed(42)
    
    def random_initial():
        """Generate initial values with some structure"""
        # Use values centered around 500 with spread
        base = random.randint(200, 800)
        spread = random.randint(50, 200)
        vals = []
        for i in range(N):
            v = base + random.randint(-spread, spread)
            v = max(1, min(MAX_VAL, v))
            vals.append(v)
        # Ensure distinct
        while len(set(vals)) < N:
            dup = random.choice([v for v in vals if vals.count(v) > 1])
            idx = vals.index(dup)
            vals[idx] = random.randint(1, MAX_VAL)
        return vals
    
    best_values = random_initial()
    best_score = compute_score(best_values)
    
    # Simulated annealing
    T = 500.0
    T_min = 0.1
    cooling = 0.995
    
    current_values = best_values[:]
    current_score = best_score
    current_detailed = compute_score_detailed(current_values)
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a random vertex to perturb
        idx = random.randint(0, N - 1)
        old_val = current_values[idx]
        
        # Perturbation: small change or larger jump
        if random.random() < 0.7:
            # Small perturbation
            delta = random.randint(-50, 50)
            new_val = old_val + delta
        else:
            # Larger jump
            new_val = random.randint(1, MAX_VAL)
        
        new_val = max(1, min(MAX_VAL, new_val))
        
        if new_val == old_val:
            continue
        
        # Check if new_val is already used by another vertex
        if new_val in current_values and current_values.index(new_val) != idx:
            continue
        
        current_values[idx] = new_val
        new_score = compute_score(current_values)
        new_detailed = compute_score_detailed(current_values)
        
        delta_score = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_score <= 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            current_score = new_score
            current_detailed = new_detailed
            if new_score < best_score:
                best_score = new_score
                best_values = current_values[:]
        else:
            current_values[idx] = old_val
        
        # Cool down
        if T > T_min:
            T = max(T_min, T * cooling)
        
        # Restart if stuck
        if step % 2000 == 0 and T <= T_min:
            T = 200.0
            # Try a different initial point
            candidate = random_initial()
            cand_score = compute_score(candidate)
            if cand_score < best_score:
                best_score = cand_score
                best_values = candidate[:]
                current_values = candidate[:]
                current_score = cand_score
                current_detailed = compute_score_detailed(current_values)
            else:
                current_values = best_values[:]
                current_score = best_score
                current_detailed = compute_score_detailed(current_values)
    
    # Final local improvement pass
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(N):
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = best_values[i] + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in best_values and best_values.index(new_val) != i:
                    continue
                old = best_values[i]
                best_values[i] = new_val
                s = compute_score(best_values)
                if s < best_score:
                    best_score = s
                    improved = True
                else:
                    best_values[i] = old
    
    # Ensure types are correct
    values = [int(v) for v in best_values]
    edges_out = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges_out}
