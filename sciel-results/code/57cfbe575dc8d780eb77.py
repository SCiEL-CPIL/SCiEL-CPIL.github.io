import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    n = 14
    m = 35
    
    def score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))
    
    def evaluate(vals, edges):
        return score(vals, edges)
    
    def greedy_edges_for_values(vals, time_limit=None):
        """Greedily select m edges minimizing max(distinct_sums, distinct_products)."""
        all_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(all_edges)
        
        selected = []
        sums_set = set()
        prods_set = set()
        
        for edge in all_edges:
            if len(selected) >= m:
                break
            if time_limit and time.time() > time_limit:
                break
            i, j = edge
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            # Try to add this edge; check if it's "good"
            new_sums = sums_set | {s}
            new_prods = prods_set | {p}
            # Accept if it doesn't increase the max too much, or always accept for now
            # Better: use a scoring that favors collisions
            selected.append(edge)
            sums_set.add(s)
            prods_set.add(p)
        
        # If we didn't get enough edges (shouldn't happen), fill randomly
        while len(selected) < m:
            for edge in all_edges:
                if edge not in selected:
                    selected.append(edge)
                    sums_set.add(vals[edge[0]] + vals[edge[1]])
                    prods_set.add(vals[edge[0]] * vals[edge[1]])
                    break
        
        return selected[:m]
    
    def local_search_edges(vals, edges, time_limit=None):
        """Try to improve edge selection via swaps."""
        edges = list(edges)
        edge_set = set(edges)
        all_edges = list(itertools.combinations(range(n), 2))
        current_score = score(vals, edges)
        
        improved = True
        while improved:
            improved = False
            if time_limit and time.time() > time_limit:
                break
            for idx, edge in enumerate(edges):
                if time_limit and time.time() > time_limit:
                    break
                for new_edge in all_edges:
                    if new_edge in edge_set:
                        continue
                    # Try swap
                    new_edges = edges[:idx] + [new_edge] + edges[idx+1:]
                    new_edge_set = set(new_edges)
                    s = score(vals, new_edges)
                    if s < current_score:
                        edges = new_edges
                        edge_set = new_edge_set
                        current_score = s
                        improved = True
                        break
                if time_limit and time.time() > time_limit:
                    break
        
        return edges, current_score
    
    def generate_structured_values():
        """Generate various structured value sets."""
        candidates = []
        
        # Powers of 2
        p2 = [2**i for i in range(n)]
        if max(p2) <= 10000:
            candidates.append(p2[:n])
        
        # Small multiples
        for base in range(1, 8):
            vals = [base * (i + 1) for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidates.append(vals)
        
        # Geometric with small ratio
        for r in [1.5, 2, 3]:
            vals = [int(r**i) + 1 for i in range(n)]
            vals = sorted(set(vals))
            if len(vals) >= n and max(vals) <= 10000:
                candidates.append(vals[:n])
        
        # Mixed: some powers, some small
        for shift in range(0, 5):
            vals = [2**i + shift for i in range(n)]
            if len(set(vals)) == n and max(vals) <= 10000:
                candidates.append(vals)
        
        # Consecutive small numbers
        for start in range(1, 10):
            vals = list(range(start, start + n))
            if max(vals) <= 10000:
                candidates.append(vals)
        
        # Fibonacci-like
        fib = [1, 1]
        for i in range(n - 2):
            fib.append(fib[-1] + fib[-2])
        if max(fib) <= 10000 and len(set(fib)) == n:
            candidates.append(fib[:n])
        
        # Random structured
        for _ in range(5):
            base = random.randint(1, 5)
            vals = []
            for i in range(n):
                v = base * (i + 1) + random.randint(0, 2)
                while v in vals or v <= 0 or v > 10000:
                    v = base * (i + 1) + random.randint(0, 5)
                vals.append(v)
            if len(set(vals)) == n and max(vals) <= 10000:
                candidates.append(vals)
        
        return candidates
    
    def local_search_values(vals, edges, time_limit=None):
        """Try to improve values via local perturbations."""
        vals = list(vals)
        current_score = score(vals, edges)
        
        improved = True
        while improved:
            improved = False
            if time_limit and time.time() > time_limit:
                break
            for idx in range(n):
                if time_limit and time.time() > time_limit:
                    break
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = vals[idx] + delta
                    if new_val <= 0 or new_val > 10000:
                        continue
                    if new_val in vals and vals.index(new_val) != idx:
                        continue
                    new_vals = vals[:idx] + [new_val] + vals[idx+1:]
                    if len(set(new_vals)) != n:
                        continue
                    s = score(new_vals, edges)
                    if s < current_score:
                        vals = new_vals
                        current_score = s
                        improved = True
                        break
        
        return vals, current_score
    
    # Main search
    best_score = float('inf')
    best_result = None
    
    # Generate initial candidates
    candidates = generate_structured_values()
    
    # Also add some random candidates
    for _ in range(10):
        vals = random.sample(range(1, 10001), n)
        candidates.append(vals)
    
    random.shuffle(candidates)
    
    for vals in candidates:
        if time.time() > deadline:
            break
        if not all(isinstance(v, int) for v in vals):
            continue
        if len(set(vals)) != n or min(vals) <= 0 or max(vals) > 10000:
            continue
        
        # Greedy edge selection
        edges = greedy_edges_for_values(vals, time_limit=deadline - 5)
        if len(edges) < m:
            continue
        
        sc = score(vals, edges)
        
        # Local search on edges
        edges, sc = local_search_edges(vals, edges, time_limit=deadline - 3)
        
        # Local search on values
        vals, sc = local_search_values(vals, edges, time_limit=deadline - 2)
        
        if sc < best_score:
            best_score = sc
            best_result = (list(vals), [list(e) for e in edges])
            
            if best_score <= 10:  # Good enough, stop
                break
    
    # Final validation
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in list(itertools.combinations(range(n), 2))[:m]]
        best_result = (vals, edges)
    
    vals, edges = best_result
    
    # Ensure types are correct
    vals = [int(v) for v in vals]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': vals, 'edges': edges}
