import time
import random
import math

def compute_metrics(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    A is assumed to be a list of distinct positive integers.
    """
    n = len(A)
    # Using set comprehensions for speed
    sum_set = {A[i] + A[j] for i in range(n) for j in range(i, n)}
    prod_set = {A[i] * A[j] for i in range(n) for j in range(i, n)}
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Candidate pool: Small numbers are preferred for collisions
    max_pool_val = 2000
    base_pool = list(range(1, max_pool_val + 1))
    
    # Generate some structured initial candidates
    initial_candidates = []
    
    # 1. Arithmetic Progressions
    initial_candidates.append(list(range(1, 21)))
    initial_candidates.append(list(range(2, 22)))
    initial_candidates.append([2*i for i in range(1, 21)])
    initial_candidates.append([3*i for i in range(1, 21)])
    
    # 2. Geometric Progressions
    initial_candidates.append([2**i for i in range(20)])
    gp3 = [3**i for i in range(20)]
    if all(x <= 10000 for x in gp3):
        initial_candidates.append(gp3)
        
    # 3. Highly Composite Numbers (approx)
    hcn = [1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560]
    if len(hcn) >= 20 and all(x <= 10000 for x in hcn):
        initial_candidates.append(hcn[:20])
        
    # 4. Random sets from small range
    for _ in range(20):
        cand = random.sample(range(1, 101), 20)
        initial_candidates.append(cand)
        
    # 5. Random sets from medium range
    for _ in range(10):
        cand = random.sample(range(1, 501), 20)
        initial_candidates.append(cand)
        
    # 6. Biased Random Sets (favouring small numbers)
    for _ in range(10):
        cand = []
        while len(cand) < 20:
            val = int(random.random()**2 * 200) + 1
            if val not in cand:
                cand.append(val)
        initial_candidates.append(cand)

    # Validate and sort initial candidates
    valid_initials = []
    for cand in initial_candidates:
        if len(cand) == 20 and len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            valid_initials.append(sorted(cand))
            
    if not valid_initials:
        valid_initials = [sorted(random.sample(base_pool, 20))]
        
    # Evaluate initial candidates to find the best starting point
    best_set = None
    best_score = float('inf')
    current_set = None
    current_score = float('inf')
    
    for cand in valid_initials:
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
            current_set = cand[:]
            current_score = s
            
    # Graph-based clustering: Build collision graph and find dense 20-node subgraph
    # Edge weight between a and b = number of sum/product collisions they share with other pool elements
    # We'll use a greedy modularity-like approach to select 20 nodes with high internal edge density
    
    def build_collision_graph(pool):
        """
        Build a collision graph where edge(a,b) = number of other pool elements c 
        such that a+c or a*c equals b+c or b*c (collision potential).
        """
        n = len(pool)
        pool_set = set(pool)
        # For each pair, count how many c in pool create sum/product collisions
        # a+c == b+c' or a*c == b*c' for some c, c' in pool
        # Simplified: edge(a,b) = |{c in pool : a+c in {b+c' for c' in pool} or a*c in {b*c' for c' in pool}}|
        
        # Precompute sum and product sets for each element
        sum_partners = {}
        prod_partners = {}
        for i, a in enumerate(pool):
            sum_partners[a] = set()
            prod_partners[a] = set()
            for j, b in enumerate(pool):
                if i != j:
                    sum_partners[a].add(a + b)
                    prod_partners[a].add(a * b)
        
        # Edge weight: for pair (a,b), count how many c have a+c == b+c' or a*c == b*c'
        # This is expensive, so we'll use a cheaper proxy:
        # edge(a,b) = number of common sum partners + number of common product partners
        # i.e., |sum_partners(a) ∩ sum_partners(b)| + |prod_partners(a) ∩ prod_partners(b)|
        
        edge_weights = {}
        for i in range(n):
            for j in range(i+1, n):
                a, b = pool[i], pool[j]
                common_sums = len(sum_partners[a] & sum_partners[b])
                common_prods = len(prod_partners[a] & prod_partners[b])
                edge_weights[(a, b)] = common_sums + common_prods
        
        return edge_weights, sum_partners, prod_partners
    
    # Use a smaller pool for graph construction to keep it tractable
    graph_pool_size = 500
    graph_pool = random.sample(base_pool, graph_pool_size)
    
    if time.time() < deadline - 10.0:
        edge_weights, sum_partners, prod_partners = build_collision_graph(graph_pool)
        
        # Greedy selection: start with highest-degree node, then greedily add nodes
        # that maximize internal edge density
        def greedy_dense_subgraph(pool, edge_weights, k=20):
            """
            Greedy modularity maximization: select k nodes with highest internal edge density.
            """
            n = len(pool)
            pool_set = set(pool)
            
            # Compute degree (sum of edge weights) for each node
            degrees = {x: 0 for x in pool}
            for (a, b), w in edge_weights.items():
                degrees[a] += w
                degrees[b] += w
            
            # Start with highest-degree node
            selected = [max(pool, key=lambda x: degrees[x])]
            selected_set = set(selected)
            
            for _ in range(k - 1):
                best_candidate = None
                best_gain = -1
                
                for candidate in pool:
                    if candidate in selected_set:
                        continue
                    
                    # Compute gain in internal edge density if we add candidate
                    gain = 0
                    for s in selected:
                        if (candidate, s) in edge_weights:
                            gain += edge_weights[(candidate, s)]
                        elif (s, candidate) in edge_weights:
                            gain += edge_weights[(s, candidate)]
                    
                    # Normalize by current size to get density-like measure
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected.append(best_candidate)
                    selected_set.add(best_candidate)
                else:
                    break
            
            return sorted(selected)
        
        # Run greedy selection multiple times with different random pools
        for restart in range(3):
            if time.time() >= deadline - 10.0:
                break
            
            gp = random.sample(base_pool, graph_pool_size)
            ew, sp, pp = build_collision_graph(gp)
            candidate = greedy_dense_subgraph(gp, ew, 20)
            
            if len(candidate) == 20 and all(1 <= x <= 10000 for x in candidate):
                s = compute_metrics(candidate)
                if s < best_score:
                    best_score = s
                    best_set = candidate[:]
                    current_set = candidate[:]
                    current_score = s
            
            # Also try local search from this starting point
            if time.time() < deadline - 5.0:
                # Quick hill climbing from the graph-based solution
                for i in range(20):
                    old_val = current_set[i]
                    for new_val in range(1, 201):
                        if new_val in current_set:
                            continue
                        current_set[i] = new_val
                        s = compute_metrics(current_set)
                        if s < current_score:
                            current_score = s
                            if s < best_score:
                                best_score = s
                                best_set = current_set[:]
                        else:
                            current_set[i] = old_val
                    current_set[i] = old_val
    
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    num_steps = 0
    max_steps = 1000000
    
    while time.time() < deadline - 2.0:
        step = num_steps
        num_steps += 1
        
        # Cooling schedule
        progress = min(1.0, num_steps / 50000.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Random Move: Replace one element
        idx = random.randint(0, 19)
        old_val = current_set[idx]
        
        # Choose new value
        new_val = int(random.random()**2 * max_pool_val) + 1
        
        # Check duplicates
        if new_val in current_set:
            found = False
            for _ in range(5):
                new_val = int(random.random()**2 * max_pool_val) + 1
                if new_val not in current_set:
                    found = True
                    break
            if not found:
                continue
            
        current_set[idx] = new_val
        new_score = compute_metrics(current_set)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta <= 0 or (T > 0 and random.random() < min(1.0, -delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
            
        # Occasionally do a swap
        if num_steps % 10 == 0 and time.time() < deadline - 2.0:
             idx1 = random.randint(0, 19)
             idx2 = random.randint(0, 19)
             if idx1 != idx2:
                 v1 = current_set[idx1]
                 v2 = current_set[idx2]
                 new_v1 = int(random.random()**2 * max_pool_val) + 1
                 new_v2 = int(random.random()**2 * max_pool_val) + 1
                 
                 if new_v1 == new_v2:
                     continue
                 if new_v1 in current_set or new_v2 in current_set:
                     continue
                     
                 current_set[idx1] = new_v1
                 current_set[idx2] = new_v2
                 new_score = compute_metrics(current_set)
                 
                 delta = new_score - current_score
                 if delta <= 0 or (T > 0 and random.random() < min(1.0, -delta / T)):
                     current_score = new_score
                     if new_score < best_score:
                         best_score = new_score
                         best_set = current_set[:]
                 else:
                     current_set[idx1] = v1
                     current_set[idx2] = v2

    # Final Polish: Hill Climbing on best_set
    if time.time() < deadline - 1.0:
        improved = True
        polish_deadline = min(deadline - 0.5, time.time() + 10.0)
        polish_pool = list(range(1, 1001))
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                for new_val in polish_pool:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val
                
    return sorted(best_set)
