import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def compute_metrics_fast(A):
    """Faster computation using set operations where possible."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            sum_set.add(a + A[j])
            prod_set.add(a * A[j])
    return max(len(sum_set), len(prod_set))

def build_collision_graph(pool, max_collisions=5000):
    """
    Build a weighted graph where edge weight between nodes i,j represents
    the number of sum/product collisions they create with each other.
    Higher weight = more collisions = better to group together.
    """
    n = len(pool)
    # For each pair, count how many times a+b or a*b appears
    # We'll use a simpler proxy: shared small prime factors and additive closeness
    
    # Precompute for each element: its sum pairs and product pairs
    # Edge weight = number of (i,j) pairs where a_i + a_j = a_k + a_l or a_i * a_j = a_k * a_l
    # This is expensive, so use a heuristic:
    # 1. Multiplicative: share small prime factors
    # 2. Additive: are close in value (more likely to collide with others)
    
    # For efficiency, compute pairwise collision potential
    # Use: for each element, count how many other elements it "collides" with
    # in terms of creating duplicate sums/products
    
    collision_count = [0] * n
    
    # Sum collisions: for each pair (i,j), a_i+a_j might equal a_k+a_l
    # Product collisions: for each pair (i,j), a_i*a_j might equal a_k*a_l
    
    # Efficient approach: build frequency maps
    sum_freq = {}
    prod_freq = {}
    pair_sum = {}
    pair_prod = {}
    
    for i in range(n):
        for j in range(i, n):
            s = pool[i] + pool[j]
            p = pool[i] * pool[j]
            sum_freq[s] = sum_freq.get(s, 0) + 1
            prod_freq[p] = prod_freq.get(p, 0) + 1
            pair_sum[(i,j)] = s
            pair_prod[(i,j)] = p
    
    # For each element, count how many pairs involving it create collisions
    elem_sum_collisions = [0] * n
    elem_prod_collisions = [0] * n
    
    for (i,j), s in pair_sum.items():
        if sum_freq[s] > 1:
            # This sum value appears multiple times, so both i and j get credit
            elem_sum_collisions[i] += sum_freq[s] - 1
            elem_sum_collisions[j] += sum_freq[s] - 1
    
    for (i,j), p in pair_prod.items():
        if prod_freq[p] > 1:
            elem_prod_collisions[i] += prod_freq[p] - 1
            elem_prod_collisions[j] += prod_freq[p] - 1
    
    # Combine: total collision potential per element
    collision_score = [elem_sum_collisions[i] + elem_prod_collisions[i] for i in range(n)]
    
    return collision_score

def greedy_modularity_selection(pool, target_size=20, collision_scores=None):
    """
    Greedy selection: start with empty set, add element that maximizes
    internal collision density (modularity-like objective).
    """
    if collision_scores is None:
        collision_scores = build_collision_graph(pool)
    
    n = len(pool)
    selected = []
    selected_set = set()
    
    # Precompute pairwise collision potential
    # For efficiency, we'll compute on the fly using the pool values
    
    for _ in range(target_size):
        best_idx = -1
        best_score = -1
        
        for i in range(n):
            if i in selected_set:
                continue
            
            # Score = how well this element fits with already selected
            # Use: sum of collision scores of selected elements + 
            #       number of collisions created with selected elements
            
            fit_score = 0
            if selected:
                # Count collisions with selected elements
                for j in selected:
                    # Sum collision
                    s = pool[i] + pool[j]
                    # Product collision
                    p = pool[i] * pool[j]
                    # Check if this sum/product appears with other selected pairs
                    # Approximate: if pool[i] and pool[j] share structure
                    # Use: gcd-based heuristic for multiplicative, 
                    #       difference-based for additive
                    g = math.gcd(pool[i], pool[j])
                    if g > 1:
                        fit_score += math.log(g)
                    # Additive: small differences suggest more collisions
                    diff = abs(pool[i] - pool[j])
                    if diff < 10:
                        fit_score += 1.0
                    elif diff < 50:
                        fit_score += 0.5
            
            # Also consider the element's own collision potential
            fit_score += collision_scores[i] * 0.1
            
            if fit_score > best_score:
                best_score = fit_score
                best_idx = i
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
    
    return [pool[i] for i in selected]

def local_search_optimize(initial_set, pool, deadline, max_iterations=10000):
    """
    Local search: try replacing one element at a time to improve score.
    """
    current = sorted(initial_set)
    current_score = compute_metrics(current)
    best = current[:]
    best_score = current_score
    
    pool_set = set(pool)
    current_set = set(current)
    
    iteration = 0
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        improved = False
        
        for i in range(len(current)):
            old_val = current[i]
            # Try replacing with elements from pool
            # Prioritize elements with high collision potential
            for new_val in pool:
                if new_val in current_set or new_val == old_val:
                    continue
                
                current[i] = new_val
                current_set.discard(old_val)
                current_set.add(new_val)
                
                new_score = compute_metrics(current)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                else:
                    current[i] = old_val
                    current_set.discard(new_val)
                    current_set.add(old_val)
            
            if not improved and time.time() > deadline - 5:
                break
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit
    
    # Generate candidate pool: focus on numbers with high collision potential
    # Small smooth numbers, small integers, and structured sequences
    
    pool = set()
    
    # 2,3,5,7-smooth numbers up to 5000
    for a in range(0, 13):
        for b in range(0, 9):
            for c in range(0, 6):
                for d in range(0, 5):
                    val = (1 << a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val <= 5000:
                        pool.add(val)
    
    # Small integers 1-200
    for i in range(1, 201):
        pool.add(i)
    
    # Structured: multiples of small numbers
    for base in [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 60, 64]:
        for k in range(1, 300):
            v = base * k
            if v <= 5000:
                pool.add(v)
    
    # Powers and near-powers
    for base in [2, 3, 4, 5, 6]:
        val = 1
        for _ in range(15):
            if val <= 5000:
                pool.add(val)
            val *= base
    
    pool = sorted(pool)
    
    # Limit pool size for efficiency
    if len(pool) > 500:
        # Keep smallest 500 (more collision potential)
        pool = pool[:500]
    
    # Build collision scores
    collision_scores = build_collision_graph(pool)
    
    # Initial candidates
    initial_candidates = []
    
    # 1. Greedy modularity selection
    greedy_result = greedy_modularity_selection(pool, 20, collision_scores)
    if len(greedy_result) == 20:
        initial_candidates.append(sorted(greedy_result))
    
    # 2. Top collision-score elements
    top_indices = sorted(range(len(pool)), key=lambda i: collision_scores[i], reverse=True)[:20]
    top_candidates = sorted([pool[i] for i in top_indices])
    if len(set(top_candidates)) == 20:
        initial_candidates.append(top_candidates)
    
    # 3. Small smooth numbers
    smooth_small = [x for x in pool if x <= 200][:20]
    if len(smooth_small) == 20:
        initial_candidates.append(sorted(smooth_small))
    
    # 4. Powers of 2
    powers2 = [1 << i for i in range(20) if (1 << i) <= 5000]
    if len(powers2) >= 20:
        initial_candidates.append(powers2[:20])
    
    # 5. Mixed 2^i * 3^j
    mixed = []
    for i in range(0, 10):
        for j in range(0, 10 - i):
            val = (1 << i) * (3 ** j)
            if val <= 5000:
                mixed.append(val)
    mixed = sorted(set(mixed))
    if len(mixed) >= 20:
        initial_candidates.append(mixed[:20])
    
    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    for cand in initial_candidates:
        if len(set(cand)) != 20 or max(cand) > 10000:
            continue
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
    
    if not best_set:
        best_set = list(range(1, 21))
        best_score = compute_metrics(best_set)
    
    # Local search optimization
    best_set, best_score = local_search_optimize(best_set, pool, deadline, max_iterations=20000)
    
    # Additional restarts with different initializations if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Different greedy with random perturbation
            random.seed(start_time + restart * 1000)
            perturbed_pool = pool[:]
            random.shuffle(perturbed_pool)
            # Mix: keep top 100 by collision score, shuffle rest
            top_100 = sorted(range(len(pool)), key=lambda i: collision_scores[i], reverse=True)[:100]
            perturbed_pool = [pool[i] for i in top_100] + [pool[i] for i in range(len(pool)) if i not in set(top_100)]
            
            greedy_result = greedy_modularity_selection(perturbed_pool[:300], 20, collision_scores[:300])
            if len(greedy_result) == 20:
                candidate = sorted(greedy_result)
                s = compute_metrics(candidate)
                if s < best_score:
                    best_score = s
                    best_set = candidate[:]
                    best_set, best_score = local_search_optimize(best_set, pool, deadline, max_iterations=5000)
    
    return sorted(best_set)
