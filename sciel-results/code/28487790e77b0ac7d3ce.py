import time
import random
import itertools

def _compute_alpha_branch_and_bound(triples):
    """Compute independence number using branch and bound on the hypergraph.
    
    We want the largest subset S such that no triple is fully contained in S.
    Equivalently: find min vertex cover of the hypergraph (smallest set hitting all triples),
    then alpha = 16 - min_vertex_cover_size.
    
    Actually, independence number = largest set with no triple contained.
    This is NOT the same as 16 - min vertex cover in general hypergraphs.
    Let me reconsider: alpha = max |S| s.t. no triple ⊆ S.
    
    We can use inclusion-exclusion or branch and bound directly on subsets.
    For n=16, we can use a smarter approach: try sizes from 16 down, and use
    backtracking to check if there exists an independent set of size k.
    """
    n_points = 16
    triple_masks = []
    for t in triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
    
    if not triple_masks:
        return n_points
    
    # For each size k from n_points down to 1, check if there's an independent set of size k
    # Use backtracking with pruning
    def has_independent_set_of_size(k, points_available, idx_in_triples, depth):
        """Check if we can pick k more points from points_available such that 
        no triple is fully contained in the chosen set."""
        if depth == k:
            return True
        if len(points_available) < depth:
            return False
        
        # Pruning: if remaining points can't fill depth, fail
        if len(points_available) < depth:
            return False
        
        # Try to pick a point
        for i, p in enumerate(points_available):
            new_points = points_available[:i] + points_available[i+1:]
            # Check: does adding p complete any triple?
            # A triple is "completed" if all 3 points are in our chosen set
            # We track which triples are "almost complete" (2 points chosen, need 1 more)
            # This is complex; let's use a simpler check
            pass
        
        # Simpler approach: just enumerate subsets of size k and check
        # But that's C(16,k) which for k=8 is 12870, manageable
        return False
    
    # Direct approach: for k from 16 down to 1, check all C(16,k) subsets
    # C(16,8)=12870, C(16,7)=11440, etc. Total is manageable.
    # But we can be smarter: start from largest k and stop when we find one.
    
    # Actually, let's use a different strategy: 
    # The complement of an independent set is a vertex cover (hitting all triples).
    # So alpha = 16 - tau, where tau is the minimum vertex cover size.
    # We can find tau by trying all subsets of size t from 1 up.
    
    # Try vertex cover sizes from 1 upward
    for cover_size in range(1, n_points + 1):
        # Check if there exists a set of cover_size points that hits all triples
        for cover in itertools.combinations(range(n_points), cover_size):
            cover_set = set(cover)
            hits_all = True
            for t in triples:
                if not (cover_set & set(t)):
                    hits_all = False
                    break
            if hits_all:
                return n_points - cover_size
    
    return 0

def _is_linear(triples):
    """Check that any two triples share at most one point."""
    pair_used = set()
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                if key in pair_used:
                    return False
                pair_used.add(key)
    return True

def _greedy_build(seed=0, strategy='random'):
    """Build a maximal linear triple system using greedy approach."""
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    if strategy == 'random':
        rng.shuffle(all_triples)
    elif strategy == 'balanced':
        # Prefer triples that use less common points
        point_counts = [0] * 16
        for _ in range(len(all_triples)):
            t = min(all_triples, key=lambda x: sum(point_counts[x[i]] for i in range(3)))
            if all_triples:
                all_triples.remove(t)
            all_triples.append(t)
            for i in range(3):
                point_counts[t[i]] += 1
        rng.shuffle(all_triples)  # Add some randomness
    
    chosen = []
    pair_used = set()
    
    for t in all_triples:
        valid = True
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                if key in pair_used:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    key = (min(a, b), max(a, b))
                    pair_used.add(key)
            chosen.append(list(t))
    
    return chosen

def _simulated_annealing(triples, seed=0, max_iters=10000, start_temp=10.0, end_temp=0.01):
    """Simulated annealing to minimize independence number."""
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    current = [list(t) for t in triples]
    current_alpha = _compute_alpha_branch_and_bound(current)
    best = [list(t) for t in current]
    best_alpha = current_alpha
    
    # Precompute for efficiency
    def get_pair_used(triples):
        pu = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    pu.add((min(a, b), max(a, b)))
        return pu
    
    pair_used = get_pair_used(current)
    
    temp = start_temp
    cooling = (start_temp - end_temp) / max(1, max_iters)
    
    for it in range(max_iters):
        temp -= cooling
        if temp < end_temp:
            temp = end_temp
        
        # Pick a random triple to replace
        idx = rng.randrange(len(current))
        removed = current[idx]
        
        # Remove its pairs
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = removed[i], removed[j]
                pair_used.discard((min(a, b), max(a, b)))
        
        # Find valid candidates
        candidates = []
        for t in all_triples:
            valid = True
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    if (min(a, b), max(a, b)) in pair_used:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                candidates.append(t)
        
        if not candidates:
            # Restore
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = removed[i], removed[j]
                    pair_used.add((min(a, b), max(a, b)))
            current[idx] = removed
            continue
        
        new_triple = list(rng.choice(candidates))
        current[idx] = new_triple
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = new_triple[i], new_triple[j]
                pair_used.add((min(a, b), max(a, b)))
        
        new_alpha = _compute_alpha_branch_and_bound(current)
        
        if new_alpha <= best_alpha:
            best_alpha = new_alpha
            best = [list(t) for t in current]
        
        # Accept if better, or with probability exp(-(delta)/temp)
        delta = new_alpha - current_alpha
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temp, 0.001)):
            current_alpha = new_alpha
        else:
            # Revert
            current[idx] = removed
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = new_triple[i], new_triple[j]
                    pair_used.discard((min(a, b), max(a, b)))
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = removed[i], removed[j]
                    pair_used.add((min(a, b), max(a, b)))
    
    return best, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_triples = None
    best_alpha = 16
    
    # Phase 1: Multiple greedy builds with different strategies and seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        strategy = 'random' if seed % 2 == 0 else 'balanced'
        triples = _greedy_build(seed, strategy)
        if not _is_linear(triples):
            continue
        alpha = _compute_alpha_branch_and_bound(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
    
    # Phase 2: Simulated annealing on best solution
    if best_triples is not None and time.time() < deadline - 10:
        improved, alpha = _simulated_annealing(best_triples, seed=42, max_iters=5000, start_temp=5.0, end_temp=0.01)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved
    
    # Phase 3: More diverse starts + SA if time permits
    for seed in range(30, 100):
        if time.time() > deadline:
            break
        strategy = 'random' if seed % 3 != 0 else 'balanced'
        triples = _greedy_build(seed, strategy)
        if not _is_linear(triples):
            continue
        alpha = _compute_alpha_branch_and_bound(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
        
        if time.time() < deadline - 5:
            improved, a2 = _simulated_annealing(triples, seed=seed + 1000, max_iters=2000, start_temp=3.0, end_temp=0.01)
            if a2 < best_alpha:
                best_alpha = a2
                best_triples = improved
    
    if best_triples is None:
        # Fallback: simple disjoint triples
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
