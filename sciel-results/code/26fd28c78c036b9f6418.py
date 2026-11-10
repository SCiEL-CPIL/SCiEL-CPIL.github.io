import time
import random
import itertools

# Precompute all triples and compatibility
N = 16
ALL_TRIPLES = []
TRIPLE_MASKS = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append((a, b, c))
            TRIPLE_MASKS.append(mask)

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute compatibility: compat[i] is a set of indices j such that triple i and j are compatible
COMPAT = [set() for _ in range(NUM_TRIPLES)]
for i in range(NUM_TRIPLES):
    for j in range(i + 1, NUM_TRIPLES):
        # Compatible if intersection size <= 1
        # Intersection size is popcount(mask_i & mask_j)
        inter = TRIPLE_MASKS[i] & TRIPLE_MASKS[j]
        if bin(inter).count('1') <= 1:
            COMPAT[i].add(j)
            COMPAT[j].add(i)

def calc_alpha(edge_indices):
    """
    Calculate the independence number of the hypergraph defined by edge_indices.
    Returns the size of the largest independent set.
    """
    if not edge_indices:
        return N
    
    # Convert to set of masks for fast lookup? 
    # Actually, checking independence of a subset S:
    # S is independent if no edge e is a subset of S.
    # i.e., for all e in edges, (mask_e & mask_S) != mask_e
    
    edge_masks = [TRIPLE_MASKS[i] for i in edge_indices]
    
    # Check from largest possible size down
    for k in range(N, 0, -1):
        # Generate all subsets of size k
        # Optimization: If k is large, there are few subsets. If k is small, many.
        # But we stop at the first k where an independent set exists.
        # Usually alpha is small, so we check k=16, 15, ... down to alpha.
        # Wait, if alpha is 5, we check 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6 (all dependent), then 5 (find one).
        # This is efficient if alpha is small.
        
        found_independent = False
        for subset in itertools.combinations(range(N), k):
            mask_s = 0
            for v in subset:
                mask_s |= (1 << v)
            
            is_indep = True
            for em in edge_masks:
                if (mask_s & em) == em:
                    is_indep = False
                    break
            
            if is_indep:
                found_independent = True
                break
        
        if found_independent:
            return k
            
    return 0 # Should not happen

def greedy_construct():
    """
    Construct a maximal linear system using a greedy randomized approach.
    Returns list of edge indices.
    """
    order = list(range(NUM_TRIPLES))
    random.shuffle(order)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        # Check if idx is compatible with all selected
        compatible = True
        for s_idx in selected:
            if s_idx not in COMPAT[idx]:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
            selected_set.add(idx)
            
    return selected

def local_search(edge_indices, max_iters=1000):
    """
    Try to improve the independence number by swapping edges.
    """
    current_edges = list(edge_indices)
    current_alpha = calc_alpha(current_edges)
    
    for _ in range(max_iters):
        if len(current_edges) < 2:
            break
            
        # Pick a random edge to remove
        rem_idx = random.choice(range(len(current_edges)))
        rem_edge = current_edges[rem_idx]
        
        # Remove it
        temp_edges = current_edges[:rem_idx] + current_edges[rem_idx+1:]
        temp_set = set(temp_edges)
        
        # Find a random compatible edge to add
        # We want an edge that is compatible with all in temp_set
        candidates = []
        # To speed up, we can just pick a random triple and check
        # Or precompute? 
        # Let's just try a few random triples
        added = False
        for _ in range(10): # Try a few random candidates
            cand_idx = random.randint(0, NUM_TRIPLES - 1)
            if cand_idx in temp_set:
                continue
            
            # Check compatibility
            ok = True
            for s_idx in temp_edges:
                if s_idx not in COMPAT[cand_idx]:
                    ok = False
                    break
            
            if ok:
                new_edges = temp_edges + [cand_idx]
                new_alpha = calc_alpha(new_edges)
                
                if new_alpha < current_alpha:
                    current_edges = new_edges
                    current_alpha = new_alpha
                    added = True
                    break
                elif new_alpha == current_alpha and random.random() < 0.1:
                    # Accept equal to explore
                    current_edges = new_edges
                    added = True
                    break
        
        if not added:
            # Try to just add an edge if possible? 
            # Greedy already made it maximal, so we can't add without removing.
            # But maybe we can remove 2 and add 2?
            # For now, stick to 1-swap. If stuck, restart greedy.
            pass
            
    return current_edges, current_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 110s safety margin
    
    best_alpha = N
    best_system = []
    
    # Initial greedy runs
    while time.time() < deadline:
        # Generate a new greedy system
        sys = greedy_construct()
        
        # Local search to improve
        sys, alpha = local_search(sys, max_iters=500)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = sys
            
        # If we hit a very low alpha, we might break early
        # Theoretical lower bound for N=16?
        # If alpha <= 4, that's very good.
        if best_alpha <= 4:
            break
            
    # Convert indices back to triples
    return [ALL_TRIPLES[i] for i in best_system]
