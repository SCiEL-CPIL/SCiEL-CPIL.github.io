import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Try to build graphs of decreasing target size using greedy sequential construction
    for target_n in [21, 20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break
        
        budget = 60 if target_n == 21 else 30
        search_deadline = min(deadline - 3, time.time() + budget)
        
        graph = _greedy_sequential_build(target_n, search_deadline)
        if graph is not None:
            return graph
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

def _greedy_sequential_build(n, deadline):
    """
    Greedy sequential vertex addition:
    - Start with empty graph on 0 vertices
    - For each new vertex (1..n-1), try all 2^k possible connection patterns to existing vertices
    - Choose the pattern that creates the fewest new regular induced subgraphs of size >= 6
    - Use randomization and multiple restarts to escape local minima
    """
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    max_restarts = 20 if n <= 18 else 8
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initial seed for diversity
        seed_adj = np.zeros((min(n, 6), min(n, 6)), dtype=np.uint8)
        if min(n, 6) >= 2:
            for i in range(min(n, 6)):
                for j in range(i + 1, min(n, 6)):
                    if rng.random() < 0.5:
                        seed_adj[i, j] = 1
                        seed_adj[j, i] = 1
        
        adj = _build_greedy(seed_adj, n, deadline, rng)
        
        if adj is not None:
            # Full verification
            if _is_valid_full(adj, n):
                return _adj_to_edges(adj, n)
            else:
                # Count violations as a heuristic score
                viol_count = _count_violations(adj, n)
                if viol_count < best_score:
                    best_score = viol_count
                    best_result = adj.copy()
    
    # If we have a near-miss, try to repair it
    if best_result is not None and best_score <= 10:
        repaired = _repair_local_search(best_result, n, deadline)
        if repaired is not None and _is_valid_full(repaired, n):
            return _adj_to_edges(repaired, n)
    
    return None

def _build_greedy(seed_adj, n, deadline, rng):
    """
    Build graph sequentially from seed_adj.
    """
    k = seed_adj.shape[0]
    adj = np.zeros((n, n), dtype=np.uint8)
    adj[:k, :k] = seed_adj
    
    for new_v in range(k, n):
        if time.time() > deadline - 1:
            return None
        
        # Try all 2^k connection patterns (or a random sample if k is large)
        if k <= 15:
            patterns = list(itertools.product([0, 1], repeat=k))
            # Shuffle for diversity
            rng.shuffle(patterns)
            # Limit to top candidates to save time
            if len(patterns) > 500:
                patterns = patterns[:500]
        else:
            # For large k, sample random patterns
            patterns = []
            for _ in range(200):
                pattern = [rng.randint(0, 1) for _ in range(k)]
                patterns.append(tuple(pattern))
        
        best_pattern = None
        best_new_violations = float('inf')
        best_total_score = float('inf')
        
        for pattern in patterns:
            # Create candidate adj
            for i in range(k):
                adj[new_v, i] = pattern[i]
                adj[i, new_v] = pattern[i]
            
            # Count new regular induced subgraphs involving new_v
            new_viols = _count_new_violations(adj, new_v, k, n)
            
            # Also consider a small penalty for high degree variance (heuristic)
            degs = adj[:new_v+1, :new_v+1].sum(axis=1)
            deg_var = np.var(degs)
            
            total_score = new_viols + 0.01 * deg_var
            
            if total_score < best_total_score:
                best_total_score = total_score
                best_new_violations = new_viols
                best_pattern = pattern
        
        # Apply best pattern
        if best_pattern is not None:
            for i in range(k):
                adj[new_v, i] = best_pattern[i]
                adj[i, new_v] = best_pattern[i]
        else:
            # Fallback: random connections
            for i in range(k):
                val = rng.randint(0, 1)
                adj[new_v, i] = val
                adj[i, new_v] = val
    
    return adj

def _count_new_violations(adj, new_v, k, n):
    """
    Count regular induced subgraphs of size >= 6 that include new_v.
    Only check subsets that include new_v and at least one existing vertex.
    """
    count = 0
    existing = list(range(k))
    
    # Check subsets of size 6..min(n, 10) that include new_v
    max_size = min(n, 10)  # Limit to save time
    for size in range(6, max_size + 1):
        # Choose (size-1) from existing vertices
        for subset in itertools.combinations(existing, size - 1):
            full_subset = list(subset) + [new_v]
            if _is_regular(adj, full_subset):
                count += 1
                if count > 50:  # Early termination
                    return count
    
    return count

def _count_violations(adj, n):
    """Count all regular induced subgraphs of size >= 6."""
    count = 0
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                count += 1
                if count > 100:
                    return count
    return count

def _repair_local_search(adj, n, deadline):
    """
    Local search to fix remaining violations.
    """
    rng = random.Random(999)
    adj = adj.copy()
    
    while time.time() < deadline - 1:
        # Find a violation
        viol = _find_one_violation(adj, n)
        if viol is None:
            return adj
        
        # Flip a random edge in the violation
        u, v = rng.sample(viol, 2)
        adj[u, v] = 1 - adj[u, v]
        adj[v, u] = adj[u, v]
    
    return adj if _is_valid_full(adj, n) else None

def _find_one_violation(adj, n):
    """Find one regular induced subgraph of size >= 6, or None."""
    # Check size 6 first (most common)
    for subset in itertools.combinations(range(n), 6):
        if _is_regular(adj, subset):
            return list(subset)
    # Check larger sizes
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return list(subset)
    return None

def _is_regular(adj, subset):
    """Check if induced subgraph on subset is regular."""
    sub_adj = adj[np.ix_(subset, subset)]
    degs = sub_adj.sum(axis=1)
    return np.all(degs == degs[0])

def _is_valid_full(adj, n):
    """Check if NO subset of size >= 6 is regular."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return False
    return True

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges
