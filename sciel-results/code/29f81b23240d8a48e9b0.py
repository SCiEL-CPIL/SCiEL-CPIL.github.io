"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _count_violations_fast(adj, n, k_min=6, k_max=None):
    """
    Count number of regular induced subgraphs of size k in [k_min, k_max].
    Uses bitmask operations for speed.
    """
    if k_max is None:
        k_max = min(n, 8) # Focus on small k for fast evaluation
    
    count = 0
    # Precompute combinations for speed? No, itertools is fast enough.
    for k in range(k_min, k_max + 1):
        for subset in itertools.combinations(range(n), k):
            # Create mask for subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Calculate degrees within subset
            # deg[i] = popcount(adj[i] & mask)
            degrees = []
            for v in subset:
                d = (adj[v] & mask).bit_count()
                degrees.append(d)
            
            # Check if regular
            d0 = degrees[0]
            is_reg = True
            for d in degrees[1:]:
                if d != d0:
                    is_reg = False
                    break
            
            if is_reg:
                count += 1
    return count

def _has_violation_full(adj, n):
    """
    Check if ANY induced subgraph of size >= 6 is regular.
    This is the expensive check, done only on candidates.
    """
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degrees = []
            for v in subset:
                d = (adj[v] & mask).bit_count()
                degrees.append(d)
            
            d0 = degrees[0]
            is_reg = True
            for d in degrees[1:]:
                if d != d0:
                    is_reg = False
                    break
            
            if is_reg:
                return True, subset
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    best_n = 5
    best_edges = []
    
    # We aim for n=20. If we fail, we'll settle for lower.
    # Strategy: Simulated Annealing / Local Search with fast objective (k=6,7,8)
    # Then verify with full check.
    
    target_ns = [20, 19, 18, 17, 16]
    
    for n in target_ns:
        if time.time() > deadline - 20:
            break
            
        # Try multiple random restarts for this n
        max_restarts = 20
        if n == 20:
            max_restarts = 50 # More effort for the target
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Initial score (fast)
            current_score = _count_violations_fast(adj, n, 6, 8)
            
            # Simulated Annealing
            T = 10.0
            T_min = 0.1
            cooling = 0.95
            
            # We want to reach score 0
            # Max iterations per restart
            max_iter = 1000
            if n == 20:
                max_iter = 2000
                
            it = 0
            while it < max_iter and T > T_min and time.time() < deadline - 2:
                it += 1
                
                # Pick a random edge to flip
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                    
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_score = _count_violations_fast(adj, n, 6, 8)
                delta = new_score - current_score
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0:
                    current_score = new_score
                else:
                    # If delta is 0, always accept to diversify
                    if delta == 0 or random.random() < pow(2.71828, -delta / T):
                        current_score = new_score
                    else:
                        # Revert
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                T *= cooling
                
                if current_score == 0:
                    # Candidate found! Verify with full check
                    has_v, _ = _has_violation_full(adj, n)
                    if not has_v:
                        # Valid graph!
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        break # Break SA loop, go to next n or restart? 
                    # If not valid, continue SA to try to fix larger k violations
                    # Or restart? Restarting might be better if stuck in local min for k=6,7,8 but failing k=9
                    # Let's break and try a new restart for this n
                    break
            
            # If we found a valid graph, we can stop searching this n
            if n == best_n and best_edges:
                 # Check if we actually updated best_edges for this n
                 # The logic above updates best_n and best_edges.
                 # If best_n is updated to current n, we can break restarts
                 if best_n == n:
                     break

    # If we didn't find a good graph, try a simple greedy or random search for smaller n if needed
    # But the loop above handles n=20 down to 16.
    
    # Final fallback: if best_n is still low, try random graphs for n=20 one last time with just fast check?
    # No, the SA is robust enough.
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
