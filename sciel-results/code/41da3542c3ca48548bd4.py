"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_edge_props(vals, edge):
        i, j = edge
        return vals[i] + vals[j], vals[i] * vals[j]

    def greedy_edge_selection(vals):
        """Selects m edges greedily to minimize max distinct sums/prods."""
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining_edges = list(all_edges)
        
        # Sort edges initially? Maybe by sum value to cluster small sums?
        # Or just pick randomly/greedily based on current state.
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            
            for edge in remaining_edges:
                s, p = get_edge_props(vals, edge)
                
                # Count how many new distinct values this edge would add
                new_sums_count = 0 if s in current_sums else 1
                new_prods_count = 0 if p in current_prods else 1
                
                # Heuristic score: prefer edges that don't add new distincts
                # If tied, prefer edges with smaller sums/products?
                score = (new_sums_count, new_prods_count, s, p)
                
                if score < best_score:
                    best_score = score
                    best_edge = edge
            
            if best_edge is None:
                break
                
            s, p = get_edge_props(vals, best_edge)
            current_sums.add(s)
            current_prods.add(p)
            selected_edges.append(best_edge)
            remaining_edges.remove(best_edge)
            
        return selected_edges, max(len(current_sums), len(current_prods))

    def dfs_edge_selection(vals, K):
        """
        Checks if it's possible to select m edges with <= K distinct sums and <= K distinct prods.
        Returns True if feasible, False otherwise.
        """
        # Pre-compute sums and products for all edges to avoid repeated calculation
        edge_props = {}
        for e in all_edges:
            edge_props[e] = get_edge_props(vals, e)
            
        # Sort edges to find solutions faster? 
        # Edges that share sums/prods with many others might be good.
        # For now, use a fixed order or random shuffle for diversity in search.
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        # To make DFS effective, we can group edges by their sum and product?
        # Let's just use standard DFS with pruning.
        
        selected = []
        current_sums = set()
        current_prods = set()
        
        # Recursion depth limit
        max_depth = m
        
        def dfs(idx, selected_count):
            if selected_count == m:
                return True
            
            # Pruning: if we can't pick enough edges from remaining
            if selected_count + (len(edges_to_try) - idx) < m:
                return False
            
            # Pruning: if current distinct counts exceed K
            if len(current_sums) > K or len(current_prods) > K:
                return False
            
            # Heuristic Pruning: 
            # Even if we pick all remaining edges, the distinct counts might not drop (they only grow).
            # So if current is already > K, we pruned above.
            
            for i in range(idx, len(edges_to_try)):
                e = edges_to_try[i]
                s, p = edge_props[e]
                
                # If this edge adds a new sum or product, check if we'd exceed K
                new_sums = current_sums | {s}
                new_prods = current_prods | {p}
                
                if len(new_sums) > K or len(new_prods) > K:
                    continue
                    
                # Select this edge
                selected.append(e)
                current_sums = new_sums
                current_prods = new_prods
                
                if dfs(i + 1, selected_count + 1):
                    return True
                
                # Backtrack
                selected.pop()
                current_sums = current_sums - {s} if s not in current_sums - {s} else current_sums # This is inefficient set ops
                # Better to use a stack or pass sets by value (copy) or use a counter map.
                # For n=14, copying sets is fast enough? 91 edges, depth 35.
                # Let's optimize set handling.
                
            return False
        
        # To optimize set backtracking, we can pass sets as arguments (immutable) or use a list of added elements.
        # Given the small size, copying sets in Python is O(N) where N is set size. Max size K.
        # This is acceptable.
        
        # Refactored DFS with set copying
        def dfs_optimized(idx, selected_count, c_sums, c_prods):
            if selected_count == m:
                return True
            if selected_count + (len(edges_to_try) - idx) < m:
                return False
            if len(c_sums) > K or len(c_prods) > K:
                return False
            
            for i in range(idx, len(edges_to_try)):
                e = edges_to_try[i]
                s, p = edge_props[e]
                
                new_sums = c_sums.copy()
                new_prods = c_prods.copy()
                
                new_sums.add(s)
                new_prods.add(p)
                
                if len(new_sums) > K or len(new_prods) > K:
                    continue
                    
                if dfs_optimized(i + 1, selected_count + 1, new_sums, new_prods):
                    return True
                    
            return False
            
        return dfs_optimized(0, 0, set(), set())

    def find_min_K(vals):
        """Find the minimum K for given values."""
        # Get a good upper bound from greedy
        _, K_greedy = greedy_edge_selection(vals)
        
        # Binary search or linear search for min K
        # Lower bound: max(1, m) ? No, sums can collide.
        # Min possible distinct sums is at least 1.
        # Let's try K from a low estimate up to K_greedy
        
        # A simple lower bound: we have 35 edges.
        # If all sums are same, K=1.
        # Let's try K from 1 to K_greedy.
        
        # Optimization: Start from K_greedy and go down? Or start low and go up?
        # Going up from 1 might be slow if K_greedy is large.
        # Let's try a few specific K values or binary search.
        
        # Since K_greedy is a valid solution, we know K <= K_greedy.
        # Let's try to find the smallest K.
        
        # Heuristic: The number of distinct sums in a random set is high.
        # Let's just check K_greedy, K_greedy-1, etc? 
        # Or check small K first?
        
        # Let's use a hybrid: check a few small K, then binary search?
        # Actually, DFS with small K prunes very fast.
        
        low = 1
        high = K_greedy
        best_K = K_greedy
        
        # Try to lower the bound
        while low <= high:
            mid = (low + high) // 2
            if dfs_edge_selection(vals, mid):
                best_K = mid
                high = mid - 1
            else:
                low = mid + 1
                
        return best_K

    # 1. Search for optimal values
    best_vals = None
    best_K = float('inf')
    
    # Initial Candidates
    candidates = []
    # Powers of 2
    candidates.append([2**i for i in range(1, 15)])
    # Powers of 3
    candidates.append([3**i for i in range(1, 15)])
    # Mix
    candidates.append([2**i * 3**(i%2) for i in range(1, 15)])
    # Arithmetic
    candidates.append(list(range(1, 15)))
    
    # Random candidates
    for _ in range(50):
        candidates.append(random.sample(range(1, 10001), n))
    
    # Local Search / Hill Climbing on Values
    iterations = 0
    while time.time() - start_time < time_limit * 0.8:
        iterations += 1
        # Pick a candidate
        if best_vals and random.random() < 0.7:
            current = list(best_vals)
        else:
            current = list(random.choice(candidates))
            
        if len(set(current)) != n or min(current) < 1 or max(current) > 10000:
            continue
            
        K_current = find_min_K(current)
        
        if K_current < best_K:
            best_K = K_current
            best_vals = list(current)
            
        # Perturb
        idx = random.randint(0, n - 1)
        old_val = current[idx]
        
        r = random.random()
        if r < 0.5:
            # Small step
            step = random.randint(-10, 10)
            new_val = old_val + step
        else:
            # Larger step
            new_val = random.randint(max(1, old_val - 100), min(10000, old_val + 100))
            
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in current and current.index(new_val) != idx:
            continue
            
        current[idx] = new_val
        
        if len(set(current)) != n:
            continue
            
        # Evaluate new value
        K_new = find_min_K(current)
        
        if K_new < best_K:
            best_K = K_new
            best_vals = list(current)
            
        # Simulated Annealing accept?
        # If K_new is worse, accept with small prob to escape local minima
        elif K_new > best_K and random.random() < 0.1:
            pass # Keep current as 'current' for next iteration? 
            # Actually, we are modifying 'current' in place.
            # If we accept a worse solution, we keep it as the basis for next perturbation.
            # If we reject, we should revert?
            # Let's just keep the best one as the main anchor and use random restarts.
            
    if best_vals is None:
        best_vals = list(range(1, n + 1))
        best_K = find_min_K(best_vals)

    # 2. Final Edge Selection for best_vals
    # We know best_K is the minimum. We need to return the actual edges.
    # We can run the DFS again to retrieve the edges, or just use Greedy if it achieves best_K.
    # Greedy might not achieve the optimal K found by DFS.
    # So we need a version of DFS that returns the edges.
    
    def dfs_get_edges(vals, K):
        edge_props = {}
        for e in all_edges:
            edge_props[e] = get_edge_props(vals, e)
            
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        selected = []
        
        def dfs(idx, selected_count, c_sums, c_prods):
            if selected_count == m:
                return True
            
            if selected_count + (len(edges_to_try) - idx) < m:
                return False
            if len(c_sums) > K or len(c_prods) > K:
                return False
            
            for i in range(idx, len(edges_to_try)):
                e = edges_to_try[i]
                s, p = edge_props[e]
                
                new_sums = c_sums.copy()
                new_prods = c_prods.copy()
                
                new_sums.add(s)
                new_prods.add(p)
                
                if len(new_sums) > K or len(new_prods) > K:
                    continue
                    
                selected.append(e)
                
                if dfs(i + 1, selected_count + 1, new_sums, new_prods):
                    return True
                
                selected.pop()
            return False
            
        if dfs(0, 0, set(), set()):
            return selected
        return None

    final_edges = dfs_get_edges(best_vals, best_K)
    
    if final_edges is None or len(final_edges) != m:
        # Fallback to greedy
        final_edges, _ = greedy_edge_selection(best_vals)
        
    return {'values': best_vals, 'edges': [list(e) for e in final_edges]}
# EVOLVE-BLOCK-END
