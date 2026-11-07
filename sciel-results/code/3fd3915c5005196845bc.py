"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    def generate_value_candidates():
        """Generate a diverse set of candidate value arrays."""
        candidates = []
        
        # 1. Small integers (1-14) and permutations
        base_small = list(range(1, n + 1))
        candidates.append(base_small)
        # Shuffle a few times
        for _ in range(5):
            v = list(base_small)
            random.shuffle(v)
            candidates.append(v)
        
        # 2. Arithmetic Progressions
        for start in range(1, 10):
            for diff in range(1, 5):
                vals = [start + i * diff for i in range(n)]
                if max(vals) <= 10000 and len(set(vals)) == n:
                    candidates.append(vals)
        
        # 3. Small clusters (many small numbers)
        # Try to keep values low to increase collision probability for sums/prods
        for _ in range(10):
            vals = sorted(random.sample(range(1, 30), n))
            # Ensure distinct
            if len(set(vals)) == n:
                candidates.append(vals)
                
        # 4. Geometric-like (powers)
        # 1, 2, 4, 8... might be too spread out for products?
        # But 2, 3, 4, 6, 8, 9, 12... (highly composite numbers)
        highly_composite = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48]
        if max(highly_composite) <= 10000:
            candidates.append(highly_composite)
            
        # 5. Random small
        for _ in range(20):
            vals = random.sample(range(1, 50), n)
            if len(set(vals)) == n:
                candidates.append(vals)
                
        return candidates

    def evaluate_edge_selection_bnb(vals, K):
        """
        Check if we can select m edges with <= K distinct sums and <= K distinct products.
        Returns the list of edges if found, else None.
        """
        n = len(vals)
        all_edges = []
        edge_sums = {}
        edge_prods = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                s = vals[i] + vals[j]
                p = vals[i] * vals[j]
                edge_id = (i, j)
                all_edges.append(edge_id)
                edge_sums[edge_id] = s
                edge_prods[edge_id] = p
        
        # Pre-compute counts of edges per sum and per product to sort edges
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        for e in all_edges:
            sum_counts[edge_sums[e]] += 1
            prod_counts[edge_prods[e]] += 1
            
        # Sort edges: Prefer edges that are "popular" (share sum/prod with many others)
        # This helps find cliques/sets faster? 
        # Actually, for B&B, we want to pick edges that add FEWER new sums/prods.
        # So, sort by sum_count desc, then prod_count desc.
        sorted_edges = sorted(all_edges, key=lambda e: (-sum_counts[edge_sums[e]], -prod_counts[edge_prods[e]]))
        
        # B&B State
        found_solution = [None]
        
        def dfs(idx, count, current_sums, current_prods, selected_edges):
            if found_solution[0] is not None:
                return True
            
            if count == m:
                found_solution[0] = list(selected_edges)
                return True
            
            if idx >= len(sorted_edges):
                return False
            
            # Pruning: Not enough edges left
            if count + (len(sorted_edges) - idx) < m:
                return False
                
            # Pruning: Exceeded K
            if len(current_sums) > K or len(current_prods) > K:
                return False
            
            # Heuristic Pruning:
            # Even if we pick all remaining edges, can we stay within K?
            # This is hard to estimate quickly. Skip for now.
            
            e = sorted_edges[idx]
            s = edge_sums[e]
            p = edge_prods[e]
            
            # Branch 1: Include e
            # Check if adding e violates K
            new_sums_count = len(current_sums) + (0 if s in current_sums else 1)
            new_prods_count = len(current_prods) + (0 if p in current_prods else 1)
            
            if new_sums_count <= K and new_prods_count <= K:
                if s not in current_sums:
                    current_sums.add(s)
                if p not in current_prods:
                    current_prods.add(p)
                
                selected_edges.append(e)
                
                if dfs(idx + 1, count + 1, current_sums, current_prods, selected_edges):
                    return True
                
                selected_edges.pop()
                
                if s not in current_sums:
                    current_sums.discard(s)
                if p not in current_prods:
                    current_prods.discard(p)
            
            # Branch 2: Exclude e
            if dfs(idx + 1, count, current_sums, current_prods, selected_edges):
                return True
                
            return False

        dfs(0, 0, set(), set(), [])
        return found_solution[0]

    def get_lower_bound_K(vals):
        """Calculate a lower bound for K."""
        n = len(vals)
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        for i in range(n):
            for j in range(i + 1, n):
                s = vals[i] + vals[j]
                p = vals[i] * vals[j]
                sum_counts[s] += 1
                prod_counts[p] += 1
        
        max_sum_group = max(sum_counts.values()) if sum_counts else 1
        max_prod_group = max(prod_counts.values()) if prod_counts else 1
        
        lb_s = (m + max_sum_group - 1) // max_sum_group
        lb_p = (m + max_prod_group - 1) // max_prod_group
        
        return max(lb_s, lb_p)

    best_vals = None
    best_edges = None
    best_K = float('inf')
    
    candidates = generate_value_candidates()
    
    # Shuffle candidates to ensure we try diverse ones early
    random.shuffle(candidates)
    
    for vals in candidates:
        if time.time() - start_time > time_limit:
            break
            
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > 10000:
            continue
            
        lb_K = get_lower_bound_K(vals)
        
        # Upper bound: just use all possible distinct sums/prods
        all_sums = set()
        all_prods = set()
        for i in range(n):
            for j in range(i + 1, n):
                all_sums.add(vals[i] + vals[j])
                all_prods.add(vals[i] * vals[j])
        ub_K = max(len(all_sums), len(all_prods))
        
        # Binary search for minimal K
        low = lb_K
        high = ub_K
        found_K = None
        found_edges = None
        
        # Limit binary search steps to save time
        # If lb_K is close to ub_K, it's fast.
        while low <= high:
            mid = (low + high) // 2
            
            if time.time() - start_time > time_limit:
                break
                
            edges = evaluate_edge_selection_bnb(vals, mid)
            
            if edges:
                found_K = mid
                found_edges = edges
                high = mid - 1
            else:
                low = mid + 1
                
        if found_K is not None and found_K < best_K:
            best_K = found_K
            best_vals = vals
            best_edges = found_edges
            
            # If we find K=1 or very small, it's likely optimal or near optimal
            # But K=1 is impossible for 35 edges on 14 nodes?
            # Max edges with same sum: e.g. 1+4=2+3. Group size is small.
            # So K will be at least 5-10.
            
    if best_vals is None:
        # Fallback
        best_vals = list(range(1, n + 1))
        # Greedy edge selection
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        best_edges = all_edges[:m]
        
    return {'values': best_vals, 'edges': best_edges}
# EVOLVE-BLOCK-END
