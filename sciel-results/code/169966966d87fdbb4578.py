"""
Strategy: Branch-and-Bound Maximum Independent Set on Conflict Graph
We model the problem as finding a clique of size n_rows in the complement of the conflict graph
(or an independent set of size n_rows in the conflict graph).
Nodes are candidate rows (subsets of {0..M} of size k+1 starting at 0).
Edges connect rows that share at least one difference.
We use a bitset-based branch-and-bound solver to find if an independent set of size n_rows exists
for a given maximum entry M. We binary search on M to find the minimum possible scope.
This is structurally different from local search: it is an exact combinatorial search with
strong pruning, aiming for optimality within the time budget, rather than heuristic improvement.
"""
import time
import random
import sys
import itertools

def construct(n_rows, k):
    # Deadline: 145 seconds (5s safety margin from 150s limit)
    deadline = time.time() + 145.0
    
    # 1. Generate a baseline greedy solution to have a valid fallback
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_scope = _max_entry(baseline)
    
    # If the problem is very small, return baseline immediately
    if n_rows <= 2 and k <= 3:
        return baseline

    # 2. Binary search on the scope M (largest entry)
    # Lower bound: simple counting bound or greedy result - some margin
    # Upper bound: best known (112) or greedy result
    
    # Counting bound: n_rows * k(k+1)/2 differences must fit in 1..M
    # So M >= sqrt(2 * n_rows * k(k+1)/2) roughly, but actually M is the max element.
    # The number of differences is n_rows * C(k+1, 2).
    # These differences must be distinct integers in 1..M.
    # So M >= n_rows * C(k+1, 2).
    # Wait, the differences are a_j - a_i. The max difference is M.
    # So we need at least n_rows * C(k+1, 2) distinct integers in 1..M.
    # Thus M >= n_rows * C(k+1, 2).
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lower_bound = max(total_diffs, 0)
    
    # We know 112 is achievable for (7,5). 
    # For general n_rows, k, we can start high and go down.
    upper_bound = max(best_scope, 112, lower_bound + 10)
    
    # If the greedy solution is already very good (close to lower bound), we might not need to search much.
    # But let's try to improve.
    
    # We will search for the smallest M such that an independent set of size n_rows exists
    # in the graph of rows with max entry <= M.
    
    # To make the graph manageable, we limit the number of candidate rows.
    # We generate a diverse set of rows with max entry <= M.
    
    # Strategy:
    # 1. Start with M = upper_bound.
    # 2. Generate a pool of candidate rows with max entry <= M.
    # 3. Check if an independent set of size n_rows exists.
    # 4. If yes, update best_scope and best_rows, and try smaller M.
    # 5. If no, try larger M (shouldn't happen if upper_bound is valid, but just in case).
    
    # Actually, binary search on M is tricky because the graph changes.
    # Instead, we can iterate M downwards from best_scope - 1 to lower_bound.
    # For each M, we generate a pool of rows with max entry <= M.
    # Then we run the branch-and-bound.
    
    # To save time, we only try a few M values, or we use a heuristic to jump.
    # Let's try M = best_scope - 1, best_scope - 2, ... down to max(lower_bound, best_scope - 20).
    # If we find a solution, we return it.
    
    # Also, we should include the greedy rows in the pool for each M to ensure we don't miss the baseline.
    
    for M in range(best_scope - 1, max(lower_bound, best_scope - 30), -1):
        if time.time() > deadline - 5:
            break
            
        # Generate pool of candidates with max entry <= M
        # We need a diverse pool. 
        # Randomized greedy generation is good.
        # We also include the greedy rows that fit in M.
        
        pool = []
        pool_diffs = []
        
        # Add greedy rows that fit
        for row in baseline:
            if max(row) <= M:
                pool.append(row)
                pool_diffs.append(_get_diffs(row))
        
        # Generate random rows
        # Number of random rows: proportional to time available and M
        # Let's aim for ~100-200 random rows
        target_pool_size = 150
        num_random = target_pool_size - len(pool)
        
        for _ in range(num_random):
            if time.time() > deadline - 5:
                break
            row = _generate_random_row(k, M)
            if row is not None:
                pool.append(row)
                pool_diffs.append(_get_diffs(row))
                
        if len(pool) < n_rows:
            continue
            
        # Build conflict graph (bitset representation)
        # nodes: 0..len(pool)-1
        # adj[i] is a bitset where bit j is 1 if i and j conflict (share a diff)
        n_pool = len(pool)
        adj = [0] * n_pool
        
        # Precompute conflict matrix
        # This is O(N^2 * D) where D is num diffs. 
        # For N=200, D=15, this is fast.
        for i in range(n_pool):
            for j in range(i + 1, n_pool):
                if pool_diffs[i] & pool_diffs[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Run branch-and-bound to find independent set of size n_rows
        # We want to find ANY independent set of size n_rows.
        # If found, we have a valid solution with scope M.
        
        # To make the search faster, we can order nodes by degree (smallest first) or by max entry.
        # But bitset B&B is usually effective with simple ordering.
        
        # Let's use a recursive B&B.
        # State: candidates (bitset), current_set_size, current_max_entry
        # We want to reach current_set_size == n_rows.
        
        # Optimization: If the number of candidates is less than n_rows - current_set_size, prune.
        # Also, if the maximum possible independent set size in the candidate subgraph is less than needed, prune.
        # A simple upper bound on independent set size is the number of candidates.
        # A better bound: greedy coloring of the subgraph. But that's expensive.
        # Let's stick to a simple B&B with bitsets.
        
        found_solution = _find_independent_set(adj, n_pool, n_rows, deadline)
        
        if found_solution is not None:
            # found_solution is a list of indices
            candidate_rows = [pool[i] for i in found_solution]
            current_scope = max(max(row) for row in candidate_rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = candidate_rows
                # If we hit the theoretical lower bound, we can stop
                if best_scope <= lower_bound:
                    return best_rows
            # If we found a solution for M, we can try smaller M
            # But we are iterating M downwards, so we will naturally try smaller M next.
            # However, if we found a solution with scope < M, we should update M for the next iteration?
            # The loop iterates M downwards. If we found a solution with scope S < M, 
            # the next iteration is M-1. We might skip some scopes.
            # But since we are looking for the minimum, and we are going down, 
            # finding a solution at M means we can potentially find one at M-1.
            # So we continue.
            
        # If we didn't find a solution for this M, we try M-1.
        
    return best_rows

def _find_independent_set(adj, n, target_size, deadline):
    """
    Finds an independent set of size target_size in the graph defined by adj.
    Returns a list of node indices or None.
    Uses bitset representation for speed.
    """
    # We use a recursive helper
    # To avoid deep recursion limits, we could use a stack, but Python recursion is okay for small depths.
    # However, n can be ~150. Recursion depth up to 150 is fine.
    
    # Heuristic: Order nodes. 
    # Nodes with higher degree are "more constrained". 
    # In independent set, we want to pick nodes that don't conflict.
    # A common heuristic for Max Independent Set is to pick the node with smallest degree in the candidate set.
    # Here we just need ANY set of size target_size.
    
    # Let's use a simple recursive search.
    # To speed it up, we can use a greedy initialization to see if a simple set exists.
    # But B&B should handle it.
    
    # We represent the set of available candidates as a bitset.
    # Initially, all nodes are available.
    all_nodes_mask = (1 << n) - 1
    
    # We'll use a helper function
    def search(candidates, current_set, depth):
        if time.time() > deadline - 1:
            return None
            
        if depth == target_size:
            return current_set
            
        # Pruning: if number of candidates < target_size - depth, fail
        if candidates.bit_count() < target_size - depth:
            return None
            
        # Pick a node to branch on.
        # Heuristic: pick the node with the smallest degree in the induced subgraph of candidates?
        # Or just pick the first available node.
        # Let's pick the node with the smallest index for determinism, or random for diversity?
        # Deterministic is better for B&B consistency.
        
        # To improve pruning, we can try to find a greedy independent set first.
        # But let's stick to standard B&B.
        
        # Get the first set bit
        if candidates == 0:
            return None
            
        # Pick a node. 
        # A good heuristic: pick the node with the highest degree in the candidate subgraph?
        # No, for independent set, picking a node removes its neighbors.
        # We want to keep options open. 
        # Actually, the choice of node affects the branching factor.
        # Let's just pick the lowest index available.
        
        # Extract lowest bit
        lsb = candidates & -candidates
        node = lsb.bit_length() - 1
        
        # Option 1: Include node
        # New candidates: candidates without node and without its neighbors
        new_candidates_include = candidates & ~(adj[node] | lsb)
        res = search(new_candidates_include, current_set + [node], depth + 1)
        if res is not None:
            return res
            
        # Option 2: Exclude node
        # New candidates: candidates without node
        new_candidates_exclude = candidates & ~lsb
        res = search(new_candidates_exclude, current_set, depth)
        if res is not None:
            return res
            
        return None
        
    return search(all_nodes_mask, [], 0)

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _generate_random_row(k, max_val=200):
    """
    Generates a random valid row (strictly increasing, starts at 0) with max entry <= max_val.
    """
    if max_val < k:
        return None
        
    row = [0]
    x = 0
    while len(row) < k + 1:
        # Pick a random next value > x
        # Ensure we can fit the remaining elements
        remaining = k + 1 - len(row)
        min_next = x + 1
        max_next = max_val - (remaining - 1) # Leave space for remaining elements
        if max_next < min_next:
            return None
            
        next_x = random.randint(min_next, max_next)
        row.append(next_x)
        x = next_x
        
    if max(row) > max_val:
        return None
        
    return row

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(row) for row in rows)
