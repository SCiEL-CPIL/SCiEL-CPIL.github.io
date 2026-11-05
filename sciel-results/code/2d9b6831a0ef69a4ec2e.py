import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 90.0  # seconds
    max_val = 10000
    n_needed = 12
    max_sum = 2 * max_val
    
    # Prime sets to try, in order of increasing size
    # We want to minimize the number of distinct primes.
    # Start with smallest sets.
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
    ]
    
    # Fallback: If we run out of time or simple sets fail, 
    # we can return a heuristic solution or the trivial one.
    # But the search for small S should be fast.
    
    def generate_smooth_numbers(limit, primes):
        smooths = set()
        # Recursive generation
        def gen(current, idx):
            if current > limit:
                return
            if idx == len(primes):
                if current > 1: # 1 is not usually considered smooth in this context of sums > 2, but 1 is 2^0... 
                    # Actually, sums are at least 1+2=3. So 1 is not in the sum set.
                    # But we need to check if sum is smooth.
                    smooths.add(current)
                return
            p = primes[idx]
            # Multiply by p repeatedly
            val = current
            while val <= limit:
                gen(val, idx + 1)
                if val > limit // p:
                    break
                val *= p
        
        gen(1, 0)
        return smooths

    def find_clique(adj, k, nodes):
        """
        Find a clique of size k in the graph defined by adj.
        nodes: list of candidate nodes.
        Returns list of nodes or None.
        """
        if k == 0:
            return []
        if not nodes:
            return None
        
        # Pick a node
        # Heuristic: pick node with highest degree in the subgraph induced by nodes?
        # For simplicity, just iterate.
        for i, node in enumerate(nodes):
            # Neighbors of node that are also in nodes and appear after i
            # We want distinct indices, but the values must be distinct too.
            # Since we pick from a set of unique values, distinct values implies distinct indices in the final set.
            
            # Filter neighbors
            # adj[node] is a set of neighbors
            # We need neighbors that are in the remaining candidate pool
            # To ensure we don't revisit, we maintain a list of candidates.
            
            # Optimization: Only consider neighbors that are in the `nodes` list.
            # Since `nodes` is small in recursive steps, this is okay.
            # But `adj[node]` can be large.
            
            # Better: Pre-filter `nodes` to only those connected to `node`.
            # new_nodes = [n for n in nodes[i+1:] if n in adj[node]]
            
            # This is O(N^2) per step. With N=10000 initially, this is slow if not pruned.
            # However, we can restrict the initial candidate pool.
            
            pass # Placeholder for logic, implemented in main loop below for better control

    # Improved Search Strategy:
    # 1. Generate smooth sums for a prime set S.
    # 2. Build adjacency.
    # 3. Use a greedy/backtracking approach with pruning.
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 10: # Safety margin
            break
            
        smooth_sums = generate_smooth_numbers(max_sum, primes)
        
        # Build adjacency list
        # adj[u] = set of v such that u+v in smooth_sums
        # We only care about v > u to avoid duplicates? No, we need a set of 12.
        # Let's just store all valid neighbors.
        
        # To speed up clique finding, we restrict candidates.
        # A candidate u is "good" if it has many neighbors.
        # Let's compute degrees first? That requires iterating all u,v.
        # 10000 * 10000 is 100M ops. Too slow in Python.
        
        # Alternative:
        # Iterate u from 1 to 10000.
        # For each u, find valid v.
        # valid_v = { s - u for s in smooth_sums if 1 <= s-u <= 10000 and s-u != u }
        
        # We want to find 12 numbers.
        # Let's use a backtracking search that picks the first number, then the second, etc.
        
        def backtrack(selected, candidates):
            if time.time() > deadline - 5:
                return None
            if len(selected) == n_needed:
                return selected
            
            # If not enough candidates left
            if len(candidates) < n_needed - len(selected):
                return None
                
            # Try each candidate
            # Heuristic: Try candidates with high connectivity to already selected?
            # Or just iterate.
            
            # Optimization: Prune candidates that cannot connect to ALL already selected?
            # No, in a clique, every new node must connect to ALL previous nodes.
            # So for a new node v, v must be in intersection of adj[x] for all x in selected.
            
            if selected:
                # Intersection of neighbors of all selected nodes
                # Start with neighbors of the last selected node (or first, doesn't matter)
                # Actually, v must be neighbor to x1, x2, ..., xk.
                # So v must be in adj[x1] AND adj[x2] ... AND adj[xk].
                
                # Let's compute the common neighbors of `selected` that are in `candidates`.
                # To do this efficiently, we can maintain a set of "common neighbors" as we grow the clique.
                # Let's restructure:
                
                pass # Logic moved to outer function for clarity

        # Let's implement a specific backtracking that maintains the intersection of neighbors.
        
        def search_clique(primes):
            smooth_sums = generate_smooth_numbers(max_sum, primes)
            
            # Precompute adj for all u? 
            # Memory: 10000 sets. Each set size ~ |smooth_sums|.
            # |smooth_sums| for {2,3} is ~100.
            # 10000 * 100 = 1M ints. ~8MB. Okay.
            
            adj = [set() for _ in range(max_val + 1)]
            for u in range(1, max_val + 1):
                for s in smooth_sums:
                    v = s - u
                    if 1 <= v <= max_val and v != u:
                        adj[u].add(v)
            
            # Now find clique of size 12.
            # We can pick the first node u1.
            # Then u2 must be in adj[u1].
            # Then u3 must be in adj[u1] AND adj[u2].
            
            # To reduce search space, we can sort nodes by degree?
            # Or just iterate u1 from 1 to 10000.
            
            # For a fixed u1, the candidates for the rest are adj[u1].
            # We need to find a clique of size 11 in the subgraph induced by adj[u1].
            
            # Let's try a recursive helper.
            def find_clique_in_subgraph(k, candidates_set):
                """
                Find clique of size k in graph induced by candidates_set.
                candidates_set is a set of nodes.
                Returns list of nodes or None.
                """
                if k == 0:
                    return []
                if len(candidates_set) < k:
                    return None
                if time.time() > deadline - 5:
                    return None
                
                # Pick a node from candidates_set
                # Heuristic: Pick node with highest degree within candidates_set?
                # Calculating degrees is expensive.
                # Just pick the smallest? Or random?
                # Let's pick the smallest to be deterministic.
                node = min(candidates_set)
                
                # New candidates must be neighbors of `node` AND in `candidates_set`
                new_candidates = candidates_set & adj[node]
                
                # Remove `node` from new_candidates? 
                # We are building a set. We include `node` in the result.
                # The rest of the clique must be in new_candidates.
                
                res = find_clique_in_subgraph(k - 1, new_candidates)
                if res is not None:
                    return [node] + res
                
                # If this node didn't lead to a solution, try the next one?
                # This simple approach only tries the smallest node. 
                # It might miss solutions if the smallest node is not part of a max clique.
                # We need to try multiple nodes.
                
                # Better: Iterate through candidates.
                # But `candidates_set` can be large.
                
                # Let's modify: 
                # Try each node in candidates_set.
                # To avoid exponential blowup, we prune.
                
                # Actually, if we pick `node`, we restrict to `new_candidates`.
                # If it fails, we remove `node` from `candidates_set` and try again.
                
                # This is standard Bron-Kerbosch style.
                
                # Let's implement a loop.
                candidates_list = list(candidates_set)
                for node in candidates_list:
                    if time.time() > deadline - 5:
                        return None
                    new_candidates = candidates_set & adj[node]
                    # Ensure we don't pick node again
                    # The recursive call will pick from new_candidates.
                    # Since node is not in new_candidates (no self-loops), it's fine.
                    
                    res = find_clique_in_subgraph(k - 1, new_candidates)
                    if res is not None:
                        return [node] + res
                    
                    # Remove node from candidates_set for subsequent iterations
                    # Note: candidates_set is a set, so we can remove.
                    # But we are iterating over a list copy.
                    candidates_set.discard(node)
                    if len(candidates_set) < k:
                        return None

                return None

            # Iterate through possible first nodes
            # To speed up, we can skip nodes with low degree.
            # A node must have degree >= 11 to be part of a clique of size 12.
            
            for u in range(1, max_val + 1):
                if time.time() > deadline - 5:
                    return None
                if len(adj[u]) < 11:
                    continue
                
                # Candidates for the rest are adj[u]
                # We need a clique of size 11 in adj[u]
                res = find_clique_in_subgraph(11, adj[u])
                if res is not None:
                    return [u] + res
            
            return None

        result = search_clique(primes)
        if result is not None:
            # Verify result
            # Check distinctness
            if len(set(result)) != n_needed:
                continue
            # Check range
            if any(x < 1 or x > max_val for x in result):
                continue
            # Check sums
            valid = True
            for i in range(n_needed):
                for j in range(i+1, n_needed):
                    s = result[i] + result[j]
                    # Check if s is smooth over primes
                    temp = s
                    for p in primes:
                        while temp % p == 0:
                            temp //= p
                    if temp != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result
    
    # Fallback: Return a known good set or trivial
    # If we reach here, we might have timed out or failed to find small prime set.
    # Let's return the trivial solution to ensure validity, even if score is low.
    # Or try a random search for a few seconds?
    # The trivial solution 1..12 has 9 primes.
    # A random set might have more.
    # Let's just return 1..12.
    return list(range(1, 13))

# EVOLVE-BLOCK-END
