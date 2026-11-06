import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Try to find a graph for n from 35 down to 1
    for n in range(35, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts for this n
        found = False
        attempts = 0
        max_attempts = 50  # tune this
        
        while not found and attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            adj = [0] * n
            # Initialize with random edges
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local search
            for step in range(10000):
                if time.time() > deadline - 5:
                    break
                
                # Check K4
                k4 = find_k4(adj, n)
                if k4:
                    # Remove a random edge from the K4
                    i, j, k, l = k4
                    # pick random edge among (i,j), (i,k), ...
                    edges = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                    u, v = random.choice(edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    continue
                
                # Check Independent 6 (K6 in complement)
                # Build complement adj quickly? Or just check on fly?
                # Building complement is O(n^2).
                # We can maintain it or recompute.
                # For n=35, O(n^2) is small.
                cadj = [0] * n
                full_mask = (1 << n) - 1
                for i in range(n):
                    cadj[i] = (full_mask ^ (1 << i)) & ~adj[i]
                
                k6 = find_k6(cadj, n)
                if k6:
                    # Add a random edge from the K6 in G (which is non-edge in G)
                    # k6 is a tuple of 6 vertices
                    # pick two random vertices from it
                    u, v = random.sample(k6, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
                
                # If no K4 and no K6 in complement, it's valid
                found = True
                break
            
        if found:
            # Verify
            if is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph (e.g., Paley 17)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def find_k4(adj, n):
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                cn = adj[i] & adj[j]
                # find edge in cn
                # iterate bits in cn
                temp = cn
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    temp -= lsb
                    # check if k is connected to any other in cn
                    if cn & adj[k]:
                        # find l
                        l_mask = cn & adj[k]
                        if l_mask:
                            l = (l_mask & -l_mask).bit_length() - 1
                            return (i, j, k, l)
    return None

def find_k6(cadj, n):
    # Find K6 in cadj
    # This is harder.
    # Simple approach:
    # Iterate 5-cliques? Or just random sampling?
    # Since we are in a loop, we need it to be fast.
    # Maybe just check if there is a large independent set in G?
    # Equivalent to finding K6 in complement.
    # For n=35, we can try to find a clique of size 6.
    # Heuristic:
    # Pick a random vertex, find its neighbors in cadj, etc.
    # Or:
    # Iterate all 6-subsets? 35C6 is ~1.6M. Too slow per step.
    
    # Better:
    # We can use the fact that we just want *one* K6.
    # If the graph is dense, it's easy.
    # Let's try a greedy clique finder.
    # Or, we can check specific structures.
    
    # Actually, if we just check for K4 in G, and then for K6 in comp,
    # the K6 check is the bottleneck.
    
    # Alternative:
    # Check for independent set of size 6 in G directly.
    # This is the same as K6 in comp.
    
    # Let's implement a fast K6 finder.
    # If n is small, we can use bitsets.
    # To find K6:
    # Iterate i, j, k, l, m such that they form a K5 in cadj, then check if there is a 6th.
    # This is complex.
    
    # Simpler:
    # Just try to find a clique of size 6 using a recursive backtracking with bitsets, limited to a few steps.
    # Or, just check if the independence number is >= 6 using a greedy coloring bound? No, we need exact.
    
    # Let's try a different approach for the search:
    # Instead of checking for K6 every step, we can just add edges randomly and check K4.
    # But we might create an independent set of size 6.
    
    # Let's optimize find_k6.
    # We can precompute common neighbors.
    # For n=35, we can iterate pairs (i,j) in cadj.
    # cn = cadj[i] & cadj[j]
    # If cn has size >= 4, and there is a K4 in the induced subgraph of cn, then (i,j) + K4 is a K6.
    # So we need to find K4 in cadj induced by cn.
    # This is the same as find_k4 but on a subset.
    
    def find_k4_in_subset(adj_subset, mask):
        # adj_subset is the global cadj
        # mask is the set of vertices allowed
        # We need to find 4 vertices in mask that form a K4 in adj_subset
        # Iterate pairs in mask
        # This is still potentially slow if mask is large.
        
        # However, if we just want to find *a* K6, we can be lazy.
        pass
        
    # Let's just use a simple heuristic:
    # If we can't find K6 quickly, we might miss it.
    # But the local search will eventually add edges and break it.
    # The risk is getting stuck in a state with no K4 but with independent set of 6.
    # In that case, we just add a random edge.
    # So we don't strictly *need* to find the K6. We just need to detect that we are stuck?
    # No, if we don't detect the independent set, we might return an invalid graph.
    # So we MUST verify at the end.
    # And in the loop, if we don't find K4, we can just add a random edge to try to break potential independent sets.
    # This is a "blind" move, but it works.
    
    # Revised Loop:
    # 1. Find K4. If found, remove edge.
    # 2. If no K4, check if valid (no independent set of 6).
    #    - To check validity, we can run a full check.
    #    - If invalid, add a random edge.
    #    - If valid, return.
    
    # This avoids the expensive K6 search in the inner loop.
    # The full check for independent set of 6 is also expensive, but we only do it when K4 is gone.
    
    return None

def is_valid(adj, n):
    # Check K4
    if find_k4(adj, n):
        return False
    # Check Independent Set 6
    # Find max independent set size
    # Use a greedy algorithm or branch and bound?
    # For n=35, we can use a simple branch and bound or just check all 6-subsets?
    # 35C6 = 1,623,160.
    # Checking each subset for independence is O(6^2) = 36 ops.
    # Total ~ 60M ops. This might be too slow in Python if done repeatedly.
    # But we only do it when K4 is gone.
    # How often is K4 gone?
    # Maybe rarely.
    
    # Better: Use a heuristic to check if alpha < 6.
    # Or use a bitset-based clique finder on the complement.
    # Let's implement a fast "does alpha >= 6" check.
    # We can use the fact that if there is an independent set of 6, there is a K6 in comp.
    # We can try to find a K6 in comp using a limited search.
    # If we find one, return False.
    # If we don't, we might still have one, but we assume we don't for the sake of the loop?
    # No, we must be sure.
    
    # Let's use a randomized check:
    # Pick random 6-subsets and check if they are independent.
    # If we find one, return False.
    # If we check 1000 random subsets and find none, assume valid?
    # Risky.
    
    # Let's stick to the full check but optimize it.
    # Precompute complement.
    full_mask = (1 << n) - 1
    cadj = [0] * n
    for i in range(n):
        cadj[i] = (full_mask ^ (1 << i)) & ~adj[i]
        
    # Find K6 in cadj
    # We can use a recursive search for clique of size 6.
    # Given n=35, this is feasible if pruned.
    
    def has_clique_size_k(adj_list, k):
        # adj_list is list of ints
        # recursive
        def search(clique, candidates):
            if len(clique) == k:
                return True
            if len(candidates) < k - len(clique):
                return False
            # pick a vertex
            # heuristic: pick vertex with max degree in candidates?
            # for now, just pick first
            v = candidates[0]
            # new candidates are neighbors of v in candidates
            new_cands = []
            for u in candidates[1:]:
                if adj_list[v] & (1 << u):
                    new_cands.append(u)
            if search(clique + [v], new_cands):
                return True
            # try without v?
            # this is exponential.
            return False
            
        # initial candidates
        return search([], list(range(len(adj_list))))
        
    # This is too slow.
    
    # Let's go back to the loop logic.
    # In the loop, if no K4, we add a random edge.
    # We only do the full validity check at the very end, or periodically.
    # If we return a graph, we MUST verify it exactly.
    # The verification function must be exact.
    
    # I will implement an exact verifier using bitsets and a smart K6 search.
    pass
