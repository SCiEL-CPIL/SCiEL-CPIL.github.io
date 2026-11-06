import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # We will use a local search / hill-climbing approach to find a large
    # K4-free graph with independence number <= 5.
    #
    # Strategy:
    # 1. Start with a known good seed (e.g., the 17-vertex Paley graph or similar).
    # 2. Try to add vertices one by one, finding a valid adjacency for the new vertex.
    # 3. Use local search (edge flips) to repair the graph if it becomes invalid,
    #    or to find a configuration that allows adding more vertices.
    #
    # Actually, a more robust approach for this specific problem (Ramsey R(4,6))
    # is to use a randomized local search on the edge set of a graph with n vertices.
    # We maximize n by starting with a small valid graph and growing it, or by
    # searching for a valid graph of size n and incrementing n.
    #
    # Given the time limit, let's try a specific construction based on known
    # Ramsey graphs. The Exoo graph is 35 vertices. Can we find it?
    # It's hard to hardcode 35 vertices without knowing them.
    #
    # Let's try a heuristic search:
    # 1. Generate a random graph with n vertices.
    # 2. Check if it is K4-free and alpha < 6.
    # 3. If not, modify edges (flip edges) to reduce violations.
    # 4. If valid, try to add a vertex.
    #
    # A better heuristic:
    # Start with n=20.
    # While time remains:
    #   - Try to find a valid graph of size n.
    #   - If found, increment n and restart search for n+1.
    #   - If not found in a sub-time-limit, decrement n or try different seed.
    
    # Let's implement a function to check validity efficiently.
    def is_valid(n, edges):
        # Build adjacency matrix as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check K4
        # For each pair (u, v), if edge exists, check if they have >= 2 common neighbors.
        # If so, K4 exists.
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Count bits
                    if bin(common).count('1') >= 2:
                        return False
        
        # Check Independent Set of size 6
        # We need to verify alpha(G) < 6.
        # This is NP-hard, but for n <= 40 we can use backtracking or heuristics.
        # A simple backtracking for independent set of size 6.
        
        # Optimization: If the graph is dense, alpha is small.
        # If the graph is sparse, alpha is large.
        # We want alpha <= 5.
        
        # Let's use a greedy coloring or a specific backtracking.
        # Backtracking for independent set of size k=6.
        
        # Sort vertices by degree (ascending) to prune?
        # Actually, for independent set, we want to pick vertices that are not adjacent.
        
        # Let's use a recursive function with memoization or just pruning.
        # Since n is small, we can try all subsets? No, 40C6 is large.
        
        # Better: Check if there is an independent set of size 6.
        # We can use the fact that if we remove a vertex, the max independent set
        # of the rest must be < 6 if we include it, etc.
        
        # Let's implement a simple branch and bound for alpha >= 6.
        
        # Precompute non-neighbor masks for speed?
        # non_adj[v] = ~adj[v] & ((1<<n)-1)
        
        non_adj = [0] * n
        mask_all = (1 << n) - 1
        for v in range(n):
            non_adj[v] = mask_all ^ adj[v]
            # Remove self
            non_adj[v] &= ~(1 << v)
        
        def has_independent_set(k, vertices_mask):
            if k == 0:
                return True
            if vertices_mask == 0:
                return False
            
            # Heuristic: Pick a vertex with high degree in the induced subgraph?
            # Or just pick the first available vertex.
            # Let's pick a vertex v from vertices_mask.
            # To prune, we can bound the max independent set size.
            # A simple bound: if popcount(vertices_mask) < k, return False.
            if bin(vertices_mask).count('1') < k:
                return False
            
            # Pick a vertex v. Let's pick the one with highest degree in the subgraph?
            # Or just the lowest index for determinism in search, but random is better for search.
            # For validity check, we just need to know if it exists.
            # Let's pick a vertex v.
            # To be efficient, we can use a greedy coloring bound, but that's complex.
            # Let's just pick the first set bit.
            v = (vertices_mask & -vertices_mask).bit_length() - 1
            # Wait, bit_length of a power of 2 gives index+1.
            # If mask is 1<<v, bit_length is v+1. So v = bit_length - 1.
            
            # Option 1: Include v.
            # Then we can only include non-neighbors of v from the remaining vertices.
            new_mask = vertices_mask & non_adj[v] & ~(1 << v)
            if has_independent_set(k - 1, new_mask):
                return True
            
            # Option 2: Exclude v.
            new_mask2 = vertices_mask & ~(1 << v)
            if has_independent_set(k, new_mask2):
                return True
            
            return False
        
        if has_independent_set(6, mask_all):
            return False
        
        return True

    # Initial seed: Paley graph of order 17?
    # Paley(17) is a (17, 8, 3, 4) design? No, Paley graph of order p=17.
    # Edges (u,v) if (v-u) is a quadratic residue mod 17.
    # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16.
    
    def generate_paley(p):
        # p must be prime, p % 4 == 1
        residues = set()
        for i in range(1, p):
            residues.add((i * i) % p)
        edges = []
        for u in range(p):
            for v in range(u + 1, p):
                if (v - u) % p in residues:
                    edges.append([u, v])
        return {'n': p, 'edges': edges}

    # Let's start with n=20 and try to grow.
    # But first, let's see if we can find a valid graph for n=20, 21, etc.
    
    # A better strategy:
    # Start with a random graph of size n.
    # Apply local search to make it K4-free and alpha<6.
    # If successful, increment n.
    
    # Let's try to find a valid graph for a given n using local search.
    def find_valid_graph(n, max_iters=5000):
        # Initialize with a random graph
        # Probability of edge?
        # For K4-free, we want sparse. For alpha<6, we want dense.
        # Let's try p=0.5 initially.
        
        # Actually, let's start from a known structure or random.
        # Random edges
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    edges.add((u, v))
        
        # Local search
        # Violation count:
        # 1. Number of K4s.
        # 2. 1 if alpha >= 6 else 0.
        
        # We want to minimize violations.
        
        # To speed up, we can track K4s more efficiently?
        # For now, just recompute.
        
        best_edges = None
        best_score = float('inf')
        
        current_edges = set(edges)
        
        # Check initial
        if is_valid(n, list(current_edges)):
            return list(current_edges)
        
        for i in range(max_iters):
            # Pick a random edge to flip
            # Or pick a random pair and flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            if (u, v) in current_edges:
                current_edges.remove((u, v))
            else:
                current_edges.add((u, v))
            
            # Check validity
            # This is expensive. Let's optimize is_valid or use a proxy.
            # Proxy: Check K4s involving the changed edge?
            # And check alpha?
            
            # For now, full check.
            # To make it faster, we can first check K4s.
            # If K4s exist, we can count them.
            # If no K4s, check alpha.
            
            # Let's create a faster check for K4s.
            adj = [0] * n
            for a, b in current_edges:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
            
            k4_count = 0
            for a in range(n):
                for b in range(a + 1, n):
                    if (adj[a] >> b) & 1:
                        common = adj[a] & adj[b]
                        # Count bits
                        c = bin(common).count('1')
                        if c >= 2:
                            # Each K4 is counted multiple times?
                            # A K4 has 6 edges. Each edge has 2 common neighbors.
                            # So each K4 contributes 6 * 2 = 12 to the sum?
                            # No, for each edge in the K4, the other 2 vertices are common neighbors.
                            # So each K4 is detected 6 times (once per edge).
                            k4_count += 1
            
            if k4_count > 0:
                # Revert if it made things worse?
                # We need a score.
                # Score = k4_count + (1 if alpha>=6 else 0)
                # But checking alpha is expensive.
                # Let's assume if k4_count > 0, it's bad.
                # If k4_count == 0, check alpha.
                
                # To avoid expensive alpha check every time,
                # we can only check alpha if k4_count == 0.
                
                # But we need to compare scores.
                # Let's just do:
                # If k4_count > 0, score is high.
                # If k4_count == 0, check alpha.
                
                # This is tricky.
                # Let's just run is_valid if k4_count == 0.
                # If k4_count > 0, we can estimate score.
                
                # Actually, let's just run is_valid.
                # If it's too slow, we'll see.
                
                # Optimization: If k4_count > 0, we don't need to check alpha.
                # But we need a comparable score.
                # Let's define score = k4_count * 10 + (1 if alpha>=6 else 0).
                # But we don't know alpha if k4_count > 0.
                # Let's assume alpha check is only done if k4_count == 0.
                # If k4_count > 0, we just use k4_count as score.
                
                # This is inconsistent.
                
                # Let's stick to is_valid for correctness, but optimize the K4 check inside is_valid.
                # Actually, the K4 check in is_valid is already O(n^2 * bitcount).
                # The alpha check is the bottleneck.
                
                # Let's modify is_valid to return early if K4 found.
                # And for local search, we want a gradient.
                
                # Let's try a different approach:
                # Only flip edges that are part of a K4 or part of an independent set.
                
                # For now, let's just use is_valid.
                # If it's too slow, we'll reduce max_iters.
                
                pass # Placeholder
            
            # Re-evaluate
            if is_valid(n, list(current_edges)):
                return list(current_edges)
            
            # If not valid, we might want to keep the change if it reduced violations.
            # But we don't track violations easily.
            # Let's just random walk.
            # If we don't find a valid graph, we return None.
            
            # To improve, we can use simulated annealing.
            # But let's keep it simple.
            
            # Revert if we want to stay near the initial? No, random walk is fine.
        
        return None

    # Let's try to find a graph for n=20, then 21, etc.
    best_n = 0
    best_graph = None
    
    # Start with a known valid graph if possible.
    # Paley(17) is K4-free?
    # Paley(17) has clique number 3? Yes, Paley graphs of order p=4k+1 have omega < sqrt(p).
    # sqrt(17) ~ 4.1. So omega can be 3 or 4.
    # Actually, Paley(17) is strongly regular (17, 8, 3, 4).
    # Lambda=3, Mu=4.
    # If there is a K4, it has 6 edges.
    # In a K4, every pair has 2 common neighbors within the K4.
    # But lambda=3 means any edge has 3 common neighbors.
    # So a K4 could exist.
    # Let's check Paley(17) validity.
    
    g17 = generate_paley(17)
    if is_valid(17, g17['edges']):
        best_n = 17
        best_graph = g17
    
    # Try to grow from 17 to 35.
    # We can try to add a vertex to the current graph.
    # To add a vertex v, we need to choose its neighbors such that:
    # 1. No K4 is formed.
    # 2. No independent set of size 6 is formed.
    
    # This is a constraint satisfaction problem.
    # We can use local search to find the neighbors of the new vertex.
    
    def try_add_vertex(graph, n_old, n_new):
        # n_new = n_old + 1
        # New vertex is n_old.
        # We need to assign neighbors for n_old.
        # Let's try all subsets of neighbors? 2^35 is too big.
        # Let's use local search on the neighbors of the new vertex.
        
        # Initial guess: random neighbors.
        neighbors = set()
        for u in range(n_old):
            if random.random() < 0.5:
                neighbors.add(u)
        
        edges = set(graph['edges'])
        
        # Add edges for new vertex
        for u in neighbors:
            edges.add((u, n_old))
        
        # Check validity
        if is_valid(n_new, list(edges)):
            return list(edges)
        
        # Local search on neighbors
        for i in range(1000):
            # Flip a neighbor
            u = random.randint(0, n_old - 1)
            if u in neighbors:
                neighbors.remove(u)
                edges.remove((u, n_old))
            else:
                neighbors.add(u)
                edges.add((u, n_old))
            
            if is_valid(n_new, list(edges)):
                return list(edges)
        
        return None

    # Try to grow
    current_graph = best_graph
    current_n = best_n
    
    # We need to be careful with time.
    # Let's try to grow up to 35.
    
    for target_n in range(current_n + 1, 36):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple times to add a vertex
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            new_edges = try_add_vertex(current_graph, current_n, target_n)
            if new_edges is not None:
                current_graph = {'n': target_n, 'edges': new_edges}
                current_n = target_n
                found = True
                break
        
        if not found:
            # Try a different strategy?
            # Maybe restart with a random graph of size target_n?
            # Let's try find_valid_graph(target_n)
            new_edges = find_valid_graph(target_n, max_iters=2000)
            if new_edges is not None:
                current_graph = {'n': target_n, 'edges': new_edges}
                current_n = target_n
                found = True
        
        if not found:
            # If we can't grow, maybe we should try to improve the current graph
            # or stop.
            # Let's try a few more random restarts for the current target_n.
            break

    # Final check
    if current_graph is not None and is_valid(current_n, current_graph['edges']):
        return current_graph
    
    # Fallback
    return g17

# Run construct
if __name__ == "__main__":
    # This block is for local testing only
    result = construct()
    print(result['n'])
