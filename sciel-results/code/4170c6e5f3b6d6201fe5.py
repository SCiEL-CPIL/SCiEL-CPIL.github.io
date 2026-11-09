import time
import random
import sys

def construct():
    """
    Constructs a triangle-free graph on n vertices with no independent set of 10.
    Strategy: Bit-set accelerated Local Search.
    
    1. We aim to find a graph with n as large as possible (max 40).
    2. We use bitmasks (integers) for adjacency to allow O(1) triangle checks 
       (u-v-w triangle exists iff adj[u] & adj[v] & (1<<w) != 0, or more efficiently 
       when adding edge (u,v), check if adj[u] & adj[v] == 0).
    3. We perform a local search (Simulated Annealing) to minimize the size of the 
       maximum independent set (alpha).
    4. We iterate through n from 40 down to 26. For each n, we run the search.
       If we find a valid graph (alpha < 10), we return it.
    
    Why this is different:
    - The seed program used a specific algebraic formula (Andrasfai variant) which 
      is limited in flexibility.
    - This approach treats the problem as a combinatorial optimization where the 
      state is the edge set, and moves are edge flips.
    - Bitmasks make the "triangle-free" constraint check extremely fast, allowing 
      many more iterations per second than set-based checks.
    """
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    def has_triangle(adj, u, v):
        # Check if adding edge (u,v) creates a triangle.
        # A triangle is formed if there is a w such that w is neighbor of u AND w is neighbor of v.
        # In bitmask terms: adj[u] & adj[v] != 0
        return (adj[u] & adj[v]) != 0

    def get_max_independent_set_size(adj, n, limit=10):
        """
        Returns the size of the maximum independent set, stopping if it reaches 'limit'.
        Uses a recursive branch and bound with bitmasks.
        """
        # Greedy coloring bound is complex to implement with bitmasks on the fly for arbitrary subsets.
        # We use a simple but effective backtracking with pruning.
        # To speed it up, we order vertices by degree (highest first) in the initial candidate set?
        # Actually, for independent set, we want to pick vertices that are NOT adjacent.
        
        # Heuristic: Sort vertices by degree in the current graph? 
        # Or just use the natural order but use bit operations for candidate filtering.
        
        # Let's use a standard DFS with bitmasks.
        # candidates: bitmask of vertices that can still be added to the independent set.
        # current_size: size of current independent set.
        
        def dfs(candidates, current_size):
            if current_size >= limit:
                return limit
            if candidates == 0:
                return current_size
            
            # Pruning: Even if we pick all remaining candidates, we can't reach limit?
            # We need an upper bound on the size of an independent set in the subgraph induced by candidates.
            # A simple bound is popcount(candidates).
            if current_size + candidates.bit_count() < limit:
                return current_size
            
            # Pick a vertex from candidates.
            # Heuristic: Pick the vertex with the highest degree in the induced subgraph?
            # Or just the lowest index?
            # Let's pick the vertex with the highest degree in the original graph to prune faster?
            # Actually, for independent set, picking a vertex with high degree removes more candidates.
            # Let's find the vertex in candidates with the highest degree (popcount(adj[v] & candidates)).
            
            # To avoid O(n) scan for best vertex every time, we can just pick the lowest set bit.
            # But that might be slow. Let's try picking the one with max degree in candidates.
            # For n<=40, scanning 40 bits is fast.
            
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Count neighbors in candidates
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp -= lsb
            
            if best_v == -1:
                return current_size
                
            v_mask = 1 << best_v
            
            # Option 1: Include best_v
            # New candidates: candidates without best_v and without its neighbors
            new_candidates = candidates & ~(v_mask | adj[best_v])
            res1 = dfs(new_candidates, current_size + 1)
            if res1 >= limit:
                return res1
            
            # Option 2: Exclude best_v
            # New candidates: candidates without best_v
            new_candidates2 = candidates & ~v_mask
            res2 = dfs(new_candidates2, current_size)
            
            return max(res1, res2)
        
        # Initial candidates: all vertices
        initial_candidates = (1 << n) - 1
        return dfs(initial_candidates, 0)

    def local_search(n, time_limit):
        """
        Runs local search to find a triangle-free graph on n vertices with alpha < 10.
        Returns (edges_list, alpha) or (None, inf) if not found.
        """
        # Initialize with a random triangle-free graph.
        # We can start with empty graph and add random edges that don't create triangles.
        # Or start with a random graph and remove triangles.
        # Let's start with a random graph where we only add edge (u,v) if adj[u] & adj[v] == 0.
        
        adj = [0] * n
        edges_set = set()
        
        # Random initialization: try to add edges randomly
        # We want a dense graph to minimize alpha.
        # Iterate pairs, add edge with probability p if no triangle.
        # Let's try p=0.5 initially.
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if random.random() < 0.5:
                if not has_triangle(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges_set.add((u, v))
        
        # Check initial alpha
        current_alpha = get_max_independent_set_size(adj, n, 10)
        if current_alpha < 10:
            # Convert to list of lists for output
            edges_list = [[u, v] for u, v in edges_set]
            return edges_list, current_alpha

        # Simulated Annealing / Iterated Local Search
        T = 100.0
        cooling = 0.999
        min_T = 0.1
        steps = 0
        best_alpha = current_alpha
        best_adj = adj[:]
        best_edges_set = set(edges_set)
        
        while T > min_T:
            if time.time() > time_limit:
                break
                
            # Pick a random edge to toggle
            # We can pick a random pair (u, v)
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            u_mask = 1 << u
            v_mask = 1 << v
            
            is_edge = (adj[u] & v_mask) != 0
            
            if is_edge:
                # Try to remove edge
                # Removing an edge never creates a triangle, so it's always valid.
                # But it might increase alpha (bad).
                new_alpha = get_max_independent_set_size(adj, n, 10) # This is expensive, we should estimate or use fast check
                # Actually, removing an edge can only increase alpha or keep it same.
                # We need to check if alpha becomes >= 10.
                # Since current_alpha < 10 is the goal, and we are in a state where alpha might be >= 10 (if we started bad) or < 10 (if we found one).
                # Wait, the loop above returns immediately if current_alpha < 10.
                # So we are only in the search loop if current_alpha >= 10.
                
                # If we remove an edge, alpha might increase. We want to DECREASE alpha.
                # So removing edges is generally a bad move for minimizing alpha, unless it allows us to add other edges later?
                # Actually, in triangle-free graphs, adding edges is the main way to reduce alpha.
                # So we should bias towards ADDING edges.
                
                # Let's restart the logic:
                # We want to minimize alpha.
                # Adding an edge reduces alpha (or keeps it same).
                # Removing an edge increases alpha (or keeps it same).
                # So we should primarily try to ADD edges.
                
                # If we are currently in a state with alpha >= 10, we MUST add edges to reduce it.
                # So let's only try to ADD edges.
                # If we try to add an edge that creates a triangle, we skip.
                # If we try to add an edge that is already there, we skip.
                
                # So, if is_edge is True, we skip this move (or try to remove it with low probability to escape local minima, but that increases alpha).
                # Let's just skip removals for now, as they are counter-productive for the objective.
                continue
            
            else:
                # Try to add edge (u, v)
                if has_triangle(adj, u, v):
                    continue # Cannot add
                
                # Calculate new alpha
                # We can do a fast check: Does adding this edge reduce the max independent set?
                # It's hard to know without full calculation.
                # Let's just calculate it.
                # Optimization: If current_alpha < 10, we would have returned. So current_alpha >= 10.
                
                # Apply edge
                adj[u] |= v_mask
                adj[v] |= u_mask
                edges_set.add((u, v))
                
                new_alpha = get_max_independent_set_size(adj, n, 10)
                
                # Revert for decision
                adj[u] &= ~v_mask
                adj[v] &= ~u_mask
                edges_set.discard((u, v))
                
                delta = new_alpha - current_alpha
                
                # Accept if delta < 0 (improvement) or with probability exp(-delta/T)
                if delta < 0:
                    # Accept
                    adj[u] |= v_mask
                    adj[v] |= u_mask
                    edges_set.add((u, v))
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_adj = adj[:]
                        best_edges_set = set(edges_set)
                        if best_alpha < 10:
                            # Found a valid graph
                            edges_list = [[u, v] for u, v in best_edges_set]
                            return edges_list, best_alpha
                else:
                    # Deterioration
                    if random.random() < pow(2.71828, -delta / T):
                        # Accept
                        adj[u] |= v_mask
                        adj[v] |= u_mask
                        edges_set.add((u, v))
                        current_alpha = new_alpha
                        
            T *= cooling
            steps += 1
            
            # Random restarts if stuck?
            # For now, simple cooling.
            
        # If we didn't find alpha < 10, return the best found (which is >= 10, so invalid)
        # But we only return if valid.
        return None, best_alpha

    # Try to find the largest n
    best_n = 26
    best_edges = None
    
    # Fallback: Andrasfai variant for n=26 (from seed)
    def construct_andrasfai_variant(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return edges
    
    # Initialize with seed's best
    best_edges = construct_andrasfai_variant(26)
    
    # Try larger n
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Give more time to smaller n if larger ones fail?
        # Or just try each with a fixed time budget.
        # 40 is very hard. 30 is easier.
        # Let's allocate time based on n.
        # Total time ~ 140s.
        # 15 values of n.
        # Let's try 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27.
        # 14 values.
        # 140 / 14 = 10s per n.
        
        time_budget = 10.0
        if n > 35:
            time_budget = 15.0
        elif n < 30:
            time_budget = 5.0
            
        start_n = time.time()
        edges, alpha = local_search(n, start_n + time_budget)
        
        if edges is not None:
            best_n = n
            best_edges = edges
            # If we found a valid graph for n, we can try n+1? 
            # No, we are iterating downwards. 
            # If we found valid for 30, we should have tried 40..31 first.
            # So if we get here, 40..n+1 failed. n is the best.
            # We can break and return n.
            break
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found n={res['n']}")
