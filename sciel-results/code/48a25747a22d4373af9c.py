import time
import random
import sys

# Set recursion limit high for DFS
sys.setrecursionlimit(10000)

def construct():
    """
    Constructs a K4-free graph with no independent set of size 6.
    Returns a dict {'n': int, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def time_exceeded():
        return time.time() > deadline

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if u < v:
                    edges.append([u, v])
        return edges

    def count_k4s(adj, n):
        """
        Counts the number of K4 subgraphs.
        A K4 is formed by an edge (u,v) and a triangle in the common neighbors of u and v?
        No. If u~v, and x~y, and x,y are common neighbors of u and v, and x~y, then {u,v,x,y} is a K4.
        Actually, it's simpler:
        Iterate u < v < w < x. Check if all edges exist.
        But that is O(N^4).
        
        Better:
        For each pair (u, v) with u~v:
        Let C = neighbors(u) & neighbors(v).
        For each pair (x, y) in C with x < y:
        If x~y, then {u, v, x, y} is a K4.
        
        This is still potentially heavy if N is large, but for N=35, it's manageable.
        """
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # We need to find edges within 'common'
                    # Iterate through vertices in common
                    # For each x in common, check neighbors of x within common
                    c = common
                    while c:
                        lsb_x = c & -c
                        x = lsb_x.bit_length() - 1
                        c ^= lsb_x
                        # Neighbors of x within common (and strictly greater than x to avoid double counting?)
                        # Actually, if we just check if there is ANY edge in the induced subgraph on 'common',
                        # we can do:
                        # neighbors_of_x_in_common = adj[x] & common
                        # If neighbors_of_x_in_common is not empty, there is an edge.
                        # But we need to count distinct K4s? Or just check existence?
                        # For local search, we want a cost function.
                        # Let's count the number of such edges.
                        # Note: This counts each K4 twice?
                        # If {u,v,x,y} is a K4, edge (u,v) sees edge (x,y) in common.
                        # Edge (u,x) sees edge (v,y) in common.
                        # There are 6 edges in a K4. So each K4 is counted 6 times?
                        # Let's just return the count. We will minimize it.
                        
                        ny = adj[x] & common
                        # Count bits in ny that are > x? No, just count all.
                        # To be precise and avoid double counting edges in the common subgraph:
                        # We can just check if (adj[x] & common) has any bit set.
                        # But for a smoother landscape, counting is better.
                        # Let's count edges in the subgraph induced by 'common'.
                        # We'll do a simple bit count of (adj[x] & common) for x in common?
                        # That counts each edge twice (once for each endpoint).
                        
                        # Efficient check:
                        # Does the subgraph induced by 'common' have any edge?
                        # If yes, add 1 to cost.
                        # This is O(deg^2) roughly.
                        
                        # Optimization: 
                        # precompute? No, adj changes.
                        
                        # Let's just count the number of pairs (x,y) in common such that x~y.
                        # To avoid O(deg^2) loops in Python, we can use bitwise tricks?
                        # Hard to do efficiently in pure Python for arbitrary bitmasks without iterating.
                        # But N=35 is small.
                        
                        # Actually, we can just check if there is ANY edge.
                        # If we just want to minimize violations, binary (0 or 1 per edge pair) is enough.
                        if (adj[x] & common):
                            count += 1
                            # Break? No, we want the total count to guide search.
                            # But checking every x in common is expensive.
                            # Let's refine:
                            # We just need to know if the induced subgraph on 'common' is an independent set or not.
                            # If it is NOT an independent set, it contributes to K4 count.
                            # So we can just check: is 'common' an independent set?
                            # If not, count += 1.
                            # How to check if 'common' is independent fast?
                            # For all x in common, (adj[x] & common) == 0.
                            # This is still a loop.
                            
                            # Let's stick to the loop but break early if we just need existence?
                            # No, for local search, we want a gradient.
                            # However, finding a valid graph is a 0-1 problem.
                            # Let's count the number of "bad" edge pairs (u,v) where common is not independent.
                            
                            # To speed up:
                            # We can sample? No.
                            # Let's just do the check.
                            # The loop `while c` iterates over neighbors.
                            # For each x, we do a bitmask AND and check if non-zero.
                            # This is fast in Python.
                            
                            # We don't need to break, we want the total number of such edges.
                            # But actually, if we just flip one edge, the change in K4 count is local.
                            # For now, let's return the count of edges (u,v) that form a K4 with some (x,y).
                            
                            # Note: This counts each K4 multiple times. That's fine for optimization.
        return count

    def has_independent_set_6(adj, n):
        """
        Checks if there is an independent set of size 6.
        Uses DFS with bitmasks.
        """
        full_mask = (1 << n) - 1
        # Complement adjacency for independent set search:
        # In the complement graph, an independent set of G is a clique.
        # So we look for a clique of size 6 in the complement of G.
        # Comp adjacency: c_adj[i] = ~adj[i] & full_mask & ~(1<<i)
        
        # Precompute comp adj for speed?
        # It changes with every flip. 
        # But we can compute it on the fly or update it.
        # For N=35, computing it is O(N^2) bit ops? No, O(N) bit ops.
        
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
        # DFS for clique of size 6 in comp_adj
        # Standard Bron-Kerbosch with pivot or simple DFS.
        # Given small target size (6), simple DFS is fine.
        
        def dfs(verts, depth):
            if depth == 6:
                return True
            if not verts:
                return False
            # Pruning: if number of bits in verts < 6 - depth, return False
            if verts.bit_count() < 6 - depth:
                return False
                
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in induced subgraph?
            # Or just first available.
            # Let's try first available for simplicity, maybe randomize?
            # For determinism in search, fixed order is better.
            
            v = verts
            while v:
                lsb = v & -v
                u = lsb.bit_length() - 1
                v ^= lsb
                
                # New candidates are neighbors of u in the complement graph
                # that are also in the current 'v' set (remaining candidates)
                # Note: 'v' here is the remaining candidates after removing u.
                new_verts = v & comp_adj[u]
                
                if dfs(new_verts, depth + 1):
                    return True
            return False
        
        return dfs(full_mask, 0)

    def is_valid(adj, n):
        # Check K4
        if count_k4s(adj, n) > 0:
            return False
        # Check Independent 6
        if has_independent_set_6(adj, n):
            return False
        return True

    def random_graph(n, p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def local_search(n, initial_adj, time_limit):
        """
        Improves the graph to be valid if possible.
        Returns the best adj found.
        """
        adj = initial_adj[:]
        best_adj = adj[:]
        best_score = float('inf')
        
        start = time.time()
        
        # Calculate initial score
        # Score = K4_count + 100 * (1 if has_ind_6 else 0)
        # We prioritize removing Ind 6 sets as they are harder to fix?
        # Actually, K4s are easier to count.
        
        def get_score(adj):
            k4_count = count_k4s(adj, n)
            has_ind6 = has_independent_set_6(adj, n)
            return k4_count + (1000 if has_ind6 else 0)

        current_score = get_score(adj)
        if current_score == 0:
            return adj

        # Local search loop
        while time.time() < start + time_limit:
            # Pick a random edge to flip
            # To ensure connectivity of search space, pick random u < v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip edge
            mask = (1 << u) | (1 << v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_score = get_score(adj)
            
            if new_score < current_score:
                current_score = new_score
                best_adj = adj[:]
                best_score = new_score
                if new_score == 0:
                    break
            else:
                # Reject flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                # If stuck in local minimum, maybe do a random walk?
                # For now, simple hill climbing.
                
        return best_adj

    # Main Strategy:
    # 1. Try to find a valid graph for n=35.
    # 2. If found, try n=36.
    # 3. If 36 fails, return 35.
    # 4. If 35 fails, try 34, etc.
    
    best_n = 0
    best_edges = []
    
    # Known good seed?
    # Exoo's graph is 35 vertices.
    # We don't have the explicit edges in memory, so we rely on search.
    # However, we can start with a random graph and search.
    
    # To improve success rate, we can try multiple restarts for each N.
    
    for target_n in [36, 35, 34, 33]:
        if time.time() > deadline:
            break
            
        found_valid = False
        # Number of restarts depends on time
        max_restarts = 10
        time_per_restart = (deadline - time.time()) / max_restarts - 1
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            # Use a slightly biased random graph?
            # Or start with a random graph.
            adj = random_graph(target_n, p=0.5)
            
            # Run local search
            time_limit_for_search = min(time_per_restart, deadline - time.time() - 2)
            if time_limit_for_search < 1:
                break
                
            improved_adj = local_search(target_n, adj, time_limit_for_search)
            
            if is_valid(improved_adj, target_n):
                best_n = target_n
                best_edges = edges_from_adj(improved_adj, target_n)
                found_valid = True
                break
        
        if found_valid:
            # If we found 36, we are done (new record!)
            # If we found 35, we can stop if 36 failed, or continue if time permits?
            # The goal is to MAXIMIZE n.
            # So if we found 35, we should try 36.
            # But we are iterating 36 first.
            # So if we found 36, we return it.
            # If we found 35, we return it (since 36 failed in this pass).
            # But maybe 36 works with different seeds?
            # We will do a second pass for 36 if 35 was found and time remains?
            # Let's structure:
            # 1. Try 36 with high effort.
            # 2. If fail, Try 35 with high effort.
            # 3. Return best.
            
            if target_n == 36:
                # Success!
                break
            elif target_n == 35:
                # We have a 35. Did we try 36? Yes, in previous loop.
                # If we got here, 36 failed.
                break
            else:
                # Found smaller n. Continue to try larger?
                # No, we are iterating descending.
                # So if we found 34, we already failed 35 and 36.
                break

    # If we didn't find 36 or 35, we might have 34 or less.
    # We should ensure we return the best found.
    
    # If best_n is 0, we need a fallback.
    # A trivial graph?
    # n=1 is always valid.
    if best_n == 0:
        best_n = 1
        best_edges = []
    else:
        # Re-verify the best found
        adj = build_adj(best_n, best_edges)
        if is_valid(adj, best_n):
            pass
        else:
            # Fallback to smaller N if verification failed (should not happen)
            for tn in [34, 33, 32, 30, 20]:
                adj = random_graph(tn)
                if is_valid(adj, tn):
                    best_n = tn
                    best_edges = edges_from_adj(adj, tn)
                    break
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    print(construct())
