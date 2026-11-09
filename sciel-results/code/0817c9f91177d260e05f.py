import time
import sys
from collections import defaultdict

# Set recursion limit high to handle DFS on larger graphs if needed, 
# though these graphs are small (n<=30).
sys.setrecursionlimit(10000)

def construct():
    """
    Decomposes three fixed random graphs into a minimum number of pieces (cycles and edges).
    Strategy: Greedy longest-cycle extraction.
    1. Generate the instance graph.
    2. Repeatedly find the longest cycle in the remaining graph using DFS.
    3. Remove the edges of this cycle and add the cycle as a piece.
    4. When no cycles remain, add all remaining edges as individual edge pieces.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(n, adj):
        """
        Finds the longest cycle in the graph defined by adjacency list 'adj' for 'n' nodes.
        Returns a list of vertices forming the cycle, or [] if no cycle exists.
        Uses DFS to explore paths. Since n is small (<=30), we can afford a somewhat 
        exhaustive search for the longest cycle, but we prune aggressively.
        """
        best_cycle = []
        
        # We need to find the longest cycle. 
        # A standard approach is to try starting DFS from each node, 
        # but that might be slow if done naively for every node.
        # However, n=30 is small.
        
        # To optimize, we can just find *a* long cycle. 
        # Greedy: Start from a node, DFS, try to close the loop.
        # A better heuristic for "longest" in small graphs:
        # Try all pairs (u, v) where u < v, and find the longest path from u to v 
        # that doesn't use the edge (u,v) directly? No, that's for Hamiltonian paths.
        
        # Simpler robust approach for n<=30:
        # For each edge (u, v), remove it, find longest path from u to v. 
        # If path length >= 2, then path + (v, u) is a cycle.
        # This finds the cycle containing edge (u,v) with maximum length.
        # We take the max over all edges.
        
        # Optimization: Only check edges that exist.
        # Since adj is built from remaining edges, we can iterate over adj.
        
        # Actually, a simpler DFS for "any" cycle is fast, but we want "longest" to save more pieces.
        # Let's try to find a long cycle by doing DFS from each node, keeping track of visited nodes.
        # If we hit a visited node that is the start, we found a cycle.
        # To find the LONGEST, we need to explore deeper.
        
        # Given the small size, let's use a recursive DFS that tracks the current path.
        # We will try to find the longest cycle by exploring all simple cycles? 
        # That is exponential. 
        # Instead, let's use a heuristic: 
        # 1. Find *any* cycle quickly.
        # 2. Try to extend it?
        
        # Let's stick to a reliable method: 
        # For each edge (u, v), temporarily remove it, find the longest path from u to v.
        # The cycle length is path_len + 1 (for the closing edge).
        # We pick the edge that yields the longest such path.
        
        # To find longest path from u to v in a small graph, we can use DFS with memoization 
        # or just simple DFS with pruning if the graph is sparse.
        
        # Let's implement a helper to find longest path from src to dst avoiding a specific edge.
        
        def longest_path(src, dst, forbidden_edge):
            # DFS
            max_len = -1
            # We want to find the longest path from src to dst.
            # State: (current_node, visited_set)
            # Since n=30, visited_set is a bitmask.
            # We can use memoization: (node, mask) -> max steps to dst? 
            # Or just DFS with pruning.
            
            # Let's use a simple DFS that explores all simple paths from src to dst.
            # This can be slow if the graph is dense, but n=30 and p~0.3-0.4 means it's somewhat sparse.
            # Number of simple paths can be large though.
            
            # Better: Use a recursive function that returns the max depth reached towards dst.
            # Actually, finding the *longest* simple path is NP-hard in general, but for n=30 it's doable with bitmasks if we are careful.
            
            # Let's try a simpler heuristic first: Just find *a* cycle. 
            # The prompt says "Minimize total number of pieces". 
            # A cycle of length 3 saves 2 pieces. A cycle of length 5 saves 4 pieces.
            # So longer is better.
            
            # Let's use a randomized greedy approach or a limited DFS.
            # Given time constraints, let's try to find the longest cycle by 
            # iterating over all edges (u,v), removing it, and doing a DFS from u to v 
            # to find the longest path. 
            # We will limit the DFS depth or use a heuristic to avoid exponential blowup.
            
            # Actually, for n=30, a bitmask DP for longest path is 30 * 2^30 which is too big.
            # But we only need to do this for each edge.
            
            # Alternative: Just find the longest cycle by trying to extend a path.
            
            # Let's try a simpler approach: 
            # 1. Find any cycle using standard cycle detection (DFS with colors).
            # 2. This is fast. 
            # 3. Does "longest" matter that much? 
            #    If we have a 3-cycle and a 5-cycle overlapping, taking the 5-cycle first is better.
            #    Greedy "any cycle" might take a 3-cycle and leave the 5-cycle broken.
            #    So we should try to find longer cycles.
            
            # Heuristic: 
            # Pick a random edge, remove it, try to find a long path back.
            # Or, sort edges by degree sum?
            
            # Let's implement a DFS that finds the longest simple path from src to dst 
            # with a time limit or node limit.
            
            # For simplicity and robustness within time limits, I will use a heuristic:
            # Find the longest cycle by trying each edge as the "closing" edge.
            # For each edge (u,v), find the longest path from u to v in G - (u,v).
            # To find longest path, I'll use a recursive DFS with a visited set.
            # I'll add a pruning condition: if current_path_len + remaining_nodes < best_found, prune.
            
            best = -1
            # To avoid exponential time, I'll limit the search or rely on the sparsity.
            # For n=30, simple DFS on sparse graphs is usually fine.
            
            def dfs(node, path, visited):
                nonlocal best
                if node == dst:
                    if len(path) > best:
                        best = len(path)
                    return
                # Pruning: even if we visit all remaining nodes, can we beat best?
                # remaining = n - len(visited)
                # if len(path) + remaining <= best: return
                # This is a weak bound.
                
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
                        
            # To make this faster, we can order neighbors by degree (descending) to find long paths earlier?
            # Or just standard DFS.
            
            # Let's add a global time check inside the DFS to abort if too slow.
            # But restarting the whole construct is bad.
            # Let's assume n=30 is small enough for simple DFS on sparse graphs.
            # p=0.4, n=20 -> ~190 edges. n=30 -> ~400 edges.
            # Simple paths can be many.
            
            # Optimization: Only explore if we have a chance.
            # Let's just run it. If it's too slow, we'll fall back to finding *any* cycle.
            
            # Actually, let's use a simpler heuristic for "longest":
            # Just find *a* cycle. The difference between a 3-cycle and a 4-cycle is small 
            # compared to the total savings of extracting cycles at all.
            # The biggest gain is going from 0 cycles to many cycles.
            # I will implement a fast "find any cycle" first, and if time permits, try to improve.
            # But "any cycle" is O(E).
            
            # Let's refine: Find the longest cycle by trying to extend a path.
            # I will use a randomized approach: 
            # Pick a start node, do DFS, if we find a cycle, record it.
            # Try multiple starts.
            
            # Let's stick to the edge-removal method for "longest" but with a limit on recursion depth 
            # or just accept that for n=30 it might be okay.
            
            # Actually, there is a simpler way to get good cycles:
            # Use a heuristic to find a cycle. 
            # 1. Start with an arbitrary edge (u,v).
            # 2. Try to find a path from v to u.
            # 3. If found, that's a cycle.
            # 4. To make it longer, we can try to "expand" the cycle?
            
            # Let's implement the "find any cycle" efficiently first.
            # Then, if we have time, we can try to find a longer one by checking adjacent edges.
            
            # For now, let's implement a robust "find longest simple cycle" using 
            # DFS from each node, but with a strict time limit per cycle search.
            
            # Actually, the simplest effective strategy is:
            # 1. Find ANY cycle.
            # 2. Remove it.
            # 3. Repeat.
            # This is guaranteed to be much better than 0 cycles.
            # Finding the *longest* is an optimization.
            
            # Let's implement "find any cycle" using DFS with 3 colors.
            # It returns the cycle path.
            
            # But wait, if I just take *any* cycle, I might take a triangle (3-cycle) 
            # when a 5-cycle is available, leaving the 5-cycle's other edges as single edges.
            # Taking the 5-cycle first would be better.
            
            # Let's try to find the longest cycle by doing DFS from each node 
            # and keeping track of the deepest node reached that is connected back to the start?
            # No, that's not right.
            
            # Let's use the following heuristic for "longest":
            # For each node u, perform a DFS. 
            # If we encounter a back edge to an ancestor v in the DFS tree, 
            # we have a cycle. The length is depth[u] - depth[v] + 1.
            # We can track the maximum such length found during the DFS.
            # This finds the "longest cycle in the DFS tree" which is a good approximation 
            # of the longest cycle in the graph.
            
            # This is O(V+E) per DFS. We do it for each node? No, one DFS from an arbitrary 
            # start covers the connected component. But we might miss cycles not in the tree?
            # No, all cycles in an undirected graph contain at least one back edge in any DFS.
            # So one DFS from any node in the component will find *some* cycles.
            # To find the *longest*, we might need to explore different DFS trees.
            
            # Let's just do one DFS from node 0 (or any node) and find the longest cycle 
            # implied by back edges.
            
            # Implementation:
            # color: 0=white, 1=gray, 2=black
            # parent, depth
            # When we see an edge (u, v) where v is gray (ancestor), 
            # cycle length = depth[u] - depth[v] + 1.
            # We need to reconstruct the path.
            
            # Let's implement this.
            
            n_local = n
            color = [0] * n_local
            parent = [-1] * n_local
            depth = [0] * n_local
            longest_len = 0
            longest_path = []
            
            def dfs_u(u):
                nonlocal longest_len, longest_path
                color[u] = 1
                for v in adj[u]:
                    if color[v] == 0:
                        parent[v] = u
                        depth[v] = depth[u] + 1
                        dfs_u(v)
                    elif color[v] == 1:
                        # Back edge to ancestor v
                        # Cycle is u -> ... -> v -> u
                        curr_len = depth[u] - depth[v] + 1
                        if curr_len > longest_len:
                            longest_len = curr_len
                            # Reconstruct path
                            path = []
                            node = u
                            while node != v:
                                path.append(node)
                                node = parent[node]
                            path.append(v)
                            # The cycle is path + [v] (closing edge is implied by v->u? No, v is ancestor of u, so path is u...v. The edge is (u,v). So cycle is u...v-u)
                            # The list of vertices is path. The edges are (path[0], path[1]), ..., (path[-1], path[0])?
                            # Wait, the back edge is (u, v). The path is u, parent[u], ..., v.
                            # So the cycle vertices are [u, ..., v].
                            # The edges are (u, p1), (p1, p2), ..., (pk, v), and (v, u).
                            # So the list of vertices for the cycle piece is [u, ..., v].
                            # The verifier expects a list of vertices where consecutive pairs are edges, and the last connects to the first.
                            # So if path is [u, a, b, v], edges are (u,a), (a,b), (b,v), (v,u).
                            # This matches.
                            longest_path = path
                    # If color[v] == 2, it's a cross edge or forward edge, ignore for simple cycle detection in undirected graph (already processed)
                color[u] = 2
                
            # Start DFS from all unvisited nodes to handle disconnected components
            for i in range(n_local):
                if color[i] == 0:
                    dfs_u(i)
                    
            return longest_path

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: return all edges as pieces to ensure valid output
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        
        # Build adjacency list
        adj = defaultdict(list)
        remaining_edges = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            remaining_edges.add((u, v))
            
        pieces = []
        
        # Greedy cycle extraction
        # We need to be careful: the DFS above finds the longest cycle in the *current* graph.
        # We remove its edges and repeat.
        
        # To avoid infinite loops or slow execution, we limit the number of iterations.
        # Max cycles <= E / 3.
        
        while True:
            if time.time() > deadline:
                break
                
            if not remaining_edges:
                break
                
            # Check if any cycle exists.
            # The DFS function returns [] if no cycle found (longest_len remains 0).
            cycle = find_longest_cycle(n, adj)
            
            if not cycle:
                break
            
            # Remove edges in the cycle
            # The cycle is a list of vertices [v0, v1, ..., vk].
            # Edges are (v0, v1), (v1, v2), ..., (vk, v0).
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    # Update adj
                    # Remove v from adj[u] and u from adj[v]
                    if v in adj[u]:
                        adj[u].remove(v)
                    if u in adj[v]:
                        adj[v].remove(u)
            
            # Add the cycle as a piece
            # The piece is the list of vertices.
            pieces.append(cycle)
            
        # Add remaining edges as single edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
            
        results.append(pieces)
        
    return results

# To ensure the function name is exactly construct as per the prompt
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has construct().

if __name__ == "__main__":
    # Simple test
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} pieces")
