import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_check():
        return time.time() > deadline

    # Helper: Check if a coloring of K_n has any monochromatic C6
    # Returns the number of monochromatic simple C6s
    def count_c6s(colors, n):
        # Build adjacency lists for each color
        adj = [set() for _ in range(4)]
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                adj[c].add((j, i))
                edges.append((i, j))
                idx += 1
        
        c6_count = 0
        for c in range(4):
            # For each pair of non-adjacent vertices (u, v) in color c,
            # check if there is a path of length 4 between them in color c
            # that doesn't use u or v internally.
            # Actually, simpler: enumerate all 6-subsets? No, too slow for n=18/20.
            # Use matrix multiplication for walks, then subtract non-simple walks?
            # For n<=20, we can just do a BFS/DFS from each vertex to find cycles.
            # Or, for each color c, build adjacency matrix A_c.
            # A simple C6 is a cycle of length 6.
            # Number of closed walks of length 6 is trace(A^6).
            # This includes:
            # 1. Simple C6s (each counted 12 times? 6 start nodes * 2 directions)
            # 2. Triangles traversed twice (each counted 6 times? 3 start nodes * 2 dirs * 2 loops?)
            # 3. Back-and-forth paths (u-v-u-v-u-v)
            
            # Let's use a direct enumeration for small n to be exact.
            # n is at most 20. Number of 6-subsets is C(20,6) = 38760.
            # For each 6-subset, check if it forms a C6 in color c.
            # This is feasible.
            
            vertices = list(range(n))
            for subset in combinations(vertices, 6):
                # Check if the induced subgraph on subset has a Hamiltonian cycle of color c
                # Since it's a complete graph K_6, any permutation of the 6 vertices
                # that forms a cycle and all edges are in color c is a C6.
                # Actually, we just need to know if the subgraph G[subset] contains a C6.
                # In a complete graph, the subgraph is defined by which edges are color c.
                # We can check if the adjacency matrix of the induced subgraph has a cycle of length 6.
                
                # Faster check: 
                # A 6-cycle exists if there are 6 vertices v0..v5 such that
                # (v0,v1), (v1,v2), ..., (v5,v0) are all in color c.
                
                # We can iterate through permutations? 6! = 720.
                # 38760 * 720 * 4 colors is too slow.
                
                # Better: Use the property that a C6 in a complete graph is just a cycle.
                # Let's stick to the matrix method but correct for non-simple walks?
                # No, let's use a simple BFS-based cycle detection for each color.
                
                # For each color c, let's just check if there is ANY C6.
                # We can do this by checking all 6-tuples of distinct vertices.
                # C(n,6) * 5! / 2 is the number of unique cycles in K_n.
                # For n=20, C(20,6) = 38760. 5! = 120. 38760 * 120 / 2 = 2.3 million cycles to check per color.
                # 4 colors * 2.3M = 9.2M checks. Each check is 6 edge lookups.
                # This is fast enough in Python if optimized or using NumPy?
                # Maybe borderline. Let's try to optimize the check.
                
                pass
        
        # Alternative fast check for small n:
        # Use numpy matrix powers and subtract known non-simple components?
        # Or just use the subset enumeration but break early if found.
        
        # Let's implement a fast "exists C6" check for one color.
        # If any color has a C6, return 1 (or count).
        
        def has_c6_in_color(adj_set, n):
            # adj_set is a set of edges (u,v) for this color
            # We want to find v0, v1, v2, v3, v4, v5 distinct such that
            # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) in adj_set
            
            # Optimization: Pick v0. Then v1 must be a neighbor.
            # This is essentially checking for a cycle of length 6 in the graph defined by adj_set.
            
            # For n=20, we can just use a simple DFS from each node to find cycles of length 6.
            # Or, use the fact that we only need to know IF one exists.
            
            # Let's use a randomized check if exact is too slow? No, must be exact.
            
            # Let's use the subset method but optimized.
            # Precompute neighbors for this color
            neighbors = [set() for _ in range(n)]
            for (u, v) in adj_set:
                neighbors[u].add(v)
                neighbors[v].add(u)
                
            # Iterate over all 6-subsets
            # To speed up, we can just check if the induced subgraph has a cycle of length 6.
            # A graph on 6 vertices has a C6 if and only if it has a Hamiltonian cycle.
            # We can check this by trying all permutations of the 6 vertices?
            # No, just check if the graph contains a cycle of length 6.
            
            # Simpler: Just check all 6-tuples (v0..v5) in order?
            # We can fix v0=0 to reduce symmetry? No, v0 can be any vertex.
            
            # Let's just use the matrix method trace(A^6) and assume no false positives?
            # Risky.
            
            # Let's do this:
            # For each pair (u, v) in the color, count paths of length 4 between u and v
            # that do not use u or v internally.
            # If any such path exists, we have a C6 (u-v-path-v-u).
            
            # Let A be the adjacency matrix for this color.
            # A^2 counts paths of length 2.
            # A^4 counts paths of length 4.
            # (A^4)[u,v] is the number of walks of length 4 from u to v.
            # We need to subtract walks that visit u or v internally or repeat vertices.
            # This is complex.
            
            # Given n <= 20, let's just use a simple BFS/DFS for each starting node to find a cycle of length 6.
            # Since we only need to find ONE, we can stop early.
            
            for start in range(n):
                # DFS from start, looking for a path of length 5 ending at a neighbor of start
                # Path: start -> v1 -> v2 -> v3 -> v4 -> v5 -> start
                # v1..v5 must be distinct and not start.
                
                # We can use a recursive search with pruning.
                # Depth 5. Branching factor is degree.
                # Max degree 19. 19^5 is 2.4M. Too slow per start node.
                
                # We need a better way.
                
                # Let's use the subset enumeration but in C-like speed?
                # Python is slow.
                
                # Let's try a different approach:
                # If the graph is dense, C6s are likely.
                # If the graph is sparse, we can check quickly.
                
                # For the specific task of R_4(C6), the graphs are quite dense (1/4 of edges).
                # Average degree ~ 4.5 for n=18.
                # 4.5^5 = 1800. 18 * 1800 = 32k operations per color. Very fast.
                
                def dfs(u, depth, path, visited):
                    if depth == 5:
                        # Check if u is connected to start
                        if start in neighbors[u]:
                            return True
                        return False
                    
                    for v in neighbors[u]:
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            if dfs(v, depth + 1, path, visited):
                                return True
                            path.pop()
                            visited.remove(v)
                    return False
                
                visited = {start}
                for v1 in neighbors[start]:
                    visited.add(v1)
                    if dfs(v1, 1, [v1], visited):
                        return True
                    visited.remove(v1)
                        
            return False
            
        for c in range(4):
            if has_c6_in_color(adj[c], n):
                return 1 # Just return 1 if any found, or count? 
                         # For local search, we want the count or a proxy.
                         # Let's return the number of colors that have a C6.
                         # Or better, a penalty score.
        
        return 0

    # --- Strategy ---
    
    # 1. Find a valid n=17 coloring quickly.
    # Use Cayley graph on Z_17.
    # Differences: 1..8.
    # We need to partition {1..8} into 4 sets D0, D1, D2, D3 such that
    # the Cayley graph with difference set Di has no C6.
    
    # Known valid partition for n=17?
    # Let's try to find one by random search.
    # There are 4^8 = 65536 possibilities. We can check them all quickly.
    
    def check_cayley_c6(diff_sets, n=17):
        # diff_sets: list of 4 sets of differences
        # Build adjacency for each color
        adj = [set() for _ in range(4)]
        for c in range(4):
            for d in diff_sets[c]:
                for u in range(n):
                    v = (u + d) % n
                    if u < v:
                        adj[c].add((u, v))
                    else:
                        adj[c].add((v, u))
        
        for c in range(4):
            if has_c6_in_color(adj[c], n):
                return False
        return True

    best_n = 0
    best_colors = None

    # Try to find valid n=17 Cayley coloring
    found_17 = False
    # Iterate through all 4^8 assignments of differences 1..8 to colors 0..3
    # We can prune: difference d and n-d are the same edge set.
    # So we only assign colors to 1..8.
    
    # Optimization: Use symmetry. The sets are symmetric.
    # Let's just iterate.
    
    # To speed up, we can pre-compute the adjacency for each difference set?
    # Or just build it on the fly.
    
    # Let's use a smarter search: Local search on the 8 differences.
    # Start with random assignment.
    # Check validity.
    # If invalid, flip a difference's color.
    
    # Actually, 65536 is small. Let's just brute force it with a fast check.
    # But has_c6_in_color is the bottleneck.
    
    # Let's optimize has_c6_in_color for n=17.
    # The DFS above is fine.
    
    # Let's try a few random seeds for n=17 Cayley.
    for seed in range(100):
        if time_check(): break
        random.seed(seed)
        diff_assign = [random.randint(0, 3) for _ in range(8)]
        diff_sets = [set() for _ in range(4)]
        for i in range(8):
            diff_sets[diff_assign[i]].add(i + 1)
        
        if check_cayley_c6(diff_sets, 17):
            # Build the full coloring
            colors = []
            for i in range(17):
                for j in range(i + 1, 17):
                    d = (j - i) % 17
                    d = min(d, 17 - d)
                    # Find which set contains d
                    for c in range(4):
                        if d in diff_sets[c]:
                            colors.append(c)
                            break
            best_n = 17
            best_colors = colors
            found_17 = True
            break
            
    if not found_17:
        # If we didn't find it in 100 random tries, do a full search?
        # Or just fall back to a known one if we had one.
        # Let's do a full search if time permits.
        if not time_check():
            for mask in range(4**8):
                if time_check(): break
                diff_assign = []
                temp = mask
                for _ in range(8):
                    diff_assign.append(temp % 4)
                    temp //= 4
                diff_sets = [set() for _ in range(4)]
                for i in range(8):
                    diff_sets[diff_assign[i]].add(i + 1)
                
                if check_cayley_c6(diff_sets, 17):
                    colors = []
                    for i in range(17):
                        for j in range(i + 1, 17):
                            d = (j - i) % 17
                            d = min(d, 17 - d)
                            for c in range(4):
                                if d in diff_sets[c]:
                                    colors.append(c)
                                    break
                    best_n = 17
                    best_colors = colors
                    found_17 = True
                    break
    
    # 2. Try to find n=18
    # This is harder. Z_18 is not prime.
    # We can try random colorings and local search.
    # Or try Cayley on Z_18?
    # Differences 1..9.
    # 4^9 = 262144.
    
    if not time_check() and best_n < 18:
        # Try random Cayley on Z_18
        for seed in range(500):
            if time_check(): break
            random.seed(seed)
            diff_assign = [random.randint(0, 3) for _ in range(9)]
            diff_sets = [set() for _ in range(4)]
            for i in range(9):
                diff_sets[diff_assign[i]].add(i + 1)
            
            # Check validity for n=18
            adj = [set() for _ in range(4)]
            valid = True
            for c in range(4):
                for d in diff_sets[c]:
                    for u in range(18):
                        v = (u + d) % 18
                        if u < v:
                            adj[c].add((u, v))
                        else:
                            adj[c].add((v, u))
                
                if has_c6_in_color(adj[c], 18):
                    valid = False
                    break
            
            if valid:
                colors = []
                for i in range(18):
                    for j in range(i + 1, 18):
                        d = (j - i) % 18
                        d = min(d, 18 - d)
                        for c in range(4):
                            if d in diff_sets[c]:
                                colors.append(c)
                                break
                best_n = 18
                best_colors = colors
                break

    # 3. If n=18 not found, try local search on n=18 random coloring
    if best_n < 18 and not time_check():
        n = 18
        edges_count = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(edges_count)]
        
        # Local search
        T = 10.0
        for it in range(5000):
            if time_check(): break
            
            # Pick random edge
            idx = random.randint(0, edges_count - 1)
            # Find u, v for idx
            # We need a mapping.
            # For simplicity, just pick random u, v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            # Calculate idx
            # idx = sum_{k=0}^{u-1} (n - 1 - k) + (v - u - 1)
            # This is O(n).
            
            # Better: precompute edge indices
            # But we are in a loop.
            
            # Let's just use the full color array and pick a random index, then find u,v
            # Or just pick u,v and calculate index.
            
            # Index calculation:
            # Edges are stored in lexicographic order.
            # (0,1), (0,2)...(0,n-1), (1,2)...
            # Count edges before row u: u * (n - 1) - u * (u + 1) / 2
            # Edges in row u before v: v - u - 1
            idx = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
            
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c: continue
            
            colors[idx] = new_c
            new_cost = count_c6s(colors, n)
            
            if new_cost == 0:
                best_n = 18
                best_colors = colors
                break
                
            # Simple hill climbing: accept if cost reduced
            # We don't have old_cost efficiently.
            # Let's just accept if new_cost < 1 (i.e., 0).
            # Or keep a score.
            
            # For now, just revert if not 0?
            # This is very greedy.
            colors[idx] = old_c

    # Return best found
    if best_colors is None:
        # Fallback: return a small valid coloring, e.g., n=1
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}

# Helper function used inside construct
def has_c6_in_color(adj_set, n):
    # adj_set is a set of edges (u,v) for this color
    # We want to find v0, v1, v2, v3, v4, v5 distinct such that
    # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) in adj_set
    
    # Precompute neighbors
    neighbors = [set() for _ in range(n)]
    for (u, v) in adj_set:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    # DFS from each start node
    for start in range(n):
        # We look for a path of length 5 from start to a neighbor of start
        # start -> v1 -> v2 -> v3 -> v4 -> v5 -> start
        
        # Use iterative deepening or simple recursion
        # Since n is small, recursion is fine.
        
        def dfs(u, depth, visited):
            if depth == 5:
                # Check if u is connected to start
                return start in neighbors[u]
            
            for v in neighbors[u]:
                if v not in visited:
                    visited.add(v)
                    if dfs(v, depth + 1, visited):
                        return True
                    visited.remove(v)
            return False
        
        visited = {start}
        for v1 in neighbors[start]:
            visited.add(v1)
            if dfs(v1, 1, visited):
                return True
            visited.remove(v1)
            
    return False

# Note: The function has_c6_in_color is defined inside construct in the prompt's style, 
# but I defined it outside here. I should move it inside or keep it global.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program had construct() as the main entry.
# I will put has_c6_in_color inside construct to be safe and self-contained.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_check():
        return time.time() > deadline

    def has_c6_in_color(adj_set, n):
        neighbors = [set() for _ in range(n)]
        for (u, v) in adj_set:
            neighbors[u].add(v)
            neighbors[v].add(u)
            
        for start in range(n):
            def dfs(u, depth, visited):
                if depth == 5:
                    return start in neighbors[u]
                for v in neighbors[u]:
                    if v not in visited:
                        visited.add(v)
                        if dfs(v, depth + 1, visited):
                            return True
                        visited.remove(v)
                return False
            
            visited = {start}
            for v1 in neighbors[start]:
                visited.add(v1)
                if dfs(v1, 1, visited):
                    return True
                visited.remove(v1)
        return False

    def count_c6s(colors, n):
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                adj[c].add((j, i))
                idx += 1
        
        count = 0
        for c in range(4):
            if has_c6_in_color(adj[c], n):
                count += 1
        return count

    best_n = 0
    best_colors = None

    # 1. Find n=17 Cayley
    for seed in range(200):
        if time_check(): break
        random.seed(seed)
        diff_assign = [random.randint(0, 3) for _ in range(8)]
        diff_sets = [set() for _ in range(4)]
        for i in range(8):
            diff_sets[diff_assign[i]].add(i + 1)
        
        adj = [set() for _ in range(4)]
        valid = True
        for c in range(4):
            for d in diff_sets[c]:
                for u in range(17):
                    v = (u + d) % 17
                    if u < v:
                        adj[c].add((u, v))
                    else:
                        adj[c].add((v, u))
            
            if has_c6_in_color(adj[c], 17):
                valid = False
                break
        
        if valid:
            colors = []
            for i in range(17):
                for j in range(i + 1, 17):
                    d = (j - i) % 17
                    d = min(d, 17 - d)
                    for c in range(4):
                        if d in diff_sets[c]:
                            colors.append(c)
                            break
            best_n = 17
            best_colors = colors
            break

    # 2. Try n=18 Cayley
    if best_n < 18 and not time_check():
        for seed in range(1000):
            if time_check(): break
            random.seed(seed)
            diff_assign = [random.randint(0, 3) for _ in range(9)]
            diff_sets = [set() for _ in range(4)]
            for i in range(9):
                diff_sets[diff_assign[i]].add(i + 1)
            
            adj = [set() for _ in range(4)]
            valid = True
            for c in range(4):
                for d in diff_sets[c]:
                    for u in range(18):
                        v = (u + d) % 18
                        if u < v:
                            adj[c].add((u, v))
                        else:
                            adj[c].add((v, u))
                
                if has_c6_in_color(adj[c], 18):
                    valid = False
                    break
            
            if valid:
                colors = []
                for i in range(18):
                    for j in range(i + 1, 18):
                        d = (j - i) % 18
                        d = min(d, 18 - d)
                        for c in range(4):
                            if d in diff_sets[c]:
                                colors.append(c)
                                break
                best_n = 18
                best_colors = colors
                break

    # 3. Local search for n=18
    if best_n < 18 and not time_check():
        n = 18
        edges_count = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(edges_count)]
        
        # Precompute edge indices for speed
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        
        for it in range(50000):
            if time_check(): break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            idx = edge_idx[(u, v)]
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c: continue
            
            colors[idx] = new_c
            if count_c6s(colors, n) == 0:
                best_n = 18
                best_colors = colors
                break
            colors[idx] = old_c

    if best_colors is None:
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}
