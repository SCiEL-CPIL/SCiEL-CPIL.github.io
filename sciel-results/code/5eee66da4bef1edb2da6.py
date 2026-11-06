import time
import random
from itertools import combinations, product

def construct():
    start = time.time()
    deadline = start + 118  # Safety margin before 120s limit
    
    def has_c6_bfs(adj, n):
        """Check if graph with adjacency matrix adj (list of lists of ints 0/1) on n vertices
        contains a 6-cycle. Uses BFS from each edge to find cycles of length 6.
        A 6-cycle exists iff there is a path of length 5 between two adjacent vertices
        (v,u) where the path doesn't use edge (v,u) directly, and all 6 vertices are distinct.
        Actually, a 6-cycle means there's a simple cycle of length 6.
        We can check: for each pair of vertices (a,b) that are NOT adjacent, check if there
        is a path of length 5 from a to b. If so, a-b-edge-completes-a-6-cycle? No.
        
        Better: A 6-cycle is a sequence v0,v1,v2,v3,v4,v5,v0 where all consecutive pairs
        are edges and all vertices are distinct.
        
        Efficient approach: For each ordered pair (s, t) with s != t and not adjacent,
        check if there's a path of exactly 5 edges from s to t. If yes, then s-t is not
        an edge but there's a 5-path, giving a 6-cycle.
        
        Wait, that's not quite right either. The 6-cycle is v0-v1-v2-v3-v4-v5-v0.
        So v0 and v5 are adjacent, v0 and v1 are adjacent, etc.
        
        Let me think differently. A 6-cycle contains 6 edges. Pick one edge (u,v). 
        Then the other 4 vertices must form a path of length 4 from some neighbor of u 
        (other than v) to some neighbor of v (other than u).
        
        Actually, the most straightforward exact check for small n:
        For each 6-subset of vertices, check if it contains a Hamiltonian cycle.
        C(20,6) = 38760, and checking each is fast.
        """
        # For n up to 20, C(n,6) is manageable
        # But we can be smarter: only check 6-subsets where all 6 vertices have degree >= 2 in the subgraph
        
        # Alternative: use the fact that a C6 in the graph means there exist vertices
        # a,b,c,d,e,f all distinct such that a-b-c-d-e-f-a are all edges.
        # Fix a and b as adjacent. Then we need c-d-e-f to be a path of length 3 from a neighbor of b
        # (c, c != a) to a neighbor of a (f, f != b), with all vertices distinct.
        
        # Let's use a different approach: 
        # For each edge (u, v), check if there's a path of length 4 from u to v 
        # using only vertices other than u and v in the internal positions... no wait.
        
        # Simplest correct approach for n <= 20:
        # Enumerate all 6-tuples (v0,...,v5) with v0 < v1 (to break symmetry partially)
        # Actually let's just do the 6-subset approach.
        
        for combo in combinations(range(n), 6):
            # Build subgraph on these 6 vertices
            # Check if there's a Hamiltonian cycle in this subgraph
            # 6 vertices: fix vertex 0, try permutations of remaining 5 = 120
            # But we can do better with backtracking
            
            # Map combo to local indices
            idx_map = {combo[i]: i for i in range(6)}
            
            # Local adjacency
            local_adj = [[0]*6 for _ in range(6)]
            for i in range(6):
                for j in range(i+1, 6):
                    u = combo[i]
                    v = combo[j]
                    if adj[u][v]:
                        local_adj[i][j] = 1
                        local_adj[j][i] = 1
            
            # Quick check: each vertex must have degree >= 2 in subgraph
            for i in range(6):
                deg = sum(local_adj[i])
                if deg < 2:
                    break
            else:
                # All have degree >= 2, check for Hamiltonian cycle
                # Fix vertex 0, try all permutations of 1-5
                for perm in product(range(1, 6), repeat=5):
                    if len(set(perm)) < 5:
                        continue
                    # Cycle: 0 - perm[0] - perm[1] - perm[2] - perm[3] - perm[4] - 0
                    if (local_adj[0][perm[0]] and 
                        local_adj[perm[0]][perm[1]] and 
                        local_adj[perm[1]][perm[2]] and 
                        local_adj[perm[2]][perm[3]] and 
                        local_adj[perm[3]][perm[4]] and 
                        local_adj[perm[4]][0]):
                        return True
        return False

    def count_c6(adj, n):
        """Count the number of 6-cycles in the graph."""
        count = 0
        for combo in combinations(range(n), 6):
            idx_map = {combo[i]: i for i in range(6)}
            local_adj = [[0]*6 for _ in range(6)]
            for i in range(6):
                for j in range(i+1, 6):
                    u = combo[i]
                    v = combo[j]
                    if adj[u][v]:
                        local_adj[i][j] = 1
                        local_adj[j][i] = 1
            
            valid = True
            for i in range(6):
                if sum(local_adj[i]) < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            for perm in product(range(1, 6), repeat=5):
                if len(set(perm)) < 5:
                    continue
                if (local_adj[0][perm[0]] and 
                    local_adj[perm[0]][perm[1]] and 
                    local_adj[perm[1]][perm[2]] and 
                    local_adj[perm[2]][perm[3]] and 
                    local_adj[perm[3]][perm[4]] and 
                    local_adj[perm[4]][0]):
                    count += 1
        return count

    def verify_coloring(n, colors):
        """Verify that no color class contains a C6."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        # Build adjacency for each color
        for c in range(4):
            adj = [[0]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    idx += 1
            
            if has_c6_bfs(adj, n):
                return False
        return True

    def cyclic_coloring(n, dist_assignment):
        """
        Generate a cyclic (circulant) coloring of K_n.
        dist_assignment: list of length (n-1)//2 where dist_assignment[d-1] is the color for distance d.
        For distance d, the edge (i, i+d) mod n gets color dist_assignment[d-1],
        and edge (i, i-d) mod n also gets that color (since distance is symmetric).
        For even n, distance n/2 is special: edge (i, i+n/2) mod n.
        """
        colors = []
        half = (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d_min = min(d, n - d)
                # d_min is in range 1..half (or 1..n/2 for even n)
                if n % 2 == 0 and d_min == n // 2:
                    # Special case for even n
                    c = dist_assignment[half - 1] if half > 0 else 0
                else:
                    c = dist_assignment[d_min - 1]
                colors.append(c)
        return colors

    def random_local_search(n, max_time, seed=None):
        """
        Local search: start with random coloring, try to flip colors to eliminate C6s.
        Objective: minimize total number of C6s across all color classes.
        """
        if seed is not None:
            random.seed(seed)
        
        num_edges = n * (n - 1) // 2
        
        # Initialize with random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        def build_adjs(colors):
            adjs = []
            for c in range(4):
                adj = [[0]*n for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if colors[idx] == c:
                            adj[i][j] = 1
                            adj[j][i] = 1
                        idx += 1
                adjs.append(adj)
            return adjs
        
        def total_c6s(adjs):
            total = 0
            for adj in adjs:
                if has_c6_bfs(adj, n):
                    # Count them
                    for combo in combinations(range(n), 6):
                        idx_map = {combo[i]: i for i in range(6)}
                        local_adj = [[0]*6 for _ in range(6)]
                        for i in range(6):
                            for j in range(i+1, 6):
                                u = combo[i]
                                v = combo[j]
                                if adj[u][v]:
                                    local_adj[i][j] = 1
                                    local_adj[j][i] = 1
                        
                        valid = True
                        for i in range(6):
                            if sum(local_adj[i]) < 2:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        for perm in product(range(1, 6), repeat=5):
                            if len(set(perm)) < 5:
                                continue
                            if (local_adj[0][perm[0]] and 
                                local_adj[perm[0]][perm[1]] and 
                                local_adj[perm[1]][perm[2]] and 
                                local_adj[perm[2]][perm[3]] and 
                                local_adj[perm[3]][perm[4]] and 
                                local_adj[perm[4]][0]):
                                total += 1
            return total
        
        adjs = build_adjs(colors)
        current_c6s = total_c6s(adjs)
        
        if current_c6s == 0:
            return colors
        
        best_colors = colors[:]
        best_c6s = current_c6s
        
        t = time.time()
        while time.time() - t < max_time:
            # Try flipping a random edge's color
            if random.random() < 0.3:
                # Try all edges in a random color class
                c = random.randint(0, 3)
                edges_in_c = [i for i in range(num_edges) if colors[i] == c]
                if not edges_in_c:
                    continue
                edge_idx = random.choice(edges_in_c)
                new_c = random.randint(0, 3)
                if new_c == c:
                    continue
            else:
                edge_idx = random.randint(0, num_edges - 1)
                new_c = random.randint(0, 3)
                if new_c == colors[edge_idx]:
                    continue
            
            old_c = colors[edge_idx]
            colors[edge_idx] = new_c
            
            # Rebuild adjs
            adjs = build_adjs(colors)
            new_c6s = total_c6s(adjs)
            
            if new_c6s < best_c6s:
                best_c6s = new_c6s
                best_colors = colors[:]
                if best_c6s == 0:
                    return best_colors
            else:
                # Revert
                colors[edge_idx] = old_c
        
        return best_colors

    # Strategy: Try to find valid colorings for n = 18, 17, 16, ...
    # 1. First try known cyclic constructions
    # 2. Then try local search
    
    # For n=17: distances are 1..8
    # Known good cyclic colorings use specific distance assignments
    
    # Let's try systematic search of cyclic colorings first
    
    def try_cyclic(n, max_time):
        """Try all cyclic colorings for given n."""
        half = (n - 1) // 2
        t = time.time()
        
        # For each distance 1..half, assign a color 0..3
        # Try all 4^half assignments, but prune with time limit
        # For n=17, half=8, 4^8=65536 - feasible if fast enough
        
        for assignment in product(range(4), repeat=half):
            if time.time() - t > max_time:
                break
            colors = cyclic_coloring(n, list(assignment))
            if verify_coloring(n, colors):
                return colors
        return None

    # Try n=18 first (would be a new result)
    for n_target in [18, 17, 16]:
        if time.time() > deadline - 30:
            break
        
        half = (n_target - 1) // 2
        
        # Try cyclic colorings
        # For n=18, half=9, 4^9=262144 - might be too slow for verification
        # For n=17, half=8, 4^8=65536
        
        if n_target <= 17:
            max_cyclic_time = min(40, deadline - time.time() - 20)
            result = try_cyclic(n_target, max_cyclic_time)
            if result is not None:
                return {'n': n_target, 'colors': result}
        
        # Try local search
        max_ls_time = min(25, deadline - time.time() - 15)
        if max_ls_time > 5:
            best = random_local_search(n_target, max_ls_time, seed=42)
            if best is not None and verify_coloring(n_target, best):
                return {'n': n_target, 'colors': best}
    
    # Fallback: return a known valid small coloring
    # n=9 cyclic coloring
    n = 9
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            d_min = min(d, n - d)
            # Use a simple assignment: color = (d_min - 1) % 4
            colors.append((d_min - 1) % 4)
    return {'n': n, 'colors': colors}
