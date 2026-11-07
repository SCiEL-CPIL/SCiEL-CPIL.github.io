"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random
from itertools import combinations

def _get_cycle_lengths(adj, n):
    """Find all distinct simple cycle lengths in a graph."""
    cycle_lengths = set()
    
    def dfs(start, current, path, visited):
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 4:
                cycle_lengths.add(len(path))
            elif neighbor not in visited and neighbor > start:
                # Only visit vertices > start to avoid duplicate cycles
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        dfs(start, start, [start], {start})
    
    return cycle_lengths

def _validate_edges(edges, n=12):
    """Check if edges form a valid graph."""
    if len(edges) == 0:
        return False
    # Check no loops or duplicates
    edge_set = set()
    for e in edges:
        u, v = e[0], e[1]
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
    
    # Check min degree >= 3
    degree = [0] * n
    for e in edges:
        degree[e[0]] += 1
        degree[e[1]] += 1
    for d in degree:
        if d < 3:
            return False
    
    # Check triangle-free
    adj = [set() for _ in range(n)]
    for e in edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w in adj[u]:
                        return False
    
    # Check has a cycle
    # BFS/DFS to check if graph has a cycle
    visited = [False] * n
    has_cycle = False
    for i in range(n):
        if not visited[i]:
            stack = [(i, -1, 0)]
            visited[i] = True
            while stack:
                node, parent, depth = stack.pop()
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if visited[nb]:
                        has_cycle = True
                    else:
                        visited[nb] = True
                        stack.append((nb, node, depth + 1))
    if not has_cycle:
        return False
    
    return True

def _find_cycle_4_only_graph():
    """Search for a 12-vertex bipartite graph with min degree 3, girth 4,
    where all cycles have length exactly 4."""
    deadline = time.time() + 75  # Safety margin before 90s limit
    
    # Try to construct: bipartite with partitions {0..5} and {6..11}
    # We need min degree 3, so each vertex has degree >= 3
    # To avoid 6-cycles: no three vertices in one partition should form
    # a path of length 3 back to the start through the other partition
    
    # Strategy: Start with K_{6,6} and remove edges to break 6-cycles
    # while maintaining min degree >= 3
    
    # Alternative: try specific constructions
    # A known approach: use a "linear" structure where neighborhoods are designed
    # to prevent 6-cycles
    
    # Let's try: partition L = {0,1,2,3,4,5}, R = {6,7,8,9,10,11}
    # Design neighborhoods so that for any l1,l2 in L, |N(l1) ∩ N(l2)| <= 1
    # This prevents 4-cycles... wait, we WANT 4-cycles.
    # Actually |N(l1) ∩ N(l2)| >= 2 creates a 4-cycle.
    # For 6-cycles: we need to prevent paths l1-r1-l2-r2-l3-r3-l1 where
    # l1,l2,l3 in L and r1,r2,r3 in R, all distinct.
    
    # Let me try a different approach: use a random search with validation
    
    # First, try some hand-crafted constructions
    # Construction 1: Based on a 6-cycle of neighborhoods
    # L vertices: 0,1,2,3,4,5; R vertices: 6,7,8,9,10,11
    
    # Try: each L vertex connects to 3 R vertices in a pattern
    # that avoids 6-cycles
    
    # Let's try: N(0)={6,7,8}, N(1)={7,8,9}, N(2)={8,9,10}, 
    #           N(3)={9,10,11}, N(4)={10,11,6}, N(5)={11,6,7}
    # Check 4-cycles: 0-6-5-7-0? 0 connects to 6,7,8. 5 connects to 11,6,7.
    # Common neighbors of 0 and 5: {6,7} -> 4-cycle 0-6-5-7-0 exists. Good.
    # Check 6-cycles: 0-6-5-11-4-10-2-8-0? That's length 8.
    # 0-6-5-7-1-8-0? 0-6, 6-5, 5-7, 7-1, 1-8, 8-0. Yes! This is a 6-cycle.
    # So this construction has 6-cycles.
    
    # Let me try a more systematic approach with random search
    best_edges = None
    best_num_lengths = 999
    
    # Pre-compute all possible edge sets for bipartite graph
    # L = 0..5, R = 6..11
    all_possible_edges = [(i, j) for i in range(6) for j in range(6, 12)]
    
    # Random search: start with random sparse bipartite graphs with min degree >= 3
    random.seed(42)
    
    for attempt in range(500000):
        if time.time() > deadline:
            break
        
        # Generate random bipartite graph with min degree >= 3
        # Each L vertex needs degree >= 3, each R vertex needs degree >= 3
        # Total edges >= 18
        
        # Try: start with each L vertex having exactly 3 neighbors
        # and each R vertex having degree >= 3
        
        # Random assignment
        adj_l = [set() for _ in range(6)]
        adj_r = [set() for _ in range(6)]
        
        # Randomly assign 3 neighbors to each L vertex
        for i in range(6):
            neighbors = random.sample(range(6), 3)
            adj_l[i] = set(neighbors)
            for j in neighbors:
                adj_r[j].add(i)
        
        # Check R degrees
        valid = True
        for j in range(6):
            if len(adj_r[j]) < 3:
                valid = False
                break
        
        if not valid:
            continue
        
        # Build edge list
        edges = []
        for i in range(6):
            for j in adj_l[i]:
                edges.append([i, j + 6])
        
        # Check for 6-cycles (and longer cycles)
        # Build adjacency
        adj = [set() for _ in range(12)]
        for e in edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        
        # Find all cycle lengths
        cycle_lengths = set()
        
        # Check for 4-cycles
        for i in range(6):
            for j in range(i+1, 6):
                common = adj_l[i] & adj_l[j]
                if len(common) >= 2:
                    cycle_lengths.add(4)
        
        # Check for 6-cycles: i1-j1-i2-j2-i3-j3-i1
        # i1,i2,i3 in L (0-5), j1,j2,j3 in R (0-5)
        # j1 in N(i1)∩N(i3), j2 in N(i1)∩N(i2), j3 in N(i2)∩N(i3)
        # and j1,j2,j3 all distinct, i1,i2,i3 all distinct
        has_6cycle = False
        for i1 in range(6):
            for i2 in range(6):
                if i2 == i1:
                    continue
                for i3 in range(6):
                    if i3 == i1 or i3 == i2:
                        continue
                    common_13 = adj_l[i1] & adj_l[i3]
                    common_12 = adj_l[i1] & adj_l[i2]
                    common_23 = adj_l[i2] & adj_l[i3]
                    for j1 in common_13:
                        for j2 in common_12:
                            if j2 == j1:
                                continue
                            for j3 in common_23:
                                if j3 == j1 or j3 == j2:
                                    continue
                                # Check: i1-j1-i3-j3-i2-j2-i1
                                # j1 connects to i1 and i3 ✓
                                # j3 connects to i3 and i2 ✓  
                                # j2 connects to i2 and i1 ✓
                                # This forms a 6-cycle!
                                has_6cycle = True
                                break
                            if has_6cycle:
                                break
                        if has_6cycle:
                            break
                    if has_6cycle:
                        break
                if has_6cycle:
                    break
            if has_6cycle:
                break
        
        if has_6cycle:
            cycle_lengths.add(6)
        
        # Check for longer cycles (8, 10, 12)
        # For efficiency, only check if we haven't found a 6-cycle
        # and we want to know if there are other lengths
        if len(cycle_lengths) < 2:
            # Check for 8-cycles and longer
            # This is expensive; do a BFS-based cycle detection
            # Actually, let's just check if there are any cycles of length > 4
            # by doing a proper cycle enumeration for small graphs
            
            # Simple check: if no 6-cycle, check for 8+ cycles
            # Use DFS to find cycles
            found_longer = False
            visited_cycles = set()
            
            def find_cycles_dfs(start, current, path, visited_set):
                nonlocal found_longer
                for nb in adj[current]:
                    if nb == start and len(path) > 4:
                        cycle_len = len(path)
                        if cycle_len not in cycle_lengths:
                            cycle_lengths.add(cycle_len)
                        if cycle_len > 4:
                            found_longer = True
                    elif nb not in visited_set and nb > start:
                        visited_set.add(nb)
                        path.append(nb)
                        find_cycles_dfs(start, nb, path, visited_set)
                        path.pop()
                        visited_set.remove(nb)
            
            for s in range(12):
                find_cycles_dfs(s, s, [s], {s})
        
        num_lengths = len(cycle_lengths)
        if num_lengths < best_num_lengths:
            best_num_lengths = num_lengths
            best_edges = [list(e) for e in edges]
            if num_lengths == 1:
                # Verify it's exactly {4}
                if cycle_lengths == {4}:
                    return best_edges
    
    return best_edges

def construct():
    # Try to find a graph with all cycles of length 4
    result = _find_cycle_4_only_graph()
    
    if result is not None and _validate_edges(result):
        # Double-check the cycle lengths
        adj = [set() for _ in range(12)]
        for e in result:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        
        cycle_lengths = set()
        for start in range(12):
            def dfs(s, cur, path, vis):
                for nb in adj[cur]:
                    if nb == s and len(path) >= 4:
                        cycle_lengths.add(len(path))
                    elif nb not in vis and nb > s:
                        vis.add(nb)
                        path.append(nb)
                        dfs(s, nb, path, vis)
                        path.pop()
                        vis.remove(nb)
            dfs(start, start, [start], {start})
        
        if cycle_lengths == {4}:
            return result
    
    # Fallback: two disjoint K_{3,3} components (value = 2)
    edges = []
    part1_left = [0, 1, 2]
    part1_right = [3, 4, 5]
    part2_left = [6, 7, 8]
    part2_right = [9, 10, 11]
    
    for u in part1_left:
        for v in part1_right:
            edges.append([u, v])
    for u in part2_left:
        for v in part2_right:
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
