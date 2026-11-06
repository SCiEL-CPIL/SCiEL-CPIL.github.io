import time
import random
import sys

def _precompute_k32():
    """Precompute the known valid 32-vertex 4-coloring (seed construction)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

_K32_COLORS = _precompute_k32()

def _validate(colors, n):
    if len(colors) != n * (n - 1) // 2:
        return False
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3:
                return False
            adj[c].add((i, j))
            idx += 1
    for c in range(4):
        adj_list = [[] for _ in range(n)]
        for (u, v) in adj[c]:
            adj_list[u].append(v)
            adj_list[v].append(u)
        for u in range(n):
            nbrs = adj_list[u]
            nbr_set = set(nbrs)
            for w in nbrs:
                for x in nbrs:
                    if w < x and x in nbr_set:
                        if (w, x) in adj[c]:
                            return False
    return True

def _build_adj_from_colors(colors, n):
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _extend_to_target(base_n, target_n, colors_base, deadline):
    n = target_n
    # adj[c][v] bitset
    adj = [[0] * n for _ in range(4)]
    # Initialize with base coloring
    idx = 0
    for i in range(base_n):
        for j in range(i + 1, base_n):
            c = colors_base[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    # We need to extend to n vertices.
    # Vertices are 0..base_n-1 already set.
    # We add vertices base_n, base_n+1, ..., n-1.
    # For each new vertex v, we need to assign colors to edges (u, v) for all u < v.
    # This is a subproblem of size (v - base_n) edges? No, it's (v - base_n) edges connecting to existing vertices?
    # Wait, if we add vertex v, edges are (u, v) for u in 0..v-1.
    # Since v > base_n, u can be any of 0..v-1.
    # The edges (u, w) for u, w < v are already set if u, w < base_n.
    # But if u < base_n < w < v, then edge (u, w) was set when w was added.
    # So we need to keep track of all edges.
    
    # It is easier to just maintain the full adj array for 0..n-1.
    # Initially, only 0..base_n-1 are connected.
    # We add vertices one by one.
    
    # To make it faster, we can use backtracking for each new vertex.
    # When adding vertex v, we assign colors to (0, v), (1, v), ..., (v-1, v).
    # We need to ensure no monochromatic triangle is formed.
    # A triangle (u, w, v) is monochromatic color c if:
    # 1. (u, w) is color c (already set if u, w < v)
    # 2. (u, v) is color c
    # 3. (w, v) is color c
    
    # We can backtrack on the colors for the new vertex.
    # The number of edges for new vertex v is v.
    # For v=50, that's 50 edges. 4^50 is too large.
    # But constraints are strong.
    
    # Let's implement a recursive function to assign colors to edges (u, v) for u in 0..v-1.
    # We process u in increasing order? Or random order?
    # Increasing order is standard.
    
    def solve_vertex(v, adj, deadline):
        # We need to assign colors to (u, v) for u=0..v-1.
        # Let's use a list `assign` of length v, where assign[u] is the color of (u, v).
        # Or just return a list of colors.
        
        # Optimization: Precompute for each u, the set of colors that are "blocked" by existing edges?
        # No, the conflict depends on other edges (w, v) too.
        
        # Backtracking search over u=0..v-1.
        # State: colors assigned to (0, v)...(k-1, v).
        # We maintain a set of "potential triangles" to prune?
        # Or just check locally.
        
        # Let's try a simple DFS.
        # It might be slow if v is large and base_n is small.
        # But base_n=32, v=33 is 33 edges.
        
        # We can optimize by checking conflicts as we assign.
        # When assigning color c to (u, v), we check if any w < u such that (w, v) is c and (u, w) is c.
        # This is O(u) per assignment. Total O(v^2).
        # With v=50, that's 2500 ops per step.
        # Backtracking depth 50.
        
        # Let's use a bitset trick.
        # For a fixed color c, let S_c be the set of u < v such that (u, v) is assigned color c.
        # When we assign (u, v) = c, we must ensure that no w in S_c has (u, w) = c.
        # i.e., adj[c][u] & (bitset of S_c) == 0.
        
        # We maintain bitsets S[0..3] for the new vertex v.
        # S[c] is a bitset of u's assigned color c.
        
        S = [0] * 4
        colors = [-1] * v
        
        def dfs(k):
            if time.time() > deadline:
                return False
            if k == v:
                return True
            
            # Try colors for edge (k, v)
            # Heuristic: try colors that are least frequent in S?
            # Or just 0,1,2,3.
            for c in [0, 1, 2, 3]:
                # Check conflict:
                # We need adj[c][k] & S[c] == 0
                if (adj[c][k] & S[c]) == 0:
                    # Assign
                    colors[k] = c
                    S[c] |= (1 << k)
                    
                    # Update adj for future vertices?
                    # No, adj is for the graph.
                    # We should update adj[c][k] to include v?
                    # Yes, for future vertices.
                    adj[c][k] |= (1 << v)
                    adj[c][v] |= (1 << k)
                    
                    if dfs(k + 1):
                        return True
                    
                    # Undo
                    colors[k] = -1
                    S[c] &= ~(1 << k)
                    adj[c][k] &= ~(1 << v)
                    adj[c][v] &= ~(1 << k)
            
            return False
        
        return dfs(0)

    # Start extension
    for v in range(base_n, n):
        # Copy adj to avoid modifying the original in case of failure?
        # If we fail, we return None.
        # But if we succeed, we want to keep the state.
        # Since we modify adj in place, and we only return if success, it's fine.
        # However, if we want to restart or try different orders, we need to be careful.
        # For now, single pass.
        
        # Randomize vertex order?
        # The task requires vertices 0..n-1.
        # The edges are (i, j) with i<j.
        # If we permute vertices, the edge order in the output changes.
        # The output must be in lexicographic order of 0..n-1.
        # So we can't arbitrarily permute vertices in the final output.
        # But we can permute them during construction and un-permute at the end?
        # Yes!
        # 1. Start with base coloring on 0..31.
        # 2. Create a permutation of 0..49.
        # 3. Construct graph on 0..49 using the permutation?
        # No, the vertices are labeled 0..49.
        # If we permute, we are effectively relabeling.
        # The constraint is on the graph structure, not labels.
        # So we can build the graph on any set of 50 vertices, then relabel to 0..49.
        
        # Let's stick to adding 32, 33, ..., 49 to the existing 0..31.
        # This fixes the labels.
        
        if not solve_vertex(v, adj, deadline):
            return None
    
    # Reconstruct colors array
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Find color
            # adj[c][i] has bit j set?
            c = 0
            found = False
            for c in range(4):
                if (adj[c][i] >> j) & 1:
                    colors.append(c)
                    found = True
                    break
            if not found:
                return None # Should not happen
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Try to extend 32 to 50
    # We can try multiple random restarts?
    # The solve_vertex is deterministic if we don't shuffle colors.
    # Let's shuffle the color order in solve_vertex to allow restarts.
    # But currently solve_vertex is hardcoded to 0,1,2,3.
    
    # Let's modify solve_vertex to accept a random seed or use global random.
    # Actually, let's just wrap the extension in a loop with randomization.
    # How to randomize?
    # We can randomize the order of colors tried in dfs.
    
    # Let's create a version of _extend_to_target that allows randomization.
    
    def _extend_random(base_n, target_n, colors_base, deadline):
        n = target_n
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(base_n):
            for j in range(i + 1, base_n):
                c = colors_base[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        def solve_vertex(v, adj, deadline):
            S = [0] * 4
            colors = [-1] * v
            
            def dfs(k):
                if time.time() > deadline:
                    return False
                if k == v:
                    return True
                
                # Randomize color order
                colors_to_try = [0, 1, 2, 3]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    if (adj[c][k] & S[c]) == 0:
                        colors[k] = c
                        S[c] |= (1 << k)
                        adj[c][k] |= (1 << v)
                        adj[c][v] |= (1 << k)
                        
                        if dfs(k + 1):
                            return True
                        
                        colors[k] = -1
                        S[c] &= ~(1 << k)
                        adj[c][k] &= ~(1 << v)
                        adj[c][v] &= ~(1 << k)
                
                return False
            
            return dfs(0)
        
        for v in range(base_n, n):
            if not solve_vertex(v, adj, deadline):
                return None
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                c = -1
                for cc in range(4):
                    if (adj[cc][i] >> j) & 1:
                        c = cc
                        break
                if c == -1:
                    return None
                colors.append(c)
        return colors

    # Try extending to 50
    best_n = 32
    best_colors = _K32_COLORS
    
    # Attempt 1: Direct extension with a few restarts
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        res = _extend_random(32, 50, _K32_COLORS, deadline - 2)
        if res is not None:
            # Validate
            if _validate(res, 50):
                return {'n': 50, 'colors': res}
    
    # If 50 fails, try 49, 48, ... down to 33?
    # This increases the chance of finding something better than 32.
    for target in range(49, 32, -1):
        if time.time() > deadline - 5:
            break
        # Try a few restarts for this target
        found = False
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            res = _extend_random(32, target, _K32_COLORS, deadline - 1)
            if res is not None:
                if _validate(res, target):
                    best_n = target
                    best_colors = res
                    found = True
                    break
        if found:
            break
    
    return {'n': best_n, 'colors': best_colors}
