"""Rewritten program: local search with efficient delta evaluation for C4-free 5-coloring."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit

    # We'll try to find valid colorings for increasing n
    # Start from a known good construction and try to extend
    # The Lazebnik-Woldar construction gives 26 vertices

    best_n = 11
    best_colors = _cyclic_11()

    # Try local search to find valid colorings for larger n
    for target_n in range(12, 29):
        if time.time() > deadline - 5:
            break
        result = _local_search(target_n, time_budget=deadline - time.time() - 2)
        if result is not None:
            best_n = target_n
            best_colors = result
        else:
            # If we can't find for this n, try next (maybe random init needed)
            continue

    return {'n': best_n, 'colors': best_colors}


def _cyclic_11():
    n = 11
    return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]


def _edge_index(n, i, j):
    """Map pair (i,j) with i<j to index in lexicographic edge list."""
    # Number of edges before vertex i: i*(i-1)/2
    # Within vertex i, edges to i+1, i+2, ..., j-1 come before (i,j)
    # So offset = (j - i - 1)
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)


def _build_edge_map(n):
    """Build mapping from (i,j) to edge index for fast lookup."""
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    return edge_idx


def _count_c4_violations(colors, n):
    """Count total C4 violations across all colors."""
    # For each color, count pairs of vertices that share 2+ neighbors in that color
    # A C4 in color c: vertices a,b with |N_c(a) ∩ N_c(b)| >= 2
    # Equivalently: for each color c, for each pair of vertices (u,v), 
    # count common neighbors in color c. If >= 2, that's a violation.
    
    # More efficient: for each color c, build adjacency, then for each pair (a,b),
    # count common neighbors. But that's O(n^2 * n) per color.
    
    # Alternative: for each color c, for each vertex a, for each pair of neighbors (b,d) of a in color c,
    # increment count[b,d]. If count[b,d] >= 2, violation.
    
    violations = 0
    for c in range(5):
        # Build neighbor lists for color c
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = _edge_index(n, i, j)
                if colors[idx] == c:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        # For each pair of vertices, count common neighbors in color c
        # Use: for each vertex a, for each pair (b,d) in neighbors[a], increment pair_count[b,d]
        pair_count = [[0] * n for _ in range(n)]
        for a in range(n):
            nb = neighbors[a]
            ln = len(nb)
            for x in range(ln):
                b = nb[x]
                for y in range(x + 1, ln):
                    d = nb[y]
                    if b < d:
                        pair_count[b][d] += 1
                    else:
                        pair_count[d][b] += 1
        
        for i in range(n):
            for j in range(i + 1, n):
                if pair_count[i][j] >= 2:
                    violations += 1
    
    return violations


def _local_search(n, time_budget=10.0):
    """Local search to find a valid 5-coloring of K_n with no C4 in any color class."""
    if n < 2:
        return None
    
    m = n * (n - 1) // 2
    start = time.time()
    
    # Precompute edge indices
    edge_to_pair = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_pair.append((i, j))
    
    # For each vertex, list of edges incident to it
    vertex_edges = [[] for _ in range(n)]
    for idx, (i, j) in enumerate(edge_to_pair):
        vertex_edges[i].append(idx)
        vertex_edges[j].append(idx)
    
    # For each vertex, for each other vertex, the edge between them (or -1 if same)
    # edge_between[i][j] = edge index for (i,j) if i != j
    edge_between = [[-1] * n for _ in range(n)]
    for idx, (i, j) in enumerate(edge_to_pair):
        edge_between[i][j] = idx
        edge_between[j][i] = idx
    
    # Random initial coloring
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Count violations per color
    # For each color c, for each pair of vertices (u,v), count common neighbors in color c
    # We track: common_count[c][u][v] = number of vertices w such that (u,w) and (v,w) are both color c
    
    # This is expensive to compute from scratch. Let's use a different approach.
    # Track for each color c and each pair (u,v): how many common neighbors they have in color c.
    
    # Actually, let's use a simpler violation count:
    # For each color c, a C4 exists iff there exist vertices a,b such that 
    # |N_c(a) ∩ N_c(b)| >= 2.
    
    # Let's track: for each color c, for each unordered pair {u,v} (u≠v), 
    # the number of common neighbors in color c.
    # common[c][u][v] where u < v
    
    # This is 5 * n * n / 2 entries. For n=26, that's 5*338 = 1690. Manageable.
    
    # Initialize common counts
    # common[c][u][v] = number of w != u,v such that edge(u,w) has color c AND edge(v,w) has color c
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def _recompute_common():
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cnt = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        euw = edge_between[u][w]
                        evw = edge_between[v][w]
                        if colors[euw] == c and colors[evw] == c:
                            cnt += 1
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
    
    def _count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if common[c][u][v] >= 2:
                        total += 1
        return total
    
    # Initial computation
    _recompute_common()
    current_viol = _count_violations()
    
    if current_viol == 0:
        return colors[:]
    
    # Local search: pick a random edge, try changing its color, accept if it reduces violations
    # Use simulated annealing-like approach with temperature
    
    import math
    temp = 1.0
    cooling = 0.999
    
    # For efficiency, when we change color of edge e from c1 to c2:
    # We need to update common counts for all pairs (u,v) where the change affects common neighbors.
    # Edge e connects vertices a,b. 
    # For color c1: removing this edge affects common counts for pairs (a, x) and (b, x) for all x,
    #   and also pair (a,b) itself (they lose one common neighbor... wait no).
    #   
    # Actually, edge (a,b) with color c: this contributes to common[c][a][b] via... 
    # No wait. common[c][u][v] counts w such that (u,w) and (v,w) are color c.
    # Edge (a,b) being color c means: for any w, if (a,w) is color c and (b,w) is color c, 
    # that's counted in common[c][a][b]. But the edge (a,b) itself doesn't directly appear in 
    # the common count formula except through its endpoints.
    #
    # When we change color of edge (a,b) from c1 to c2:
    # - For color c1: edge (a,b) is no longer c1. This affects:
    #   - common[c1][a][x] for all x ≠ a,b: if (b,x) was c1, then (a,x) and (b,x) both c1 contributed to common[c1][a][x]. Now (a,b) is not c1, but that doesn't change (a,x) or (b,x). Wait, I'm confusing myself.
    #
    # Let me reconsider. common[c][u][v] = |{w : w≠u,v, edge(u,w)=c, edge(v,w)=c}|
    # The edge (a,b) with color c1: 
    #   - It doesn't appear in common[c1][a][b] because w must be ≠ a,b.
    #   - It appears in common[c1][a][x] only if... no. common[c1][a][x] counts w where (a,w) and (x,w) are c1. The edge (a,b) is (a,b), so it's (a,w) with w=b. So if (x,b) is also c1, then b is a common neighbor of a and x in color c1, contributing to common[c1][a][x].
    #
    # So changing edge (a,b) from c1 to c2:
    #   For color c1: 
    #     - For each x ≠ a,b: if edge(x,b) is c1, then common[c1][a][x] decreases by 1 (b was a common neighbor, now it's not since (a,b) is no longer c1).
    #     - Similarly, if edge(x,a) is c1, then common[c1][b][x] decreases by 1.
    #   For color c2:
    #     - For each x ≠ a,b: if edge(x,b) is c2, then common[c2][a][x] increases by 1.
    #     - If edge(x,a) is c2, then common[c2][b][x] increases by 1.
    #
    # So the delta in violations can be computed by checking which pairs (u,v) have their common count cross the threshold of 2.
    
    best_colors = colors[:]
    best_viol = current_viol
    
    max_iter = 200000
    iter_count = 0
    
    while time.time() < start + time_budget and iter_count < max_iter:
        iter_count += 1
        
        # Pick random edge
        e = random.randint(0, m - 1)
        a, b = edge_to_pair[e]
        old_c = colors[e]
        
        # Try each new color
        best_new_c = None
        best_delta = 0  # We want to minimize violations, so delta = new_viol - old_viol, want negative
        
        for new_c in range(5):
            if new_c == old_c:
                continue
            
            # Compute delta in violation count
            delta = 0
            
            # For old color: removing edge (a,b) from old_c
            # For each x ≠ a,b:
            #   if edge(a,x) == old_c and edge(b,x) == old_c: 
            #     common[old_c][a][x] decreases by 1
            #     if common[old_c][a][x] was exactly 2, it becomes 1: violation removed, delta -= 1
            #     if common[old_c][a][x] was 1, it becomes 0: no change in violation status
            #     if common[old_c][a][x] was >= 3, it becomes >= 2: still violation, no change
            #   Similarly for common[old_c][b][x]
            
            # But wait, we also need to check: does removing this edge affect common[old_c][a][b]?
            # No, because common[old_c][a][b] counts w ≠ a,b, and edge (a,b) is not (a,w) or (b,w) for w≠a,b.
            
            for x in range(n):
                if x == a or x == b:
                    continue
                eax = edge_between[a][x]
                ebx = edge_between[b][x]
                
                # Check if removing (a,b) from old_c affects common[old_c][a][x]
                if colors[eax] == old_c and colors[ebx] == old_c:
                    # common[old_c][a][x] will decrease by 1
                    if common[old_c][a][x] == 2:
                        delta -= 1  # violation removed
            
                # Check if removing (a,b) from old_c affects common[old_c][b][x]
                if colors[eax] == old_c and colors[ebx] == old_c:
                    # common[old_c][b][x] will decrease by 1
                    if common[old_c][b][x] == 2:
                        delta -= 1
            
            # For new color: adding edge (a,b) to new_c
            for x in range(n):
                if x == a or x == b:
                    continue
                eax = edge_between[a][x]
                ebx = edge_between[b][x]
                
                if colors[eax] == new_c and colors[ebx] == new_c:
                    # common[new_c][a][x] will increase by 1
                    if common[new_c][a][x] == 1:
                        delta += 1  # new violation created
                    
                    # common[new_c][b][x] will increase by 1
                    if common[new_c][b][x] == 1:
                        delta += 1
            
            if delta < best_delta or (delta == best_delta and best_new_c is None):
                best_delta = delta
                best_new_c = new_c
        
        if best_new_c is None:
            continue
        
        # Accept with probability based on delta and temperature
        if best_delta <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-best_delta / max(temp, 0.01))
        
        if accept:
            # Apply the change
            new_c = best_new_c
            a, b = edge_to_pair[e]
            old_c = colors[e]
            
            # Update common counts
            for x in range(n):
                if x == a or x == b:
                    continue
                eax = edge_between[a][x]
                ebx = edge_between[b][x]
                
                if colors[eax] == old_c and colors[ebx] == old_c:
                    common[old_c][a][x] -= 1
                    common[old_c][x][a] -= 1
                    common[old_c][b][x] -= 1
                    common[old_c][x][b] -= 1
                
                if colors[eax] == new_c and colors[ebx] == new_c:
                    common[new_c][a][x] += 1
                    common[new_c][x][a] += 1
                    common[new_c][b][x] += 1
                    common[new_c][x][b] += 1
            
            colors[e] = new_c
            current_viol += best_delta
            
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
            
            if current_viol == 0:
                return colors[:]
            
            temp *= cooling
        
        # Check time
        if time.time() > start + time_budget:
            break
    
    if best_viol == 0:
        return best_colors
    return None


def _try_seeded(n, seed=None, time_budget=5.0):
    """Try local search with a specific seed."""
    if seed is not None:
        random.seed(seed)
    return _local_search(n, time_budget=time_budget)


if __name__ == "__main__":
    result = construct()
    print(f"n = {result['n']}")
