import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Returns False if it is NOT 2-colorable.
    """
    if not edges:
        return True
        
    m = len(edges)
    # Map vertex to list of edge indices containing it
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    colors = [-1] * n
    # edge_counts[i] = [count_0, count_1, count_unassigned]
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                edge_counts[e_idx][2] -= 1
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if cU == 1:
                    # Find the unassigned vertex in this edge
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        # Must be 1
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        # Must be 0
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        # Heuristic: pick vertex with highest degree in remaining constraints
        # or most constrained
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                # Prioritize vertices in edges that are close to being monochromatic
                if cU == 1:
                    if c0 == 4 or c1 == 4:
                        score += 10
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True # All assigned
            
        v = best_v
        # Try 0 then 1
        for c in [0, 1]:
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    # Initial propagation (none needed as no vars assigned)
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try n=9, 10, 11
    # n=9 is the smallest possible. Known m(5) is related to n=9 or 10 constructions.
    # The best known is 51 edges.
    
    for n in [9, 10, 11]:
        if time.time() > deadline - 20.0:
            break
            
        all_possible_edges = list(combinations(range(n), 5))
        num_edges_total = len(all_possible_edges)
        
        # Precompute edge masks for fast subset checking
        # We represent a coloring by the set S of vertices with color 0.
        # Vertex 0 is always in S.
        # Other vertices 1..n-1 can be in S or not.
        
        # Generate all colorings (subsets of 1..n-1)
        # There are 2^(n-1) such subsets.
        colorings = []
        # For speed, we can iterate over masks
        for mask in range(1 << (n - 1)):
            # mask represents vertices 1..n-1
            # S = {0} U { i+1 | bit i of mask is 1 }
            colorings.append(mask)
            
        num_colorings = len(colorings)
        
        # For each edge, determine which colorings it covers.
        # Edge e covers coloring S if e is monochromatic in S.
        # e is monochromatic if e subset of S OR e subset of V\S.
        
        # Precompute for each edge the list of coloring masks it covers.
        # This is the "Set" in Set Cover.
        edge_covers = [[] for _ in range(num_edges_total)]
        
        # To speed this up, we can iterate colorings and mark edges?
        # Or iterate edges and check colorings?
        # 2^(n-1) * C(n,5) might be large.
        # n=9: 2^8=256, C(9,5)=126. 256*126 ~ 32k checks. Fast.
        # n=10: 2^9=512, C(10,5)=252. 512*252 ~ 129k checks. Fast.
        # n=11: 2^10=1024, C(11,5)=462. 1024*462 ~ 473k checks. Fast.
        
        for e_idx, e in enumerate(all_possible_edges):
            e_set = set(e)
            for mask in colorings:
                # Reconstruct S from mask
                # S contains 0.
                # S contains i+1 if (mask >> i) & 1
                # Check if e is subset of S or e is subset of V\S
                
                # Check e subset of S:
                # 0 is in e? If yes, 0 is in S.
                # Other vertices in e must be in S.
                in_S = True
                in_not_S = True
                
                for v in e:
                    if v == 0:
                        # 0 is always in S
                        if not in_not_S: # Contradiction? No, just checking conditions
                            pass
                        # If v=0, it's in S. So for "e subset of not S", it must be false.
                        in_not_S = False
                    else:
                        bit_idx = v - 1
                        is_in_S = (mask >> bit_idx) & 1
                        if not is_in_S:
                            in_S = False
                        else:
                            in_not_S = False
                            
                if in_S or in_not_S:
                    edge_covers[e_idx].append(mask)
                    
        # Greedy Set Cover
        # uncovered: set of masks
        uncovered = set(colorings)
        chosen_edges = []
        
        # Sort edges by initial coverage? Or just greedy pick max coverage each step.
        
        while uncovered and time.time() < deadline - 10.0:
            best_e = -1
            best_count = -1
            
            for e_idx in range(num_edges_total):
                if e_idx in chosen_edges:
                    continue
                count = 0
                # Count how many uncovered colorings this edge covers
                for mask in edge_covers[e_idx]:
                    if mask in uncovered:
                        count += 1
                        if count > best_count:
                            # Early break? No, need exact count to compare with others?
                            # Actually, we can break if count is already > best_count?
                            # No, we need to find the MAX.
                            pass
                            
                if count > best_count:
                    best_count = count
                    best_e = e_idx
                    
            if best_e == -1 or best_count == 0:
                break # Cannot cover all?
                
            chosen_edges.append(best_e)
            
            # Remove covered colorings
            for mask in edge_covers[best_e]:
                uncovered.discard(mask)
                
        if not uncovered:
            current_edges = [list(all_possible_edges[e]) for e in chosen_edges]
            
            # Verify with solver
            if not is_2_colorable(n, current_edges):
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': current_edges}
                    
                # Local Search: Try to remove edges
                improved = True
                while improved and time.time() < deadline - 5.0:
                    improved = False
                    # Shuffle to try different removal orders? 
                    # Or just iterate.
                    for i in range(len(current_edges)):
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        if not is_2_colorable(n, test_edges):
                            current_edges = test_edges
                            if len(current_edges) < best_edge_count:
                                best_edge_count = len(current_edges)
                                best_result = {'n': n, 'edges': current_edges}
                            improved = True
                            break # Restart loop as list changed
        
    if best_result:
        return best_result
        
    # Fallback: Return a known valid construction if found nothing (should not happen)
    # Or just return K9^5
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
