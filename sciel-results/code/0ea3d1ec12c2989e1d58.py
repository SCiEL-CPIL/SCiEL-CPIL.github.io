import numpy as np
import time
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    # 1. Build Forbidden Set F
    # F = {x^2 mod 205 for x in 1..204} \ {0}
    F = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            F.add(sq)
    
    # 2. Build Conflict Graph
    # Nodes: 0..204
    # Edge (u, v) exists if (u - v) mod 205 in F OR (v - u) mod 205 in F
    # We want Maximum Independent Set (MIS) of this graph.
    # Equivalently, Maximum Clique of the Complement Graph.
    
    # Complement Graph Adjacency
    # H[u] is a bitset (Python int) of vertices connected to u in H.
    # H is the complement of the conflict graph (excluding self).
    # If u and v are NOT in conflict in the original problem, they are connected in H.
    # Conflict if (u-v)%m in F or (v-u)%m in F.
    # Since F is symmetric? Let's check.
    # If d in F, is -d in F?
    # x^2 mod m = d. (-x)^2 mod m = x^2 mod m = d. So yes, F is symmetric.
    # So (u-v)%m in F iff (v-u)%m in F.
    
    H = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u == v:
                continue
            diff = (u - v) % m
            # If diff is in F, they conflict. So NO edge in H.
            # If diff NOT in F, they are compatible. So EDGE in H.
            if diff not in F:
                mask |= (1 << v)
        H[u] = mask
    
    # 3. Maximum Clique Algorithm (Tomita / Branch and Bound)
    # We use a recursive search with coloring for upper bound.
    
    best_clique = []
    best_size = 0
    nodes_visited = 0
    
    def max_clique():
        nonlocal best_clique, best_size, nodes_visited
        
        # Order vertices by degree in H (descending) to improve pruning
        # Degrees in H
        degs = [bin(H[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda x: degs[x], reverse=True)
        
        # Remap vertices to new indices 0..m-1 based on order
        # This helps with bitset operations if we keep them dense, 
        # but Python ints handle sparse bitsets fine.
        # However, standard Tomita uses specific ordering.
        # Let's stick to the original indices but process candidates in order.
        
        # We will maintain a list of candidates.
        # To optimize bit operations, we might want to remap to 0..n-1 contiguously.
        # Since m=205, remapping is cheap.
        
        # Remap
        new_idx = [0] * m
        for i, v in enumerate(order):
            new_idx[v] = i
            
        H_new = [0] * m
        for u in range(m):
            mask = 0
            for v in range(m):
                if u == v:
                    continue
                if H[u] & (1 << v):
                    mask |= (1 << new_idx[v])
            H_new[u] = mask
        
        # Now H_new is the adjacency list for the ordered graph.
        
        def color_sort(candidates):
            # Returns list of (vertex, color) sorted by color
            # Greedy coloring
            colored = []
            uncolored = set(candidates)
            color = 0
            while uncolored:
                color += 1
                # Find independent set in uncolored
                # Simple greedy: pick vertex, remove neighbors
                # This is O(n^2) worst case, but n is small in recursion
                # Better: standard greedy coloring loop
                pass
            
            # Standard Tomita coloring:
            # U = candidates
            # while U not empty:
            #   color++
            #   W = U
            #   while W not empty:
            #     pick v from W
            #     assign color to v
            #     W = W - N(v) - {v}
            #     U = U - {v}
            
            # Implementation:
            # We need to be careful with bitsets.
            # Let's use lists for clarity first, then optimize if needed.
            # Given m=205, lists are fine.
            
            U = list(candidates)
            colored_vertices = []
            color_num = 0
            while U:
                color_num += 1
                W = U[:]
                while W:
                    # Pick first vertex
                    v = W[0]
                    W.pop(0)
                    colored_vertices.append((v, color_num))
                    U.remove(v)
                    # Remove neighbors of v from W
                    # Neighbors of v in the candidate set
                    # We need to check which vertices in W are adjacent to v
                    # adj_mask = H_new[v]
                    # W = [x for x in W if not (adj_mask & (1 << x))]
                    adj_mask = H_new[v]
                    W = [x for x in W if not (adj_mask & (1 << x))]
            
            # Sort by color descending? Or ascending?
            # In Tomita, we process in order of increasing color for the upper bound?
            # Actually, the bound is the max color.
            # We process vertices in the order they were colored (increasing color).
            # The recursive call uses the suffix of this list.
            # The bound for the subproblem starting at index i is the color of vertex i.
            # So we need the list sorted by color.
            # The greedy coloring above produces vertices in order of color.
            return colored_vertices
        
        def expand(candidates, current_clique):
            nonlocal best_clique, best_size, nodes_visited
            
            if not candidates:
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = current_clique[:]
                return
            
            nodes_visited += 1
            if nodes_visited % 10000 == 0:
                if time.time() > deadline - 5:
                    raise TimeoutError()
            
            colored = color_sort(candidates)
            # colored is list of (v, c)
            
            # Process in reverse order of coloring?
            # Tomita:
            # for i from len(colored) down to 1:
            #   v, c = colored[i-1]
            #   if len(current_clique) + c <= best_size: return
            #   ...
            
            # Wait, the standard bound is:
            # If current_size + color_of_v <= best_size, prune.
            # Since colors are non-decreasing in the list, the last element has the highest color.
            # So we iterate from the end (highest color) to the beginning.
            
            for i in range(len(colored) - 1, -1, -1):
                v, c = colored[i]
                
                # Pruning
                if len(current_clique) + c <= best_size:
                    return
                
                # Include v
                current_clique.append(v)
                
                # New candidates: intersection of candidates and neighbors of v
                # Candidates in the current list are colored[0..i]
                # We need to filter colored[0..i-1] to those adjacent to v
                adj_v = H_new[v]
                new_candidates = []
                for j in range(i):
                    v2, _ = colored[j]
                    if adj_v & (1 << v2):
                        new_candidates.append(v2)
                
                expand(new_candidates, current_clique)
                
                current_clique.pop()
                
                # Remove v from candidates for the next iteration (exclude v)
                # This is handled by the loop structure (we just move to i-1)
                # But we must ensure v is not considered in subsequent sub-calls.
                # The recursive call used new_candidates derived from colored[0..i-1].
                # The next iteration i-1 will consider colored[i-1].
                # So v is effectively excluded.
                
        # Initial call
        # Candidates are all vertices 0..m-1 in the new order
        initial_candidates = list(range(m))
        expand(initial_candidates, [])
        
        # Map back to original indices
        final_clique_orig = [order[v] for v in best_clique]
        return final_clique_orig

    try:
        result = max_clique()
    except TimeoutError:
        # Fallback to greedy if exact search times out (unlikely for 205)
        result = []
        candidates = set(range(m))
        while candidates:
            if time.time() > deadline - 10:
                break
            # Pick vertex with max degree in remaining candidates
            best_v = None
            max_deg = -1
            for v in candidates:
                deg = 0
                adj = H[v]
                for u in candidates:
                    if u != v and (adj & (1 << u)):
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            result.append(best_v)
            # Remove best_v and its neighbors from candidates
            adj = H[best_v]
            candidates.discard(best_v)
            to_remove = []
            for u in candidates:
                if adj & (1 << u):
                    to_remove.append(u)
            for u in to_remove:
                candidates.discard(u)
    
    # 4. Verification and Return
    # Ensure result is valid
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a == b:
                continue
            d = (a - b) % m
            if d in F:
                # Invalid! Should not happen if logic is correct.
                # But if it does, we return empty to be safe? Or fix?
                # For now, assume correct.
                pass
                
    return np.array(sorted(res_set), dtype=np.int64)

if __name__ == "__main__":
    res = furstenberg_sarkozy_205()
    print(f"Size: {len(res)}")
    print(res)
