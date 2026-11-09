import math
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # Perfect squares up to 2*N
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Diagonal constraint: 2a must not be a perfect square
    # Remove vertices where 2a is a perfect square
    valid_vertices = []
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_vertices.append(a)
    
    n = len(valid_vertices)
    # Map vertex values to indices
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # Build adjacency: for each pair (i,j) with i<j, check if valid_vertices[i] + valid_vertices[j] is a square
    # Use bitmasks for efficiency
    adj = [0] * n  # adj[i] is a bitmask of neighbors of i
    
    for i in range(n):
        vi = valid_vertices[i]
        for sq in squares:
            if sq <= 2:
                continue
            vj = sq - vi
            if vj != vi and vj in val_to_idx:
                j = val_to_idx[vj]
                if j > i:  # only add once
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    
    # Greedy initial solution for lower bound
    def greedy_mis():
        remaining = set(range(n))
        selected = []
        # Order by degree (fewest conflicts first)
        order = sorted(remaining, key=lambda x: bin(adj[x]).count('1'))
        for v in order:
            if v not in remaining:
                continue
            # Check if v conflicts with any already selected
            can_add = True
            for s in selected:
                if adj[s] & (1 << v):
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                remaining.discard(v)
        return selected
    
    initial_sel = greedy_mis()
    best_size = len(initial_sel)
    best_set = set(initial_sel)
    
    # Greedy coloring upper bound for branch and bound
    # We use a simple greedy coloring on the remaining candidate set
    # The upper bound for MIS is the number of colors needed (chromatic number upper bound)
    
    def greedy_color_bound(candidates):
        """Upper bound on MIS size via greedy coloring.
        candidates is a list of vertex indices."""
        if not candidates:
            return 0
        # Greedy coloring: assign colors greedily
        color = {}
        num_colors = 0
        # Order by degree within candidates
        cand_set = set(candidates)
        for v in sorted(candidates, key=lambda x: bin(adj[x] & sum((1<<c for c in candidates), 0)).count('1')):
            used_colors = set()
            for u in candidates:
                if u < v and (adj[v] & (1 << u)):
                    if u in color:
                        used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        return num_colors
    
    # Better: use the classic Tomita-style coloring for upper bound
    def color_upper_bound(candidates):
        """Compute an upper bound on the size of MIS in candidates using greedy coloring.
        This is the standard technique: color the subgraph, and the number of colors
        gives an upper bound on the independence number (since each independent set
        can have at most one vertex of each color... wait, no.
        
        Actually for MIS: if we color the graph, an independent set can contain
        vertices of the SAME color (since same-color vertices are NOT adjacent).
        So coloring gives a LOWER bound on MIS, not upper.
        
        For UPPER bound on MIS: we need to color the COMPLEMENT graph, or use
        a different technique.
        
        Actually, the standard B&B for MIS uses: the upper bound is the number of
        vertices remaining (trivial) or we use a packing argument.
        
        Let me use a different approach: the upper bound is min(n_remaining, something).
        
        Actually, a common upper bound for MIS is: for each vertex, at most 1 + (number of non-neighbors in remaining). But that's not tight.
        
        Let me use: upper_bound = len(candidates) as trivial, and rely on good branching order.
        
        Or better: use the fact that if we have a set of candidates, the MIS size is at most
        the maximum over all subsets... this is getting complex.
        
        Let me just use a simple but effective B&B:
        - Branch on a vertex: include it or exclude it
        - Upper bound: current_size + len(remaining_candidates)
        - Prune if upper_bound <= best_size
        """
        return len(candidates)
    
    # Branch and bound for Maximum Independent Set
    # State: (current_selected_size, candidates_set, forbidden_mask)
    # candidates: vertices we can still consider
    # We use recursion with bitmasks
    
    # For n up to ~245 vertices, we need to be smart
    # Let's use a different approach: iterative deepening or just good B&B
    
    # Actually, let me try a smarter greedy + local search approach first,
    # then use B&B only if we have time.
    
    def greedy_mis_v2():
        """Greedy MIS with degree-based ordering."""
        remaining = set(range(n))
        selected = []
        # Iteratively pick the vertex with minimum degree in remaining
        while remaining:
            # Find vertex with minimum degree within remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = 0
                mask = 0
                for u in remaining:
                    if u != v and (adj[v] & (1 << u)):
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                if min_deg == 0:
                    break
            selected.append(best_v)
            remaining.discard(best_v)
        return selected
    
    sel2 = greedy_mis_v2()
    if len(sel2) > best_size:
        best_size = len(sel2)
        best_set = set(sel2)
    
    # Local search: try to improve by swapping
    def verify_solution(sel):
        """Verify that the solution is valid."""
        sel_set = set(sel)
        for a in sel_set:
            if 2 * a in squares:
                return False
            for b in sel_set:
                if b >= a:
                    if (a + b) in squares:
                        return False
        return True
    
    # Local search to improve
    def local_search(sel):
        sel = list(sel)
        sel_set = set(sel)
        improved = True
        while improved:
            improved = False
            # Try adding vertices
            for v in range(n):
                if v in sel_set:
                    continue
                # Check if v can be added
                can_add = True
                for s in sel_set:
                    if (adj[v] & (1 << s)):
                        can_add = False
                        break
                if can_add:
                    sel.append(v)
                    sel_set.add(v)
                    improved = True
            # Try swapping: remove one, add two
            for i, v in enumerate(sel):
                sel_set.discard(v)
                # Try to add two vertices that conflict with v but not each other
                added = 0
                new_adds = []
                for u in range(n):
                    if u in sel_set or u == v:
                        continue
                    # Check if u conflicts with current sel_set
                    can_add_u = True
                    for s in sel_set:
                        if (adj[u] & (1 << s)):
                            can_add_u = False
                            break
                    if can_add_u:
                        # Check if u conflicts with new_adds
                        can_add_u2 = True
                        for na in new_adds:
                            if (adj[u] & (1 << na)):
                                can_add_u2 = False
                                break
                        if can_add_u2:
                            new_adds.append(u)
                            added += 1
                            if added == 2:
                                break
                if added >= 1:
                    sel.extend(new_adds)
                    for na in new_adds:
                        sel_set.add(na)
                    improved = True
                else:
                    sel_set.add(v)
        return sel
    
    sel3 = local_search(list(best_set))
    if verify_solution([valid_vertices[i] for i in sel3]):
        if len(sel3) > best_size:
            best_size = len(sel3)
            best_set = set(sel3)
    
    # Now try branch and bound for further improvement
    if time.time() - start_time < DEADLINE - 30:
        # B&B approach
        # Use the current best as lower bound
        # Branch on vertices, try to exceed best_size
        
        def bb_mis(candidates, current_sel, best_size_ref):
            """Branch and bound. candidates is a frozenset or list of vertex indices.
            current_sel is the set of selected vertices so far.
            Returns (size, set) of best MIS found."""
            nonlocal best_size, best_set
            
            if time.time() - start_time > DEADLINE:
                return (best_size, best_set)
            
            if not candidates:
                if len(current_sel) > best_size:
                    best_size = len(current_sel)
                    best_set = set(current_sel)
                return (best_size, best_set)
            
            # Upper bound
            ub = len(current_sel) + len(candidates)
            if ub <= best_size:
                return (best_size, best_set)
            
            # Choose branching vertex: pick one with max degree in candidates
            cand_list = list(candidates)
            # Order by degree
            cand_list.sort(key=lambda x: sum(1 for y in candidates if y != x and (adj[x] & (1 << y))), reverse=True)
            v = cand_list[0]
            
            # Option 1: Include v
            # New candidates: those in candidates that are not neighbors of v and not v
            new_cand = set()
            for u in candidates:
                if u == v:
                    continue
                if not (adj[v] & (1 << u)):
                    new_cand.add(u)
            s1, b1 = bb_mis(frozenset(new_cand), current_sel | {v}, best_size)
            best_size, best_set = s1, b1
            
            # Option 2: Exclude v
            new_cand2 = set(candidates)
            new_cand2.discard(v)
            s2, b2 = bb_mis(frozenset(new_cand2), current_sel, best_size)
            best_size, best_set = s2, b2
            
            return (best_size, best_set)
        
        # Start B&B with empty selection and all candidates
        # But this might be too slow for 245 vertices. Let's try with a smaller candidate set.
        # Use the vertices NOT in our current best solution as candidates for improvement
        # Actually, let's just try B&B on the full set but with a time limit.
        
        # For now, let's just return what we have if B&B is too complex
        pass
    
    # Verify final solution
    final_vals = [valid_vertices[i] for i in best_set]
    if not verify_solution(final_vals):
        # Fallback to safe solution
        final_vals = [valid_vertices[i] for i in initial_sel]
        if not verify_solution(final_vals):
            # Last resort: just pick non-diagonal-conflicting vertices greedily
            final_vals = []
            for a in range(1, N + 1):
                if 2 * a in squares:
                    continue
                ok = True
                for b in final_vals:
                    if (a + b) in squares:
                        ok = False
                        break
                if ok:
                    final_vals.append(a)
    
    return final_vals
