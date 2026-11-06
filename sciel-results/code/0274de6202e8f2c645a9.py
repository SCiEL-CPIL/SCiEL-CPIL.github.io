import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill

    def is_not_2colorable(n, edge_list):
        """Check if hypergraph has NO proper 2-coloring using brute-force enumeration for small n.
        Returns True if NOT 2-colorable (UNSAT), False if 2-colorable (SAT)."""
        if time.time() > deadline:
            return True  # Conservative: assume not colorable
        
        m = len(edge_list)
        if m == 0:
            return False
        
        if n <= 14:
            # Brute force: try all 2^(n-1) colorings (fix vertex 0 = 0 by symmetry)
            # Precompute edge masks
            edge_masks = []
            for e in edge_list:
                em = 0
                for v in e:
                    em |= (1 << v)
                edge_masks.append(em)
            
            # By symmetry, fix vertex 0 to color 0
            # So we iterate over masks for vertices 1..n-1
            for mask in range(1 << (n - 1)):
                # full_mask has bit 0 = 0 (vertex 0 is color 0)
                full_mask = mask  # bits 0..n-1, bit 0 is 0
                # Check all edges
                ok = True
                for em in edge_masks:
                    if (full_mask & em) == 0 or (full_mask & em) == em:
                        ok = False
                        break
                if ok:
                    return False  # Found a valid coloring, so it IS 2-colorable
            return True  # No coloring found, NOT 2-colorable
        
        # For n > 14, use backtracking with unit propagation
        assignment = [-1] * n
        edge_state = [2] * m  # 2 = unassigned, -1 = satisfied/mixed
        
        def propagate(assignment, edge_state, vert_edges, edge_sets, n, m):
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    if edge_state[i] == -1:
                        continue
                    e = edge_sets[i]
                    zeros = 0
                    ones = 0
                    unassigned = []
                    for v in e:
                        a = assignment[v]
                        if a == 0:
                            zeros += 1
                        elif a == 1:
                            ones += 1
                        else:
                            unassigned.append(v)
                    
                    if ones > 0 and zeros > 0:
                        edge_state[i] = -1
                        changed = True
                    elif ones > 0 and zeros == 0:
                        # Need at least one 0
                        if not unassigned:
                            edge_state[i] = -1
                            changed = True
                        else:
                            # All unassigned must not be forced to 1
                            # Actually: we need at least one 0 among unassigned
                            # If all unassigned become 1, conflict. For now, just mark satisfied if ones>0
                            edge_state[i] = -1
                            changed = True
                    elif zeros > 0 and ones == 0:
                        # Need at least one 1
                        if not unassigned:
                            edge_state[i] = -1
                            changed = True
                        else:
                            edge_state[i] = -1
                            changed = True
                    elif zeros == 0 and ones == 0:
                        # All unassigned, no constraint yet
                        pass
            
            # Check for conflicts
            for i in range(m):
                if edge_state[i] == -1:
                    continue
                e = edge_sets[i]
                has0 = any(assignment[v] == 0 for v in e)
                has1 = any(assignment[v] == 1 for v in e)
                if has0 and has1:
                    return False, "conflict"
            return True, ""
        
        def backtrack(assignment, edge_state, n, m):
            if time.time() > deadline:
                return None
            if not any(a == -1 for a in assignment):
                for i in range(m):
                    e = edge_sets[i]
                    colors = set(assignment[v] for v in e)
                    if len(colors) < 2:
                        return None
                return assignment.copy()
            
            # MRV heuristic
            best_v = -1
            best_degree = 0
            for v in range(n):
                if assignment[v] == -1:
                    d = len(vert_edges[v])
                    if d > best_degree:
                        best_degree = d
                        best_v = v
            if best_v == -1:
                return assignment.copy()
            
            for color in [0, 1]:
                new_assignment = assignment.copy()
                new_edge_state = edge_state.copy()
                new_assignment[best_v] = color
                ok, _ = propagate(new_assignment, new_edge_state, vert_edges, edge_sets, n, m)
                if ok:
                    result = backtrack(new_assignment, new_edge_state, n, m)
                    if result is not None:
                        return result
            return None
        
        vert_edges = [[] for _ in range(n)]
        for i, e in enumerate(edge_list):
            for v in e:
                vert_edges[v].append(i)
        edge_sets = [frozenset(e) for e in edge_list]
        
        ok, _ = propagate(assignment, edge_state, vert_edges, edge_sets, n, m)
        if not ok:
            return True
        
        result = backtrack(assignment, edge_state, n, m)
        return result is None

    def count_valid_colorings(n, edge_list):
        """Count the number of valid 2-colorings (up to symmetry, fix vertex 0 = 0)."""
        if n > 14:
            return None  # Too large for brute force
        
        edge_masks = []
        for e in edge_list:
            em = 0
            for v in e:
                em |= (1 << v)
            edge_masks.append(em)
        
        count = 0
        for mask in range(1 << (n - 1)):
            full_mask = mask
            ok = True
            for em in edge_masks:
                if (full_mask & em) == 0 or (full_mask & em) == em:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    # Greedy edge selection: start with empty set, add edges that most reduce valid colorings
    # Goal: reach 0 valid colorings (non-2-colorable)
    
    best_result = None
    best_m = 200
    
    # Try different n values
    for n in [10, 11, 12, 13, 14]:
        if time.time() > deadline - 20:
            break
        
        all_possible = list(itertools.combinations(range(n), 5))
        num_all = len(all_possible)
        
        # Greedy selection
        current_edges = []
        current_count = count_valid_colorings(n, current_edges)
        
        # Precompute edge masks for all possible edges
        all_edge_masks = []
        for e in all_possible:
            em = 0
            for v in e:
                em |= (1 << v)
            all_edge_masks.append(em)
        
        available_indices = set(range(num_all))
        
        while available_indices:
            if time.time() > deadline - 10:
                break
            
            # For each available edge, compute how many colorings it would eliminate
            # A coloring is eliminated if it makes the edge monochromatic
            # For edge with mask em, a coloring (mask) eliminates it if:
            #   (mask & em) == 0 (all 0) or (mask & em) == em (all 1)
            
            best_idx = None
            best_eliminated = 0
            
            # Sample if too many candidates
            if len(available_indices) > 200:
                candidates = random.sample(list(available_indices), 200)
            else:
                candidates = list(available_indices)
            
            for idx in candidates:
                em = all_edge_masks[idx]
                # Count colorings that make this edge monochromatic
                # AND are still valid for current edges
                eliminated = 0
                for mask in range(1 << (n - 1)):
                    # Check if this coloring is valid for current edges
                    valid_current = True
                    for ce in current_edges:
                        cem = 0
                        for v in ce:
                            cem |= (1 << v)
                        if (mask & cem) == 0 or (mask & cem) == cem:
                            valid_current = False
                            break
                    if not valid_current:
                        continue
                    # Check if this coloring makes the candidate edge monochromatic
                    if (mask & em) == 0 or (mask & em) == em:
                        eliminated += 1
                
                if eliminated > best_eliminated:
                    best_eliminated = eliminated
                    best_idx = idx
            
            if best_idx is None or best_eliminated == 0:
                break
            
            # Add the best edge
            current_edges.append(all_possible[best_idx])
            available_indices.discard(best_idx)
            
            # Check if we've reached 0 valid colorings
            new_count = count_valid_colorings(n, current_edges)
            if new_count == 0:
                if len(current_edges) < best_m:
                    best_m = len(current_edges)
                    best_result = current_edges[:]
                break
            
            current_count = new_count
    
    # Try to optimize: remove edges from the best result
    if best_result is not None:
        current = best_result[:]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                trial = current[:i] + current[i+1:]
                n = max(max(e) for e in trial) + 1 if trial else 10
                if is_not_2colorable(n, trial):
                    current = trial
                    improved = True
                    break
        
        if len(current) < best_m:
            best_m = len(current)
            best_result = current
    
    # If we found something, return it
    if best_result is not None:
        n = max(max(e) for e in best_result) + 1 if best_result else 10
        edges = [list(e) for e in best_result]
        return {'n': n, 'edges': edges}
    
    # Fallback: complete hypergraph on small n
    n = 9
    edges = [list(e) for e in itertools.combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
