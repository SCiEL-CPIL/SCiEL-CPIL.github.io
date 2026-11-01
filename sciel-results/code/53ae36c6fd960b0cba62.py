import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all candidate vectors
    # Type A: weight-2 vectors in {-1,0,1}^11 (norm² = 2)
    type_a = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_a.append(v)
    type_a = np.array(type_a, dtype=np.int64)
    n_a = len(type_a)

    # Type C: weight-4 vectors in {-1,0,1}^11 (norm² = 4)
    # ALL Type C vectors are mutually compatible (max dot between distinct = 1)
    type_c = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_c.append(v)
    type_c = np.array(type_c, dtype=np.int64)
    n_c = len(type_c)

    # Determine which Type A vectors conflict with ANY Type C vector
    # Type A (norm²=2) vs Type C (norm²=4): dist² = 6 - 2*dot
    # Conflict when dist² < 4, i.e., dot > 1, i.e., dot = 2
    # dot = 2 means: both nonzero positions of A are in C's support with same sign
    
    # For each Type A vector, check if there exists a Type C vector with dot = 2
    # A Type A vector at positions (i,j) with signs (s1,s2) conflicts with Type C
    # iff there's a Type C vector that has s1 at position i and s2 at position j
    # (and 2 other nonzeros anywhere)
    
    # Actually, for ANY Type A vector at positions (i,j) with signs (s1,s2),
    # we can construct a Type C vector with those same entries plus any 2 other positions.
    # So ALL Type A vectors conflict with SOME Type C vector!
    
    # Wait, let me re-examine. If we include ALL Type C, then no Type A can be added
    # because every Type A conflicts with at least one Type C.
    
    # Alternative: Don't include all Type C. Instead, find a better combination.
    
    # Let's think differently. The conflict graph:
    # - Type A vs Type A: dist² = 4 - 2*dot. Conflict when dot > 0, i.e., dot >= 1
    #   (since both have 2 nonzeros, dot can be -2,-1,0,1,2; conflict when dot >= 1)
    # - Type C vs Type C: dist² = 8 - 2*dot. Conflict when dot > 2, i.e., dot >= 3
    #   (max dot between distinct weight-4 vectors is 2, so NO conflicts among Type C)
    # - Type A vs Type C: dist² = 6 - 2*dot. Conflict when dot > 1, i.e., dot >= 2
    #   (dot = 2 when A's two positions match C's entries at those positions)
    # - Type B: weight-1 with value ±2 (norm²=4). Let's skip Type B for now.
    
    # So the structure is:
    # - All Type C are mutually compatible
    # - Type A conflicts with Type A when they share a position with same sign (dot >= 1)
    # - Type A conflicts with Type C when A's entries match C at both A's positions
    
    # Strategy: Include all Type C (33264 vectors), then find max Type A subset
    # where no Type A conflicts with any included Type C.
    # But since every Type A conflicts with some Type C, this gives 0 Type A.
    
    # Better: Find a large subset of Type C that excludes certain "blocking" C vectors,
    # allowing some Type A to be added.
    
    # Even better: Just work with Type A + Type C together as one big independent set problem,
    # but exploit the structure.
    
    # Actually, let's reconsider. The total candidate set is:
    # n_a = 11*10*2 = 220 Type A vectors
    # n_c = C(11,4)*16 = 330*16 = 5280 Type C vectors
    
    # Wait, C(11,4) = 330, so n_c = 330 * 16 = 5920
    
    # Let me recompute: C(11,4) = 11!/(4!*7!) = (11*10*9*8)/(4*3*2*1) = 7920/24 = 330
    # So n_c = 330 * 16 = 5280
    
    # Total candidates: 220 + 5280 = 5500
    
    # Build the full conflict graph and find MIS
    
    all_candidates = np.vstack([type_a, type_c])
    n = len(all_candidates)
    
    # Compute norms squared
    norms_sq = np.sum(all_candidates ** 2, axis=1)
    
    # Compute dot products
    dots = all_candidates @ all_candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = (dist_sq < 4).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    conflict_bool = conflicts.astype(bool)
    
    # For Type C vertices (indices n_a to n-1), they have no conflicts among themselves
    # So we can greedily add all Type C, then add Type A that don't conflict
    
    # But we want to maximize total. Let's use a smarter approach:
    # 1. Start with all Type C selected
    # 2. Remove Type C vertices that block many Type A vertices
    # 3. Add Type A vertices
    
    # Actually, let's just do a proper MIS search on the 5500-node graph
    # using greedy + local search
    
    best_selected = None
    best_count = 0
    
    # Strategy 1: Greedy minimum degree
    def greedy_min_degree(seed):
        rng = np.random.default_rng(seed)
        alive = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflict_bool[chosen] | (np.arange(n) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        return selected, selected.sum()
    
    # Run greedy with several seeds
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        sel, cnt = greedy_min_degree(seed)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
    
    # Strategy 2: Exploit structure - start with all Type C, then optimize
    if time.time() < deadline - 30:
        selected = np.zeros(n, dtype=bool)
        # Include all Type C
        selected[n_a:] = True
        
        # Find Type A vertices that don't conflict with any selected Type C
        # For each Type A, check if it conflicts with any Type C
        type_a_conflicts_with_c = conflict_bool[:n_a, n_a:]  # shape (n_a, n_c)
        a_conflict_count = type_a_conflicts_with_c.sum(axis=1)
        
        # Type A vertices with 0 conflicts with Type C can be added
        addable_a = np.where(a_conflict_count == 0)[0]
        for v in addable_a:
            selected[v] = True
        
        cnt = selected.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = selected.copy()
        
        # Now try to remove some Type C to allow more Type A
        # For each Type C, count how many Type A it blocks
        # Type A v is blocked by Type C c if conflict_bool[v, c] = 1
        # A Type A is blocked if ANY selected Type C conflicts with it
        
        # Compute for each Type C, which Type A it conflicts with
        c_conflicts_a = conflict_bool[n_a:, :n_a]  # shape (n_c, n_a)
        
        # For each Type A, list which Type C conflict with it
        a_blocked_by = []
        for i in range(n_a):
            blockers = np.where(c_conflicts_a[:, i])[0]
            a_blocked_by.append(blockers)
        
        # Greedy: remove Type C that unblocks the most Type A
        improved = True
        max_rounds = 100
        for _ in range(max_rounds):
            if time.time() > deadline - 15:
                break
            improved = False
            
            # Find Type A that are currently blocked
            blocked_a = []
            for i in range(n_a):
                if not selected[i]:
                    # Check if it's blocked by selected Type C
                    sel_c = np.where(selected[n_a:] & c_conflicts_a[:, i])[0]
                    if len(sel_c) > 0:
                        blocked_a.append((i, sel_c))
            
            if not blocked_a:
                break
            
            # For each Type C, count how many blocked Type A it blocks
            c_block_count = np.zeros(n_c, dtype=np.int64)
            for i, blockers in blocked_a:
                for c_idx in blockers:
                    if selected[n_a + c_idx]:
                        c_block_count[c_idx] += 1
            
            # Find Type C with max block count
            max_block = np.max(c_block_count)
            if max_block == 0:
                break
            
            # Try removing the Type C with max block count
            best_c = -1
            best_gain = 0
            for c_idx in np.where(c_block_count == max_block)[0]:
                # If we remove this Type C, how many Type A become unblocked?
                # A Type A becomes unblocked if this was its ONLY selected blocker
                unblocked = 0
                for i, blockers in blocked_a:
                    if c_idx in blockers:
                        # Check if this was the only selected blocker
                        other_selected_blockers = np.setdiff1d(blockers, [c_idx])
                        other_selected = other_selected_blockers[selected[n_a:][other_selected_blockers]] if len(other_selected_blockers) > 0 else np.array([])
                        if len(other_selected) == 0:
                            unblocked += 1
                
                gain = unblocked - 1  # We lose 1 Type C, gain unblocked Type A
                if gain > best_gain:
                    best_gain = gain
                    best_c = c_idx
            
            if best_c >= 0 and best_gain > 0:
                selected[n_a + best_c] = False
                improved = True
                
                # Add newly unblocked Type A
                for i in range(n_a):
                    if not selected[i]:
                        sel_c_blockers = np.where(selected[n_a:] & c_conflicts_a[:, i])[0]
                        if len(sel_c_blockers) == 0:
                            selected[i] = True
                        # Also check Type A vs Type A conflicts
                        # Need to verify no conflict with other selected Type A
                        ta_conflicts = np.where(conflict_bool[i, :n_a] & selected[:n_a])[0]
                        if len(ta_conflicts) == 0:
                            selected[i] = True
                        else:
                            selected[i] = False  # Revert if conflicts with Type A
        
        cnt = selected.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = selected.copy()
    
    # Strategy 3: Local search refinement
    if best_selected is not None:
        selected = best_selected.copy()
        improved = True
        max_rounds = 200
        round_num = 0
        
        while improved and round_num < max_rounds:
            if time.time() > deadline - 10:
                break
            improved = False
            round_num += 1
            
            non_selected_indices = np.where(~selected)[0]
            
            if len(non_selected_indices) == 0:
                break
            
            # Try to add any non-selected vertex with 0 conflicts with selected
            for v in non_selected_indices:
                if time.time() > deadline - 10:
                    break
                sel_neighbors = np.where(conflict_bool[v] & selected)[0]
                if len(sel_neighbors) == 0:
                    selected[v] = True
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one selected u, add one non-selected v
            # where v conflicts with exactly one selected (u)
            # Net change = 0, but might enable future improvements
            rng = np.random.default_rng(round_num)
            best_swap = None
            best_swap_score = 0
            
            for v in non_selected_indices:
                if time.time() > deadline - 10:
                    break
                sel_neighbors = np.where(conflict_bool[v] & selected)[0]
                if len(sel_neighbors) == 1:
                    u = sel_neighbors[0]
                    # Check if removing u and adding v enables more additions
                    # Heuristic: count how many non-selected vertices would become addable
                    # This is expensive, so just do random swaps
                    if rng.random() < 0.05:
                        selected[u] = False
                        selected[v] = True
                        improved = True
                        break
        
        # Final polish: add any vertex with no conflicts
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            changed = False
            non_selected_indices = np.where(~selected)[0]
            for v in non_selected_indices:
                sel_neighbors = np.where(conflict_bool[v] & selected)[0]
                if len(sel_neighbors) == 0:
                    selected[v] = True
                    changed = True
            if not changed:
                break
        
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    result = all_candidates[best_selected]
    return result
