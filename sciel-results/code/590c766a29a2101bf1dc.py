import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Fallback: n=11 construction
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb
    
    # Try to find valid colorings for larger n
    # Prioritize 26 (known Lazebnik-Woldar construction exists), then 27, then others
    targets = [26, 27, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        # Allocate time budget
        if n == 26:
            t_limit = 60
        elif n == 27:
            t_limit = 30
        elif n == 25:
            t_limit = 20
        elif n <= 23:
            t_limit = 15
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 3, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n that works
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    """
    Find a valid 5-coloring of K_n with no monochromatic C4 using
    constraint-guided local search with targeted violation fixing.
    """
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # adj_mask[c][v] = bitmask of neighbors of v in color c
    adj_mask = [[0] * n for _ in range(5)]
    
    def build_masks(colors):
        for c in range(5):
            for v in range(n):
                adj_mask[c][v] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            
    def count_violations():
        """Count total C4 violations using bitmask intersections."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common & (common - 1):  # At least 2 bits set
                        # Count bits using a fast method
                        cnt = 0
                        m = common
                        while m:
                            cnt += 1
                            m &= m - 1
                        total += cnt - 1
        return total
        
    def find_violating_pairs():
        """Find all (u,v,c) triples where u,v have >=2 common neighbors in color c."""
        violations = []
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common & (common - 1):  # At least 2 bits set
                        # Fast bit count
                        cnt = 0
                        m = common
                        while m:
                            cnt += 1
                            m &= m - 1
                        violations.append((u, v, c, cnt))
        return violations
        
    def compute_delta(k, old_c, new_c):
        """Compute exact delta in violation count when changing edge k from old_c to new_c.
        
        A C4 violation for pair (a,b) in color c exists when |N_c(a) ∩ N_c(b)| >= 2.
        When we remove edge (u,v) from color c, for each w in N_c(v)\\{u}:
          - If |N_c(u) ∩ N_c(w)| was exactly 2 (including v), it drops to 1 → removes 1 violation contribution
          - If it was > 2, it still has >= 2 → no change in violation count (but cnt-1 decreases)
        Actually, violation count = sum over pairs of (cnt-1) where cnt>=2.
        So removing v from N_c(u) for pair (u,w): if cnt was exactly 2, contribution goes from 1 to 0 (delta -1)
        if cnt was > 2, contribution goes from cnt-1 to cnt-2 (delta -1)
        Wait, let me reconsider. The violation count is sum over all pairs (a,b) of max(0, |N(a)∩N(b)| - 1).
        
        When we remove edge (u,v) from color c:
        - For pair (u,w) where w in N_c(v), w != u: v is in N_c(u)∩N_c(w) (since w~v in c and u~v in c)
          After removal, v is no longer in N_c(u). So |N_c(u)∩N_c(w)| decreases by 1.
          If old count was k, new count is k-1.
          Old contribution: max(0, k-1). New: max(0, k-2).
          Delta = max(0,k-2) - max(0,k-1) = -1 if k>=2, 0 if k<=1.
          So delta -= 1 if (N_c(u)∩N_c(w)) has size >= 2 BEFORE removal (i.e., currently >= 2).
          
        Similarly for pair (v,w) where w in N_c(u), w != v.
        
        When we add edge (u,v) to color d:
        - For pair (u,w) where w in N_d(v), w != u: v is now in N_d(u)∩N_d(w)
          Old count: k. New: k+1.
          Old contribution: max(0,k-1). New: max(0,k).
          Delta = max(0,k) - max(0,k-1) = 1 if k>=1, 0 if k=0.
          So delta += 1 if (N_d(u)∩N_d(w)) has size >= 1 BEFORE addition (i.e., currently >= 1).
        """
        u = edge_u[k]
        v = edge_v[k]
        delta = 0
        
        # Removing (u,v) from old_c
        # For each w in N_oldc(v), w != u: check if |N_oldc(u) ∩ N_oldc(w)| >= 2
        mask_w = adj_mask[old_c][v]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == u:
                continue
            # Check if N_oldc(u) ∩ N_oldc(w) has at least 2 elements
            # (v is one of them, so we need at least one more)
            common = adj_mask[old_c][u] & adj_mask[old_c][w]
            # common includes v; we need |common| >= 2
            # Quick check: is there any bit in common other than v's bit?
            if common & ~(1 << v):
                delta -= 1
        
        # Adding (u,v) to new_c
        # For each w in N_newc(v), w != u: check if |N_newc(u) ∩ N_newc(w)| >= 1
        mask_w = adj_mask[new_c][v]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == u:
                continue
            # Check if N_newc(u) ∩ N_newc(w) is non-empty
            if adj_mask[new_c][u] & adj_mask[new_c][w]:
                delta += 1
        
        # Removing (v,u) from old_c
        # For each w in N_oldc(u), w != v: check if |N_oldc(v) ∩ N_oldc(w)| >= 2
        mask_w = adj_mask[old_c][u]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == v:
                continue
            common = adj_mask[old_c][v] & adj_mask[old_c][w]
            if common & ~(1 << u):
                delta -= 1
        
        # Adding (v,u) to new_c
        # For each w in N_newc(u), w != v: check if |N_newc(v) ∩ N_newc(w)| >= 1
        mask_w = adj_mask[new_c][u]
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if w == v:
                continue
            if adj_mask[new_c][v] & adj_mask[new_c][w]:
                delta += 1
                
        return delta
        
    def apply_change(k, old_c, new_c, colors):
        """Apply a color change to edge k."""
        u = edge_u[k]
        v = edge_v[k]
        adj_mask[old_c][u] &= ~(1 << v)
        adj_mask[old_c][v] &= ~(1 << u)
        adj_mask[new_c][u] |= (1 << v)
        adj_mask[new_c][v] |= (1 << u)
        colors[k] = new_c
        
    def targeted_fix(colors, current_violations, deadline):
        """Try to fix violations by targeting specific violating edges."""
        for _round in range(30):
            if time.time() > deadline - 1:
                break
            if current_violations == 0:
                return True, 0
                
            violations = find_violating_pairs()
            if not violations:
                return True, 0
            
            random.shuffle(violations)
            
            found_improvement = False
            for (u, v, c, cnt) in violations:
                if time.time() > deadline - 1:
                    break
                if current_violations == 0:
                    return True, 0
                    
                # Get common neighbors of u and v in color c
                common = adj_mask[c][u] & adj_mask[c][v]
                common_list = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    mask -= lsb
                    common_list.append(w)
                    
                if len(common_list) < 2:
                    continue
                
                # Try all pairs of common neighbors to find best edge to change
                best_delta = 0
                best_k = -1
                best_old_c = -1
                best_new_c = -1
                
                for idx1 in range(len(common_list)):
                    for idx2 in range(idx1 + 1, len(common_list)):
                        w1 = common_list[idx1]
                        w2 = common_list[idx2]
                        
                        # Candidate edges to change: (u,w1), (u,w2), (v,w1), (v,w2), (w1,w2)
                        candidates = []
                        for a, b in [(u, w1), (u, w2), (v, w1), (v, w2), (w1, w2)]:
                            if a < b:
                                candidates.append((a, b))
                            elif b < a:
                                candidates.append((b, a))
                                
                        for (a, b) in candidates:
                            k = pair_to_idx.get((a, b), -1)
                            if k < 0:
                                continue
                            
                            old_c = colors[k]
                            for new_c in range(5):
                                if new_c == old_c:
                                    continue
                                delta = compute_delta(k, old_c, new_c)
                                if delta < best_delta:
                                    best_delta = delta
                                    best_k = k
                                    best_old_c = old_c
                                    best_new_c = new_c
                                    if best_delta <= -3:
                                        break
                        if best_delta <= -3:
                            break
                    if best_delta <= -3:
                        break
                    
                if best_k >= 0 and best_delta < 0:
                    apply_change(best_k, best_old_c, best_new_c, colors)
                    current_violations += best_delta
                    found_improvement = True
                    if current_violations == 0:
                        return True, 0
            
            if not found_improvement:
                break
                
        return False, current_violations
        
    # Multi-restart with constraint-guided local search
    best_violations = float('inf')
    best_colors = None
    
    # Compute restart budget based on remaining time
    remaining = deadline - time.time()
    # Use larger restarts for better exploration
    num_restarts = max(1, int(remaining / 8))  # ~8s per restart
    restart_budget = remaining / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Structured initial coloring: use a more systematic approach
        # Based on the idea that for C4-free colorings, we want to spread neighbors
        colors = [0] * m
        for i in range(n):
            for j in range(i + 1, n):
                diff = j - i
                # Use a pattern that spreads colors more evenly
                # Try multiple structured patterns and pick based on restart
                if restart % 3 == 0:
                    c = (diff * 3 + i) % 5
                elif restart % 3 == 1:
                    c = (diff * 2 + i * 2) % 5
                else:
                    c = (diff + i * 3) % 5
                colors[i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)] = c
        
        # Add some randomness to break symmetry
        if restart > 0:
            num_random = m // 4
            for _ in range(num_random):
                k = random.randint(0, m - 1)
                colors[k] = random.randint(0, 4)
                
        build_masks(colors)
        
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 1: Targeted violation fixing
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 2: Random local search with reheating
        T = 2.0
        min_T = 0.1
        phase_deadline = min(deadline - 1, time.time() + restart_budget * 0.6)
        stagnant_steps = 0
        max_stagnant = 50000
        
        while time.time() < phase_deadline and stagnant_steps < max_stagnant:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            delta = compute_delta(k, old_c, new_c)
            
            # Acceptance: always accept improvements, sometimes accept worse
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_change(k, old_c, new_c, colors)
                current_violations += delta
                
                if current_violations == 0:
                    return True, colors
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
                    
                # Reheat if stuck
                if stagnant_steps > 20000:
                    T = min(2.0, T * 1.5)
                    stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            T *= 0.99995
            if T < min_T:
                T = min_T
                
        # Phase 3: More targeted fixing after random search
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
