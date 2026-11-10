import numpy as np
import time
import random

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 15.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    # Representation: element i is a + b*alpha, i = a*p + b
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    # Build line incidences
    # Lines are defined by (m, c) for y = mx + c, and vertical lines x = c.
    # Total lines: q (slopes) * q (intercepts) + q (verticals) = 2450
    # We map each point to the list of lines passing through it.
    point_lines = [[] for _ in range(q2)]
    line_points = [[] for _ in range(2450)]
    
    # Point index: pi = x*q + y
    # Line index: 
    #   Non-vertical: lid = m*q + c
    #   Vertical: lid = q*q + x0
    
    for x in range(q):
        for y in range(q):
            pi = x * q + y
            # Vertical line
            vl = q * q + x
            point_lines[pi].append(vl)
            line_points[vl].append(pi)
            
            # Non-vertical lines
            for m in range(q):
                # y = m*x + c => c = y - m*x
                c = sub_table[y, mul_table[m, x]]
                nl = m * q + c
                point_lines[pi].append(nl)
                line_points[nl].append(pi)
    
    # Convert to sets for faster membership checks if needed, 
    # but lists are fine for iteration.
    # We need to check if a line has any other points in C.
    # For a point p and line L, L \ {p} is disjoint from C iff 
    # no point in L \ {p} is in C.
    
    def compute_witness(in_C):
        """
        For each point i, compute the set of lines that are 'witnesses' for i.
        A line L is a witness for i if L \ {i} is disjoint from in_C.
        Returns a list of sets.
        """
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                valid = True
                for q_idx in line_points[lid]:
                    if q_idx == i:
                        continue
                    if in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        return W
    
    def is_valid(in_C, W):
        for i in range(q2):
            if len(W[i]) == 0:
                return False
        return True
    
    def try_remove_point(in_C, W, wcount, p_idx):
        """
        Attempt to remove p_idx from in_C.
        Returns (success, new_in_C, new_W, new_wcount)
        """
        # If we remove p_idx, it is no longer in C.
        # This might ADD witnesses for other points on lines through p_idx.
        # Specifically, for any point x on a line L through p_idx,
        # if L was invalid for x because p_idx was in C, it might become valid now.
        # But we also need to check if p_idx itself remains valid.
        # p_idx is no longer in C, so it needs a witness line L such that L \ {p_idx} is disjoint from C_new.
        # C_new = C \ {p_idx}.
        # So for p_idx, any line L through p_idx such that (L \ {p_idx}) \cap C_new is empty is a witness.
        # Note: if p_idx was the only point of C on L, then L was a witness for p_idx in C.
        # Now p_idx is not in C. The condition for p_idx is: exists L through p_idx s.t. no other point in L is in C_new.
        
        # Let's just recompute W for the affected points.
        # Affected points are p_idx and all points on lines through p_idx.
        # Actually, removing a point from C can only make lines "cleaner", so it can only ADD witnesses, never remove them.
        # Wait. If I remove p from C:
        # 1. For p itself: It is now in N. It needs a line L through p with L\{p} subset N.
        #    This is equivalent to L \ {p} disjoint from C_new.
        #    Previously, p was in C. It needed a line L through p with L\{p} disjoint from C.
        #    Since C_new is smaller, if it was valid before, it is valid now?
        #    No. The condition for x in C is L \ {x} disjoint from C.
        #    The condition for x in N is L \ {x} disjoint from C.
        #    It is the same condition!
        #    So if x had a witness in C, it has a witness in C_new (since C_new subset C).
        #    Therefore, removing a point from C NEVER invalidates any point.
        #    It can only help.
        
        # So, if the current configuration is valid, removing ANY point from C results in a valid configuration.
        # This implies that the minimal C is the empty set?
        # If C is empty, N is the whole space.
        # For any x, take any line L through x. L \ {x} is subset of N (which is everything).
        # So the empty set C is a valid solution with size 0.
        # The problem asks for a Nikodym set N.
        # "N must be a Nikodym set: every point x ... must lie on a line ... whose points other than x all belong to N."
        # If N = F_49^2, this is trivially true.
        # But the metric is "complement size 2401 - |N|".
        # To maximize the score, we want to maximize 2401 - |N|, which means minimizing |N|?
        # No, "Metric (higher is better): the complement size 2401 - |N|".
        # So we want to MAXIMIZE 2401 - |N|, which means MINIMIZING |N|.
        # This means we want to MINIMIZE |N|, or MAXIMIZE |C|.
        
        # Let's re-read carefully.
        # "Metric (higher is better): the complement size 2401 - |N|"
        # Score is proportional to this.
        # So we want 2401 - |N| to be as large as possible.
        # This implies |N| should be as small as possible.
        # And |C| = 2401 - |N| should be as large as possible.
        
        # So we want to MAXIMIZE |C|.
        # My previous logic was "minimizing |C|". That was wrong.
        # We want the LARGEST possible set C such that every point has a witness line disjoint from C\{x}.
        
        # Okay, so the greedy ADDITION approach in the current code is correct in direction.
        # We start with small C and add points.
        # The "pruning" idea was wrong because we want to GROW C.
        
        # So, how to improve the growth?
        # The current code does greedy addition.
        # A better approach: Simulated Annealing or Local Search with Swaps?
        # Or a smarter greedy:
        # Instead of adding one point at a time, look for a point that "costs" the least in terms of destroying witnesses of others.
        # The current code does this: `damage` counts how many points would lose their last witness.
        # It picks the point with minimum damage.
        
        # Structural change: 
        # Use a priority queue to always pick the "best" candidate point to add.
        # Or, restart with different random seeds and keep the best.
        # The current code does 30 trials.
        
        # Let's try a different structural idea:
        # The constraint is local.
        # Maybe we can construct C line by line?
        # No, points are on multiple lines.
        
        # Let's stick to the greedy addition but optimize the implementation and search strategy.
        # The current code uses `rng.sample` for candidates.
        # We can use a heap to prioritize points with low "damage".
        # However, damage changes as we add points.
        
        # Alternative: 
        # Start with a random C of moderate density, then try to ADD more points.
        # The current code does this.
        
        # Let's try to optimize the `compute_witness` and `greedy_add` for speed.
        # `compute_witness` is O(N * L * L) which is slow.
        # We can maintain `wcount` incrementally.
        
        # Revised Plan:
        # 1. Initialize C.
        # 2. Compute W and wcount.
        # 3. Use a max-heap or list to track potential additions.
        # 4. When adding a point p, update W and wcount for neighbors.
        # 5. If wcount[x] becomes 0, x is invalid. We must NOT add p if it causes this.
        #    Actually, the current code checks `if wcount[x_idx] == 1` and `lid in W[x_idx]`, then damage++.
        #    If damage > 0, it means some point would lose its LAST witness.
        #    So we only add if damage == 0.
        
        # This is a valid constraint satisfaction problem.
        # To improve, we can try to add points in a more intelligent order.
        # E.g., points that are "isolated" or on lines with many N points.
        
        # Let's implement a faster version of the current logic with better randomization and a deadline guard.
        
        pass

    # --- Actual Implementation ---
    
    best_in_C = None
    best_size = 0
    
    rng = random.Random(42)
    
    def compute_witness_and_count(in_C):
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                valid = True
                for q_idx in line_points[lid]:
                    if q_idx == i:
                        continue
                    if in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        return W, wcount
    
    def try_greedy_add(in_C, W, wcount, rng, max_steps=20000):
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        if not is_valid(in_C, W):
            return in_C, W, wcount, 0
        
        candidates = [i for i in range(q2) if not in_C[i]]
        added = 0
        
        for _ in range(max_steps):
            if time.time() > _DEADLINE - _MARGIN:
                break
            if not candidates:
                break
            
            best_p = -1
            best_damage = float('inf')
            
            # Sample candidates to speed up
            sample_size = min(len(candidates), 200)
            if sample_size < len(candidates):
                sample = rng.sample(candidates, sample_size)
            else:
                sample = candidates
            
            for p_idx in sample:
                if in_C[p_idx]:
                    continue
                damage = 0
                for lid in point_lines[p_idx]:
                    for x_idx in line_points[lid]:
                        if x_idx == p_idx:
                            continue
                        if lid in W[x_idx]:
                            if wcount[x_idx] == 1:
                                damage += 1
                                if damage >= best_damage:
                                    break
                    if damage >= best_damage:
                        break
                if damage < best_damage:
                    best_damage = damage
                    best_p = p_idx
            
            if best_p == -1 or best_damage > 0:
                break
            
            in_C[best_p] = True
            added += 1
            candidates.remove(best_p)
            
            for lid in point_lines[best_p]:
                for x_idx in line_points[lid]:
                    if x_idx == best_p:
                        continue
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
                        wcount[x_idx] -= 1
        
        return in_C, W, wcount, added

    # Try different initial densities
    densities = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    for trial in range(50):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        density = densities[trial % len(densities)]
        in_C = np.zeros(q2, dtype=bool)
        n_points = int(q2 * density)
        indices = rng.sample(range(q2), n_points)
        for i in indices:
            in_C[i] = True
        
        W, wcount = compute_witness_and_count(in_C)
        
        if not is_valid(in_C, W):
            # Try to repair by removing points
            # This is expensive, so maybe just skip if invalid?
            # Or do a simple repair: remove random points until valid.
            valid = False
            for _ in range(1000):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                # Find a point with no witness
                bad = [i for i in range(q2) if wcount[i] == 0]
                if not bad:
                    valid = True
                    break
                # Remove a random point from C
                c_pts = [i for i in range(q2) if in_C[i]]
                if not c_pts:
                    break
                rem = rng.choice(c_pts)
                in_C[rem] = False
                # Recompute W and wcount is expensive.
                # Let's just recompute for simplicity in repair, but limit iterations.
                W, wcount = compute_witness_and_count(in_C)
            
            if not valid:
                continue
        
        in_C, W, wcount, added = try_greedy_add(in_C, W, wcount, rng, max_steps=5000)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
            
    if best_in_C is None:
        # Fallback: Empty C (N = all points)
        best_in_C = np.zeros(q2, dtype=bool)
        
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
