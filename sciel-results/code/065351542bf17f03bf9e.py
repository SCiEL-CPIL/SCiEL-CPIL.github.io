import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 9
    # Fallback valid coloring for n=9 (Mod 3)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]
    best_colors = mod_color(9, 3)
    
    def is_valid(n, colors):
        """Verify a coloring is valid."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        # Check Red K4
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        # Check Blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def solve_n_bitparallel(n, time_limit):
        """
        Solve for n using bit-parallel local search.
        Variables: x_i = 1 if Red, 0 if Blue.
        Violations:
        - Red K4: 4 vertices where all 4 triples are Red.
        - Blue K5: 5 vertices where all 10 triples are Blue.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        quads = list(combinations(range(n), 4))
        num_quads = len(quads)
        quad_triples = []
        for a, b, c, d in quads:
            quad_triples.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
            
        quints = list(combinations(range(n), 5))
        num_quints = len(quints)
        quint_triples = []
        for q5 in quints:
            quint_triples.append([triple_to_idx[t] for t in combinations(q5, 3)])
            
        # Precompute bitmasks for each triple
        # mask_red[t] has bit qi set if triple t is in quad qi
        # mask_blue[t] has bit qi set if triple t is in quint qi
        # Using Python ints for arbitrary length bitmasks (up to ~280k bits for n=34)
        
        mask_red = [0] * num_triples
        for qi, qt in enumerate(quad_triples):
            bitmask = 1 << qi
            for tidx in qt:
                mask_red[tidx] |= bitmask
                
        mask_blue = [0] * num_triples
        for qi, qt in enumerate(quint_triples):
            bitmask = 1 << qi
            for tidx in qt:
                mask_blue[tidx] |= bitmask
        
        # State
        # colors: 1 for Red, 0 for Blue
        # We want to minimize Violations.
        # Violation R: Quad qi has 4 Reds.
        # Violation B: Quint qi has 10 Blues.
        
        def init_state(init_colors):
            # init_colors: list of 0/1
            # Return: colors list, red_quad_counts (list of ints), blue_quint_counts (list of ints)
            # Actually, for bit-parallel, we don't need counts for the whole state if we just check violations.
            # But for local search, we need to know which quads/quints are violating.
            # Let's maintain counts for fast delta calculation, OR just recompute violation sets?
            # Recomputing violation sets from scratch is O(M * n).
            # Delta update is O(1) per flip if we maintain counts.
            
            colors = np.array(init_colors, dtype=np.int8)
            red_qc = np.zeros(num_quads, dtype=np.int32)
            blue_qc = np.zeros(num_quints, dtype=np.int32)
            
            for qi in range(num_quads):
                cnt = 0
                for tidx in quad_triples[qi]:
                    if colors[tidx] == 1:
                        cnt += 1
                red_qc[qi] = cnt
                
            for qi in range(num_quints):
                cnt = 0
                for tidx in quint_triples[qi]:
                    if colors[tidx] == 0:
                        cnt += 1
            blue_qc[qi] = cnt
            
            # Count violations
            rk4 = np.sum(red_qc == 4)
            bk5 = np.sum(blue_qc == 10)
            
            return colors, red_qc, blue_qc, rk4, bk5

        def local_search(init_colors):
            colors, red_qc, blue_qc, rk4, bk5 = init_state(init_colors)
            
            if rk4 == 0 and bk5 == 0:
                return colors.tolist()
                
            # Lists of violating quads and quints
            # We will update these lists incrementally.
            violating_quads = set(np.where(red_qc == 4)[0].tolist())
            violating_quints = set(np.where(blue_qc == 10)[0].tolist())
            
            total_violations = rk4 + bk5
            best_total = total_violations
            best_colors = colors.copy()
            
            temp = 1.0
            min_temp = 0.01
            steps = 0
            max_steps = 10000000 # High limit, time bounded
            
            # Precompute inverse maps for fast flip updates
            # triple_in_quads[t] = list of quad indices
            # triple_in_quints[t] = list of quint indices
            triple_in_quads = [[] for _ in range(num_triples)]
            for qi, qt in enumerate(quad_triples):
                for tidx in qt:
                    triple_in_quads[tidx].append(qi)
                    
            triple_in_quints = [[] for _ in range(num_triples)]
            for qi, qt in enumerate(quint_triples):
                for tidx in qt:
                    triple_in_quints[tidx].append(qi)

            while time.time() < time_limit:
                if total_violations == 0:
                    return colors.tolist()
                    
                # Select a triple to flip
                # Strategy: Pick a random violation, pick a triple in it.
                # If no violations, random.
                if total_violations > 0:
                    if rk4 > 0 and (bk5 == 0 or random.random() < 0.5):
                        # Pick random violating quad
                        if violating_quads:
                            qi = random.choice(list(violating_quads))
                            tidx = quad_triples[qi][random.randint(0, 3)]
                        else:
                            tidx = random.randint(0, num_triples - 1)
                    else:
                        if violating_quints:
                            qi = random.choice(list(violating_quints))
                            tidx = quint_triples[qi][random.randint(0, 9)]
                        else:
                            tidx = random.randint(0, num_triples - 1)
                else:
                    tidx = random.randint(0, num_triples - 1)
                    
                old_color = colors[tidx]
                new_color = 1 - old_color
                
                # Calculate Delta
                delta_rk4 = 0
                delta_bk5 = 0
                
                # Update Quad Counts
                for qi in triple_in_quads[tidx]:
                    old_cnt = red_qc[qi]
                    if old_color == 1: # Red -> Blue
                        # Red count decreases
                        if old_cnt == 4:
                            delta_rk4 -= 1
                            violating_quads.discard(qi)
                        red_qc[qi] = old_cnt - 1
                    else: # Blue -> Red
                        # Red count increases
                        if old_cnt == 3:
                            delta_rk4 += 1
                            violating_quads.add(qi)
                        red_qc[qi] = old_cnt + 1
                        
                # Update Quint Counts
                for qi in triple_in_quints[tidx]:
                    old_cnt = blue_qc[qi]
                    if old_color == 0: # Blue -> Red
                        # Blue count decreases
                        if old_cnt == 10:
                            delta_bk5 -= 1
                            violating_quints.discard(qi)
                        blue_qc[qi] = old_cnt - 1
                    else: # Red -> Blue
                        # Blue count increases
                        if old_cnt == 9:
                            delta_bk5 += 1
                            violating_quints.add(qi)
                        blue_qc[qi] = old_cnt + 1
                        
                delta_total = delta_rk4 + delta_bk5
                
                # Acceptance
                if delta_total <= 0:
                    colors[tidx] = new_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5
                    total_violations += delta_total
                else:
                    # Accept with probability exp(-delta/temp)
                    if random.random() < np.exp(-delta_total / max(temp, 1e-10)):
                        colors[tidx] = new_color
                        rk4 += delta_rk4
                        bk5 += delta_bk5
                        total_violations += delta_total
                    else:
                        # Revert counts
                        for qi in triple_in_quads[tidx]:
                            if old_color == 1:
                                red_qc[qi] += 1
                                if red_qc[qi] == 4:
                                    violating_quads.add(qi)
                                elif red_qc[qi] == 3:
                                    violating_quads.discard(qi)
                            else:
                                red_qc[qi] -= 1
                                if red_qc[qi] == 4:
                                    violating_quads.add(qi)
                                elif red_qc[qi] == 3:
                                    violating_quads.discard(qi)
                                    
                        for qi in triple_in_quints[tidx]:
                            if old_color == 0:
                                blue_qc[qi] += 1
                                if blue_qc[qi] == 10:
                                    violating_quints.add(qi)
                                elif blue_qc[qi] == 9:
                                    violating_quints.discard(qi)
                            else:
                                blue_qc[qi] -= 1
                                if blue_qc[qi] == 10:
                                    violating_quints.add(qi)
                                elif blue_qc[qi] == 9:
                                    violating_quints.discard(qi)
                                  
                # Track best
                if total_violations < best_total:
                    best_total = total_violations
                    best_colors = colors.copy()
                    
                # Cooling
                temp *= 0.9999
                if temp < min_temp:
                    temp = min_temp
                    
                steps += 1
                if steps % 10000 == 0 and total_violations > 0:
                    # Restart if stuck?
                    pass

            # If we didn't find 0, return None or best if best is 0?
            # We only return if valid.
            if best_total == 0:
                return best_colors.tolist()
            return None

        # Initializations
        inits = []
        # 1. Random
        inits.append(np.random.randint(0, 2, size=num_triples).tolist())
        # 2. Mod 3
        inits.append([1 if (a+b+c)%3 == 0 else 0 for (a,b,c) in triples]) # Note: 1=Red here
        # 3. Mod 4
        inits.append([1 if (a+b+c)%4 == 0 else 0 for (a,b,c) in triples])
        # 4. Random biased
        inits.append([1 if random.random() < 0.5 else 0 for _ in range(num_triples)])
        
        random.shuffle(inits)
        
        for init in inits:
            if time.time() > time_limit - 5:
                break
            res = local_search(init)
            if res:
                return res
        return None

    # Main loop
    # We aim for n=34.
    # Start from 30 to ensure we have a valid base if 34 fails immediately?
    # No, just try 34. If it fails, try 33, etc.
    # But 34 is the target.
    
    # To be safe, ensure we return at least n=9.
    # Try n=34 first with most of the time.
    
    target_n = 34
    time_limit = deadline - 10
    res = solve_n_bitparallel(target_n, time_limit)
    
    if res:
        if is_valid(target_n, res):
            return {'n': target_n, 'colors': res}
    
    # If 34 failed, try 33?
    if time.time() < deadline - 30:
        target_n = 33
        res = solve_n_bitparallel(target_n, deadline - 10)
        if res:
            if is_valid(target_n, res):
                return {'n': target_n, 'colors': res}
                
    # Fallback to smaller n if time permits
    for n_try in range(32, 8, -1):
        if time.time() > deadline - 5:
            break
        res = solve_n_bitparallel(n_try, deadline - 5)
        if res:
            if is_valid(n_try, res):
                return {'n': n_try, 'colors': res}
                
    return {'n': best_n, 'colors': best_colors}
