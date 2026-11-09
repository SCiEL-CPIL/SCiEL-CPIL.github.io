"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # We aim for n=34, which is the known maximum.
    # If we can't find it, we fall back to the best found.
    target_n = 34
    
    # Precompute structures for the target n
    # This is done once.
    triple_list = list(combinations(range(target_n), 3))
    num_triples = len(triple_list)
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}
    
    # Structures for Red K4 (4 vertices, 4 triples)
    all_quads = list(combinations(range(target_n), 4))
    quad_triples_idx = []
    for a, b, c, d in all_quads:
        quad_triples_idx.append([
            triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
            triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
        ])
    num_quads = len(quad_triples_idx)
    
    # Structures for Blue K5 (5 vertices, 10 triples)
    all_quints = list(combinations(range(target_n), 5))
    quint_triples_idx = []
    for q5 in all_quints:
        quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
    num_quints = len(quint_triples_idx)
    
    # Inverse mappings: for each triple, which quads/quints contain it
    triple_in_quads = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quad_triples_idx):
        for pos, tidx in enumerate(qt):
            triple_in_quads[tidx].append((qi, pos))
            
    triple_in_quints = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quint_triples_idx):
        for pos, tidx in enumerate(qt):
            triple_in_quints[tidx].append((qi, pos))

    # Bitsets for quick violation checking
    # red_mask bit i is 1 if triple i is Red (0)
    # Note: In the problem, color 0 is Red, color 1 is Blue.
    # Red K4 violation: All 4 triples in a quad are Red (0).
    # Blue K5 violation: All 10 triples in a quint are Blue (1).
    
    # We will maintain:
    # colors: list of 0/1 (0=Red, 1=Blue)
    # red_mask: int, bit i=1 if colors[i]==0
    
    best_n_found = 9
    best_colors_found = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to solve for target_n
    # We will run multiple restarts if time permits, but usually one long run with good SA is better.
    # Let's try a few short restarts first to see if we get lucky, then a long one.
    
    def solve_n(n_target, time_budget):
        # Build structures for this specific n if it differs from target_n?
        # For now, we only target 34. If we want to be flexible, we'd need to rebuild.
        # Given the time limit, focusing on 34 is best.
        if n_target != target_n:
             # Fallback to smaller n if needed, but let's stick to 34 for this logic.
             return None, False
        
        # Initialize
        # Strategy: Random start, or slightly biased?
        # Random is 50/50.
        colors = np.array([random.randint(0, 1) for _ in range(num_triples)], dtype=np.int8)
        red_mask = 0
        for i in range(num_triples):
            if colors[i] == 0:
                red_mask |= (1 << i)
        
        # Count initial violations
        # Red K4: quad is violating if (red_mask & quad_bits) == quad_bits
        # Blue K5: quint is violating if (red_mask & quint_bits) == 0 (all blue)
        
        # Precompute bitsets for quads and quints
        quad_bits = [0] * num_quads
        for i, qt in enumerate(quad_triples_idx):
            b = 0
            for t in qt:
                b |= (1 << t)
            quad_bits[i] = b
            
        quint_bits = [0] * num_quints
        for i, qt in enumerate(quint_triples_idx):
            b = 0
            for t in qt:
                b |= (1 << t)
            quint_bits[i] = b
            
        # Count violations
        rk4 = 0
        for qb in quad_bits:
            if (red_mask & qb) == qb:
                rk4 += 1
                
        bk5 = 0
        for qb in quint_bits:
            if (red_mask & qb) == 0:
                bk5 += 1
                
        if rk4 == 0 and bk5 == 0:
            return colors.tolist(), True

        # Simulated Annealing
        start_time_sa = time.time()
        max_time = time_budget
        T_start = 2.0
        T_end = 0.001
        max_iters = 1000000 # Limit by time instead
        
        current_rk4 = rk4
        current_bk5 = bk5
        
        while time.time() - start_time_sa < max_time:
            if current_rk4 == 0 and current_bk5 == 0:
                return colors.tolist(), True
                
            # Decay temperature
            # Simple linear decay or exponential? Exponential is standard.
            # We need to estimate progress. Let's just use a fixed number of iterations for T decay,
            # or base it on time.
            # Let's do 10000 iterations per "step" of temperature? No, just compute T based on elapsed time.
            elapsed = time.time() - start_time_sa
            if elapsed > max_time: break
            # Map elapsed time to [0, 1]
            progress = elapsed / max_time
            T = T_start * (T_end / T_start) ** progress
            
            # Pick a triple to flip
            # Bias towards fixing violations
            tidx = -1
            r = random.random()
            
            # If there are red K4s, pick one and flip a triple in it
            if current_rk4 > 0 and r < 0.5:
                # Find a violating red K4
                # Randomly sample quads until we find one? 
                # Or keep a list of violating quads? Maintaining list is expensive.
                # Sampling is O(1) average if violation rate is high.
                for _ in range(50):
                    qi = random.randint(0, num_quads - 1)
                    if (red_mask & quad_bits[qi]) == quad_bits[qi]:
                        qt = quad_triples_idx[qi]
                        tidx = qt[random.randint(0, 3)]
                        break
                        # If not found in 50 tries, fall through to random
                        # But if current_rk4 > 0, probability of finding one in 50 tries is high
                        # unless rk4 is very small relative to num_quads.
                        
            elif current_bk5 > 0 and r < 0.85:
                # Find a violating blue K5
                for _ in range(50):
                    qi = random.randint(0, num_quints - 1)
                    if (red_mask & quint_bits[qi]) == 0:
                        qt = quint_triples_idx[qi]
                        tidx = qt[random.randint(0, 9)]
                        break
                        
            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)
                
            # Calculate delta
            old_color = colors[tidx]
            new_color = 1 - old_color
            
            delta_rk4 = 0
            delta_bk5 = 0
            
            # Update Red K4 count
            # For each quad containing tidx:
            # If old_color was Red (0):
            #   If the other 3 were Red, this quad was violating. Now it's not. delta_rk4 -= 1
            #   If the other 3 were not all Red, no change.
            # If old_color was Blue (1):
            #   If the other 3 were Red, this quad will become violating. delta_rk4 += 1
            #   If the other 3 were not all Red, no change.
            
            for qi, pos in triple_in_quads[tidx]:
                qt = quad_triples_idx[qi]
                # Check if other 3 are Red
                others_all_red = True
                for p in range(4):
                    if p == pos: continue
                    if colors[qt[p]] != 0:
                        others_all_red = False
                        break
                        
                if others_all_red:
                    if old_color == 0:
                        delta_rk4 -= 1
                    else:
                        delta_rk4 += 1
                        
            # Update Blue K5 count
            # Similar logic for Blue (1)
            for qi, pos in triple_in_quints[tidx]:
                qt = quint_triples_idx[qi]
                others_all_blue = True
                for p in range(10):
                    if p == pos: continue
                    if colors[qt[p]] != 1:
                        others_all_blue = False
                        break
                        
                if others_all_blue:
                    if old_color == 1:
                        delta_bk5 -= 1
                    else:
                        delta_bk5 += 1
                        
            delta = delta_rk4 + delta_bk5
            
            # Accept if delta <= 0 or with probability exp(-delta/T)
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                # Apply flip
                colors[tidx] = new_color
                if new_color == 0:
                    red_mask |= (1 << tidx)
                else:
                    red_mask &= ~(1 << tidx)
                current_rk4 += delta_rk4
                current_bk5 += delta_bk5
                
        return None, False

    # Run search
    # We have ~115 seconds.
    # Let's do 5 restarts of 20s each? Or 1 long run?
    # SA is stochastic. Multiple restarts with different seeds might help.
    # However, rebuilding structures is expensive. We built them for 34.
    # Let's do 3 restarts of ~35s each.
    
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        colors, success = solve_n(target_n, time_per_restart)
        
        if success:
            # Verify just in case
            # (Verification is expensive, but we trust the delta logic if implemented correctly.
            # Let's do a quick check on a sample or full check if n is small. For n=34, full check is O(n^5) ~ 10^6 ops, which is fast.)
            # Let's do a full verification to be safe.
            if _verify(target_n, colors):
                return {'n': target_n, 'colors': colors}
            else:
                # Should not happen if logic is correct, but fallback
                pass
        else:
            # If we didn't find a solution, we can try to salvage a partial one?
            # No, the task requires a valid coloring.
            # We could try smaller n if we fail on 34.
            pass

    # If we failed to find n=34, try n=33, 32...
    # This requires rebuilding structures.
    # Given the time, if we failed on 34, we likely don't have time to rebuild and search for 33.
    # However, the current program returns n=9. We should at least return a valid smaller n if we can't get 34.
    # The modular coloring for n=9 is valid.
    # Can we quickly check if a random coloring for n=10, 11... is valid?
    # Probability is low.
    
    # Let's stick with the best known valid small n if we fail.
    # But wait, the inspiration found n=26. We should try to find something better than 9.
    # Let's try a quick search for n=20, 25, 30 if 34 fails.
    
    # Actually, the structures are specific to n.
    # Let's just return the fallback n=9 if 34 fails, as rebuilding is costly.
    # OR, we can modify the code to allow searching for a specific n passed as argument.
    
    # Let's refactor to allow searching for any n, but optimize for 34.
    
    def solve_general(n_target, time_budget):
        if n_target == target_n:
            # Reuse precomputed structures? 
            # The structures are local to this function in the current draft.
            # Let's just call the inner logic.
            pass
            
    # For the sake of the 120s limit and complexity, if we don't hit 34, 
    # we return the fallback. The jump from 9 to 34 is huge.
    # The probability of hitting 34 with this SA is non-trivial but exists.
    
    return {'n': best_n_found, 'colors': best_colors_found}

def _verify(n, colors):
    """Verify a coloring is valid."""
    if len(colors) != len(list(combinations(range(n), 3))):
        return False
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

# EVOLVE-BLOCK-END
