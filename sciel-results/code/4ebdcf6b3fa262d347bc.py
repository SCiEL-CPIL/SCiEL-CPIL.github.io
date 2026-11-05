"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def check_validity(n, colors):
        """Check if coloring is valid: no red K4^(3) and no blue K5^(3)."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check red K4^(3)
        for quad in combinations(range(n), 4):
            idxs = [triple_to_idx[(quad[i], quad[j], quad[k])] 
                    for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
            if all(colors[i] == 0 for i in idxs):
                return False
        
        # Check blue K5^(3)
        for quint in combinations(range(n), 5):
            idxs = [triple_to_idx[(quint[i], quint[j], quint[k])] 
                    for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
            if all(colors[i] == 1 for i in idxs):
                return False
        
        return True

    def generate_initial_coloring(n, seed_offset):
        """Generate a high-quality initial coloring using finite field algebra."""
        rng = np.random.RandomState(seed_offset)
        
        # Use GF(2^k) arithmetic for a structured random-like start
        # We simulate a field of size 2^8 = 256 which is enough for n up to 40
        # Primitive polynomial for GF(256): x^8 + x^4 + x^3 + x^2 + 1
        def gf_mul(a, b):
            p = 0
            for _ in range(8):
                if b & 1:
                    p ^= a
                b >>= 1
                a <<= 1
                if a & 0x100:
                    a ^= 0x11D # x^8 + x^4 + x^3 + x^2 + 1
            return p

        # Map vertices to field elements
        # We want a "generic" position for each vertex
        # Using a random permutation of field elements
        field_elements = list(range(256))
        rng.shuffle(field_elements)
        vertex_map = {v: field_elements[v] for v in range(n)}
        
        # Coloring rule: Red (0) if sum of field elements is in a specific subset
        # Let's use: Red if gf_sum(x, y, z) has a specific property
        # Actually, a simple rule: Red if (x+y+z) in GF(256) is "small" or satisfies a linear condition
        # A better rule for 3-uniform: Red if x, y, z are "affinely independent" in a certain way?
        
        # Let's try a simpler algebraic rule:
        # Color (a,b,c) Red if gf_mul(vertex_map[a], vertex_map[b]) + vertex_map[c] == 0 ? No, that's too restrictive.
        
        # Let's use: Red if (vertex_map[a] ^ vertex_map[b] ^ vertex_map[c]) is in a precomputed "Red Set"
        # The Red Set should be roughly half the field to balance colors.
        # To avoid K4, we need to ensure no 4 points have their 4 triples all in Red Set.
        
        # Alternative: Use a random hash of the triple indices but with a bias towards avoiding local clusters.
        # For now, let's stick to a strong random start but with a "structured" seed.
        
        # Actually, the most effective initial state for SA is often just random, 
        # but we can try a "modular" start which is known to work for small n.
        
        # Let's try a mix:
        colors = np.zeros(n*(n-1)*(n-2)//6, dtype=np.int8)
        
        # Modular construction: (a+b+c) % 3
        # This works for n=9, but breaks for larger n.
        # Let's use it as a base and then perturb.
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Use a more complex modular rule
                    # (a*7 + b*11 + c*13) % 5 < 2 ?
                    val = (a * 7 + b * 11 + c * 13) % 5
                    colors[idx] = 0 if val < 2 else 1
                    idx += 1
        
        return colors

    def solve_sa(n, max_time, seed_offset):
        """Solve using Simulated Annealing with vectorized violation counting."""
        m = n * (n - 1) * (n - 2) // 6
        if m == 0:
            return np.array([], dtype=np.int8)
        
        # Precompute triples
        triples = np.array(list(combinations(range(n), 3)))
        
        # Precompute indices for Red K4s and Blue K5s
        # For speed, we only check violations when needed, but we need to know which triples to flip.
        
        # Initial coloring
        colors = generate_initial_coloring(n, seed_offset)
        
        # Helper to get violation counts
        def get_violations(cols):
            red_v = 0
            blue_v = 0
            # Check Red K4
            for quad in combinations(range(n), 4):
                idxs = []
                for i in range(4):
                    for j in range(i+1, 4):
                        for k in range(j+1, 4):
                            # Find index of (quad[i], quad[j], quad[k]) in triples
                            # This is slow. We need a mapping.
                            pass
            # This naive check is too slow for SA.
            # We need a faster way to count violations or just accept/reject based on energy change.
            # Energy = number of red K4s + number of blue K5s.
            # We can't easily compute delta E without checking all affected K4s/K5s.
            
            # Strategy:
            # 1. Pick a random batch of triples to flip.
            # 2. Compute new violations for the whole graph? No, too slow.
            # 3. Compute delta E by checking only K4s/K5s that contain at least one flipped triple.
            
            return 0, 0 # Placeholder
        
        # Since vectorizing the delta E calculation is complex due to the combinatorial nature,
        # let's use a simpler "Local Search" that is faster than the original random search.
        # The original search was slow because it counted ALL violations every step.
        # We will maintain a list of current violations.
        
        # Precompute all K4s and K5s as lists of triple indices
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        red_k4s = []
        for quad in combinations(range(n), 4):
            idxs = [triple_to_idx[(quad[i], quad[j], quad[k])] 
                    for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
            red_k4s.append(idxs)
        
        blue_k5s = []
        for quint in combinations(range(n), 5):
            idxs = [triple_to_idx[(quint[i], quint[j], quint[k])] 
                    for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
            blue_k5s.append(idxs)
        
        # Count initial violations
        red_viol_counts = np.array([np.sum(colors[i] == 0 for i in k4) for k4 in red_k4s], dtype=np.int8)
        blue_viol_counts = np.array([np.sum(colors[i] == 1 for i in k5) for k5 in blue_k5s], dtype=np.int8)
        
        current_red_v = np.sum(red_viol_counts == 4)
        current_blue_v = np.sum(blue_viol_counts == 10)
        
        best_colors = colors.copy()
        best_score = current_red_v + current_blue_v
        
        temperature = 1.0
        cooling = 0.999
        
        rng = np.random.RandomState(seed_offset)
        
        while time.time() < max_time:
            # Pick a random triple to flip
            flip_idx = rng.randint(0, m)
            
            # Find all red K4s containing this triple
            # And all blue K5s containing this triple
            # To make this fast, we precompute "triple_to_k4s" and "triple_to_k5s"
            
            # For now, let's do a simple random flip and recalculate affected violations.
            # This is still O(N^4) or O(N^5) per step if not careful.
            
            # Optimization: Only flip if it reduces violations or with probability exp(-delta/T).
            
            # Let's try a different approach: 
            # Since we have time, let's just do a smarter random search that keeps the best state.
            # The original code's weakness was the slow `count_violations` function.
            # By maintaining `red_viol_counts` and `blue_viol_counts`, we can update them incrementally.
            
            # Update logic:
            # When we flip triple `flip_idx`:
            # 1. For every red K4 containing `flip_idx`:
            #    If color was 0 (red), it becomes 1. The count of reds in that K4 decreases by 1.
            #    If color was 1 (blue), it becomes 0. The count of reds in that K4 increases by 1.
            # 2. Similarly for blue K5s.
            
            # We need `triple_to_k4s` and `triple_to_k5s` mappings.
            
            # Build mappings
            if not hasattr(solve_sa, 'mappings'):
                solve_sa.mappings = {}
            if n not in solve_sa.mappings:
                t2k4 = [[] for _ in range(m)]
                t2k5 = [[] for _ in range(m)]
                for i, k4 in enumerate(red_k4s):
                    for t_idx in k4:
                        t2k4[t_idx].append(i)
                for i, k5 in enumerate(blue_k5s):
                    for t_idx in k5:
                        t2k5[t_idx].append(i)
                solve_sa.mappings[n] = (t2k4, t2k5, red_k4s, blue_k5s)
            else:
                t2k4, t2k5, red_k4s, blue_k5s = solve_sa.mappings[n]
            
            # Perform flip
            old_color = colors[flip_idx]
            new_color = 1 - old_color
            colors[flip_idx] = new_color
            
            # Update red violations
            delta_red_v = 0
            for k4_idx in t2k4[flip_idx]:
                old_count = red_viol_counts[k4_idx]
                if old_color == 0: # was red, now blue
                    new_count = old_count - 1
                else: # was blue, now red
                    new_count = old_count + 1
                
                # Update global violation count
                if old_count == 4:
                    current_red_v -= 1
                if new_count == 4:
                    current_red_v += 1
                
                red_viol_counts[k4_idx] = new_count
            
            # Update blue violations
            delta_blue_v = 0
            for k5_idx in t2k5[flip_idx]:
                old_count = blue_viol_counts[k5_idx]
                if old_color == 1: # was blue, now red
                    new_count = old_count - 1
                else: # was red, now blue
                    new_count = old_count + 1
                
                # Update global violation count
                if old_count == 10:
                    current_blue_v -= 1
                if new_count == 10:
                    current_blue_v += 1
                
                blue_viol_counts[k5_idx] = new_count
            
            # Acceptance criterion
            score = current_red_v + current_blue_v
            if score < best_score:
                best_score = score
                best_colors = colors.copy()
                if best_score == 0:
                    break
            else:
                # Metropolis acceptance
                if rng.rand() < np.exp(-(score - best_score) / temperature):
                    pass # Accept
                else:
                    # Reject: flip back
                    colors[flip_idx] = old_color
                    # Revert updates
                    for k4_idx in t2k4[flip_idx]:
                        old_count = red_viol_counts[k4_idx]
                        if old_color == 0:
                            new_count = old_count + 1
                        else:
                            new_count = old_count - 1
                        if old_count == 4:
                            current_red_v += 1
                        if new_count == 4:
                            current_red_v -= 1
                        red_viol_counts[k4_idx] = new_count
                    for k5_idx in t2k5[flip_idx]:
                        old_count = blue_viol_counts[k5_idx]
                        if old_color == 1:
                            new_count = old_count + 1
                        else:
                            new_count = old_count - 1
                        if old_count == 10:
                            current_blue_v += 1
                        if new_count == 10:
                            current_blue_v -= 1
                        blue_viol_counts[k5_idx] = new_count
            
            temperature *= cooling
        
        return best_colors

    # Strategy: Try increasing n, using SA with incremental violation updates.
    best_n = 0
    best_colors = None
    
    # We know n=9 is easy. Let's try to push higher.
    # The published construction is n=34.
    # We will try to find a valid coloring for n=10, 11, ... up to time limit.
    
    # To save time, we'll try n in a specific order, giving more time to larger n if smaller ones fail quickly?
    # No, larger n is harder. Let's try to find the max n possible.
    # We'll iterate n from 9 to 40.
    # If we find a valid one, we keep it.
    # We should allocate time based on n.
    
    # Actually, let's just run SA for each n and keep the best valid one.
    # To maximize n, we should try larger n first? Or smaller?
    # If we find n=34, we don't care about n=10.
    # So we should try n=40, 39, ... down to 9?
    # But n=40 is very hard. It might take all the time and fail.
    # Better to try a range and see what sticks.
    
    # Let's try n=15 first as a warmup, then jump to higher.
    # Or just loop n from 9 to 40, and if we find a valid one, continue to see if we can do better.
    
    # Given the time limit, let's try a few promising n values.
    # The seed program tried random local search which is slow.
    # Our SA with incremental updates should be much faster.
    
    target_ns = [10, 12, 15, 18, 20, 25, 30, 34, 35, 36, 37, 38, 39, 40]
    
    # Sort by difficulty? Or just try them?
    # Let's try in increasing order of n, but with a time budget for each.
    # If we find a valid n, we record it.
    
    # To be safe, let's ensure we return at least n=9.
    # We'll generate n=9 manually if nothing else works.
    
    # Time allocation: 10s per n? That's 140s for 14 values.
    # Let's give 5s per n.
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on remaining time
        remaining = deadline - time.time() - 5
        time_for_n = min(10, remaining / 2) # Give up to 10s, but not more than half remaining
        
        if time_for_n < 1:
            continue
            
        colors = solve_sa(n, time.time() + time_for_n, seed_offset=n*123)
        
        if check_validity(n, colors):
            best_n = n
            best_colors = colors
            # If we found a valid one, we can try to extend?
            # Or just continue to next n to see if we can do better.
            # Since we are iterating increasing n, we will naturally find the max.
            
    # Fallback
    if best_n == 0:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
