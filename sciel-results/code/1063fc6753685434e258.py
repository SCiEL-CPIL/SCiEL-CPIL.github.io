"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute all structures needed for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, list of quad indices it belongs to
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices it belongs to
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints)

    def solve_local_search(target_n, init_colors, max_steps=50000):
        """
        Local search with incremental violation counting.
        """
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = build_structures(target_n)
        
        colors = np.array(init_colors, dtype=np.int8)
        
        # Initial violation counts
        red_v = 0
        for qt in quad_triples_idx:
            if all(colors[i] == 0 for i in qt):
                red_v += 1
                
        blue_v = 0
        for qt in quint_triples_idx:
            if all(colors[i] == 1 for i in qt):
                blue_v += 1
        
        best_viol = red_v + blue_v
        best_colors = colors.copy()
        
        # Precompute "redness" and "blueness" counts for each quad/quint to speed up checks?
        # Or just check on the fly. With incremental updates, we only need to check affected ones.
        
        # To speed up "all red" check for a quad, we can maintain a count of red edges in each quad.
        red_count_quad = np.zeros(num_quads, dtype=np.int8)
        for qi, qt in enumerate(quad_triples_idx):
            red_count_quad[qi] = sum(1 for i in qt if colors[i] == 0)
            
        blue_count_quint = np.zeros(num_quints, dtype=np.int8)
        for qi, qt in enumerate(quint_triples_idx):
            blue_count_quint[qi] = sum(1 for i in qt if colors[i] == 1)

        # Re-initialize violation counts based on these counts
        red_v = np.sum(red_count_quad == 4)
        blue_v = np.sum(blue_count_quint == 10)
        
        # List of currently violating quads/quints for quick selection
        violating_quads = [i for i in range(num_quads) if red_count_quad[i] == 4]
        violating_quints = [i for i in range(num_quints) if blue_count_quint[i] == 10]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if red_v + blue_v == 0:
                return colors.tolist()
            
            if red_v + blue_v < best_viol:
                best_viol = red_v + blue_v
                best_colors = colors.copy()
                
            # Choose a move: flip a triple in a violating quad or quint
            # Probability proportional to violation counts
            total_viol = red_v + blue_v
            if total_viol == 0:
                break
                
            if random.random() < red_v / total_viol:
                # Pick a random violating quad
                if not violating_quads:
                    continue
                qi = random.choice(violating_quads)
                qt = quad_triples_idx[qi]
                tri_idx = random.choice(qt)
            else:
                # Pick a random violating quint
                if not violating_quints:
                    continue
                qi = random.choice(violating_quints)
                qt = quint_triples_idx[qi]
                tri_idx = random.choice(qt)
            
            # Determine new color
            new_color = 1 - colors[tri_idx]
            
            # Update counts for affected quads
            affected_quads = triple_in_quads[tri_idx]
            for aqi in affected_quads:
                qt = quad_triples_idx[aqi]
                if colors[tri_idx] == 0:
                    # Was red, now blue -> red count decreases
                    red_count_quad[aqi] -= 1
                    if red_count_quad[aqi] == 4:
                        red_v += 1
                        violating_quads.append(aqi)
                    elif red_count_quad[aqi] == 3:
                        red_v -= 1
                        # Remove from violating_quads? Efficient removal is hard.
                        # Instead, we can just let the list have stale entries and filter when picking.
                else:
                    # Was blue, now red -> red count increases
                    red_count_quad[aqi] += 1
                    if red_count_quad[aqi] == 4:
                        red_v += 1
                        violating_quads.append(aqi)
                    elif red_count_quad[aqi] == 3:
                        # It was 3, now 4? No, 3->4 is handled above.
                        # If it was 4 and now 3, we handled it in the first branch.
                        pass

            # Update counts for affected quints
            affected_quints = triple_in_quints[tri_idx]
            for aqi in affected_quints:
                qt = quint_triples_idx[aqi]
                if colors[tri_idx] == 1:
                    # Was blue, now red -> blue count decreases
                    blue_count_quint[aqi] -= 1
                    if blue_count_quint[aqi] == 10:
                        blue_v += 1
                        violating_quints.append(aqi)
                    elif blue_count_quint[aqi] == 9:
                        blue_v -= 1
                else:
                    # Was red, now blue -> blue count increases
                    blue_count_quint[aqi] += 1
                    if blue_count_quint[aqi] == 10:
                        blue_v += 1
                        violating_quints.append(aqi)
                    elif blue_count_quint[aqi] == 9:
                        pass

            # Apply the flip
            colors[tri_idx] = new_color
            
            # Periodically clean up violating lists to prevent them from growing too large with stale entries
            if step % 1000 == 0:
                violating_quads = [i for i in violating_quads if red_count_quad[i] == 4]
                violating_quints = [i for i in violating_quints if blue_count_quint[i] == 10]
                # Also add any that might have been missed? No, we add on update.
                # But we might have missed some if they became violating due to other flips?
                # The incremental update handles this.
                # However, if a quad becomes violating, we add it. If it stops, we don't remove immediately.
                # The cleanup removes stale ones.
                
        return None

    # Try to solve for increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        solved = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            if attempt == 0:
                # Modular initialization
                init = mod_color(target_n, 3)
            else:
                # Random initialization
                num_triples = target_n * (target_n - 1) * (target_n - 2) // 6
                init = [random.randint(0, 1) for _ in range(num_triples)]
            
            result = solve_local_search(target_n, init, max_steps=20000)
            if result is not None:
                best_n = target_n
                best_colors = result
                solved = True
                break
        
        if solved:
            continue
        else:
            # If we failed for this n, we might still try larger n with different inits?
            # Usually if we can't find n, we can't find n+1 easily. But let's try.
            pass

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
