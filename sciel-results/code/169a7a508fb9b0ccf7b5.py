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
        """Precompute all structural mappings for a given n."""
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

        # Inverse mappings: for each triple, which quads/quints contain it
        # Store (structure_idx, position_in_structure)
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

        return {
            'triple_list': triple_list,
            'num_triples': num_triples,
            'quad_triples_np': quad_triples_np,
            'quint_triples_np': quint_triples_np,
            'num_quads': num_quads,
            'num_quints': num_quints,
            'triple_in_quads': triple_in_quads,
            'triple_in_quints': triple_in_quints,
        }

    def compute_metrics(color_arr, structs):
        """Compute violation counts and near-violation counts."""
        quad_colors = structs['quad_triples_np']
        quad_red_counts = np.sum(color_arr[quad_colors] == 0, axis=1)
        rk4 = int(np.sum(quad_red_counts == 4))
        near_rk4 = int(np.sum(quad_red_counts == 3))

        quint_colors = structs['quint_triples_np']
        quint_blue_counts = np.sum(color_arr[quint_colors] == 1, axis=1)
        bk5 = int(np.sum(quint_blue_counts == 10))
        near_bk5 = int(np.sum(quint_blue_counts == 9))

        return rk4, bk5, near_rk4, near_bk5, quad_red_counts, quint_blue_counts

    def calculate_delta_energy(tidx, old_color, color_arr, structs, quad_red_counts, quint_blue_counts):
        """
        Calculate the change in energy if we flip tidx.
        Energy = 100*Violations + 10*NearViolations
        """
        delta_energy = 0
        
        # Process Quads
        for (qi, pos) in structs['triple_in_quads'][tidx]:
            # Get the red count of the quad excluding this triple
            # quad_red_counts[qi] includes tidx if it's red.
            current_red_count = quad_red_counts[qi]
            
            if old_color == 0: # tidx is Red
                # Others have current_red_count - 1 reds
                others_red = current_red_count - 1
                # New count will be others_red (since tidx becomes Blue)
                new_red_count = others_red
                
                # Old contribution:
                if current_red_count == 4:
                    delta_energy -= 100 # Remove Violation
                elif current_red_count == 3:
                    delta_energy -= 10  # Remove Near Violation
                
                # New contribution:
                if new_red_count == 4:
                    delta_energy += 100 # Add Violation
                elif new_red_count == 3:
                    delta_energy += 10  # Add Near Violation
                    
            else: # tidx is Blue
                # Others have current_red_count reds
                others_red = current_red_count
                # New count will be others_red + 1 (since tidx becomes Red)
                new_red_count = others_red + 1
                
                # Old contribution:
                if current_red_count == 4:
                    delta_energy -= 100
                elif current_red_count == 3:
                    delta_energy -= 10
                
                # New contribution:
                if new_red_count == 4:
                    delta_energy += 100
                elif new_red_count == 3:
                    delta_energy += 10

        # Process Quints
        for (qi, pos) in structs['triple_in_quints'][tidx]:
            current_blue_count = quint_blue_counts[qi]
            
            if old_color == 1: # tidx is Blue
                # Others have current_blue_count - 1 blues
                others_blue = current_blue_count - 1
                # New count will be others_blue (since tidx becomes Red)
                new_blue_count = others_blue
                
                # Old contribution:
                if current_blue_count == 10:
                    delta_energy -= 100
                elif current_blue_count == 9:
                    delta_energy -= 10
                
                # New contribution:
                if new_blue_count == 10:
                    delta_energy += 100
                elif new_blue_count == 9:
                    delta_energy += 10

            else: # tidx is Red
                # Others have current_blue_count blues
                others_blue = current_blue_count
                # New count will be others_blue + 1 (since tidx becomes Blue)
                new_blue_count = others_blue + 1
                
                # Old contribution:
                if current_blue_count == 10:
                    delta_energy -= 100
                elif current_blue_count == 9:
                    delta_energy -= 10
                
                # New contribution:
                if new_blue_count == 10:
                    delta_energy += 100
                elif new_blue_count == 9:
                    delta_energy += 10

        return delta_energy

    def local_search_penalized(color_arr, structs, max_iters, deadline, temp_start=5.0, temp_end=0.1):
        """
        Local search using penalized energy (violations + near-violations).
        Returns (colors, rk4, bk5).
        """
        num_triples = structs['num_triples']
        
        rk4, bk5, near_rk4, near_bk5, quad_red_counts, quint_blue_counts = compute_metrics(color_arr, structs)
        
        if rk4 == 0 and bk5 == 0:
            return color_arr, rk4, bk5

        current_energy = 100 * rk4 + 10 * near_rk4 + 100 * bk5 + 10 * near_bk5
        
        it = 0
        while it < max_iters:
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                break

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip
            # Bias towards violating structures if possible, else random
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.4:
                violating_quads = np.where(quad_red_counts == 4)[0]
                if len(violating_quads) > 0:
                    qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                    qt = structs['quad_triples_np'][qi]
                    tidx = int(qt[random.randint(0, 3)])
            elif bk5 > 0 and r < 0.7:
                violating_quints = np.where(quint_blue_counts == 10)[0]
                if len(violating_quints) > 0:
                    qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                    qt = structs['quint_triples_np'][qi]
                    tidx = int(qt[random.randint(0, 9)])
            
            if tidx == -1:
                # If no direct violations or random choice, pick from near-violations or random
                if near_rk4 > 0 and r < 0.8:
                    near_viol_quads = np.where(quad_red_counts == 3)[0]
                    if len(near_viol_quads) > 0:
                        qi = near_viol_quads[random.randint(0, len(near_viol_quads)-1)]
                        qt = structs['quad_triples_np'][qi]
                        # Pick the blue one in this quad? Or random?
                        # If we pick a blue one and flip to red, we create a violation. Bad.
                        # We should pick a red one and flip to blue? No, that reduces near-violation.
                        # Actually, flipping a red one in a 3-red quad makes it 2-red.
                        # Flipping a blue one makes it 4-red (Violation).
                        # So we should prefer flipping the RED ones in near-violating quads.
                        # Let's just pick random for now to keep it simple, or pick the one that minimizes delta.
                        # For speed, pick random.
                        tidx = int(qt[random.randint(0, 3)])
                        
                if near_bk5 > 0 and r < 0.9:
                    near_viol_quints = np.where(quint_blue_counts == 9)[0]
                    if len(near_viol_quints) > 0:
                        qi = near_viol_quints[random.randint(0, len(near_viol_quints)-1)]
                        qt = structs['quint_triples_np'][qi]
                        tidx = int(qt[random.randint(0, 9)])

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = int(color_arr[tidx])
            
            delta_energy = calculate_delta_energy(tidx, old_color, color_arr, structs, quad_red_counts, quint_blue_counts)

            # Acceptance
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(temp, 1e-10)):
                color_arr[tidx] = 1 - old_color
                
                # Update counts incrementally
                # This is tricky to do perfectly without recalculating, so we recalc metrics every few steps or use delta.
                # For simplicity and correctness in this edit, we will recalculate metrics if we accept.
                # This is slower but safer. Optimization: only recalc if we accept?
                # Actually, we need quad_red_counts for the NEXT iteration's delta calc.
                # So we MUST update quad_red_counts and quint_blue_counts.
                
                # Update quad_red_counts
                for (qi, pos) in structs['triple_in_quads'][tidx]:
                    if old_color == 0: # Was Red, now Blue -> Red count decreases
                        quad_red_counts[qi] -= 1
                    else: # Was Blue, now Red -> Red count increases
                        quad_red_counts[qi] += 1
                        
                # Update quint_blue_counts
                for (qi, pos) in structs['triple_in_quints'][tidx]:
                    if old_color == 1: # Was Blue, now Red -> Blue count decreases
                        quint_blue_counts[qi] -= 1
                    else: # Was Red, now Blue -> Blue count increases
                        quint_blue_counts[qi] += 1
                        
                # Update global counts
                # We need to update rk4, bk5, near_rk4, near_bk5
                # This requires knowing the old and new states of the affected structures.
                # We can just recompute these 4 integers from the arrays.
                rk4 = int(np.sum(quad_red_counts == 4))
                near_rk4 = int(np.sum(quad_red_counts == 3))
                bk5 = int(np.sum(quint_blue_counts == 10))
                near_bk5 = int(np.sum(quint_blue_counts == 9))
                
                current_energy = 100 * rk4 + 10 * near_rk4 + 100 * bk5 + 10 * near_bk5

            it += 1

        return color_arr, rk4, bk5

    def generate_initial_coloring(target_n, restart_idx, num_triples, triple_list):
        """Generate diverse initial colorings."""
        if restart_idx == 0:
            return np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 1:
            return np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 2:
            return np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 3:
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif restart_idx == 4:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 0
            return arr
        elif restart_idx == 5:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 1
            return arr
        else:
            base_mod = 3 + (restart_idx % 3)
            base = np.array([0 if (a+b+c) % base_mod == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            flip_mask = np.random.random(num_triples) < 0.15
            base[flip_mask] = 1 - base[flip_mask]
            return base

    def iterated_local_search(target_n, structs, deadline, max_restarts=100):
        """
        Iterated Local Search using Penalized Energy.
        """
        num_triples = structs['num_triples']
        triple_list = structs['triple_list']
        
        best_colors = None
        best_violations = float('inf')

        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            color_arr = generate_initial_coloring(target_n, restart, num_triples, triple_list)
            
            # Run local search
            color_arr, rk4, bk5 = local_search_penalized(color_arr, structs, max_iters=5000, deadline=deadline)
            
            violations = rk4 + bk5
            if violations < best_violations:
                best_violations = violations
                best_colors = color_arr.copy()
                
            if violations == 0:
                return color_arr

            # Perturb if not solved
            if time.time() > deadline - 10:
                break
            
            # Strong perturbation: flip 5-10% of triples
            perturb_rate = 0.05 + 0.05 * random.random()
            mask = np.random.random(num_triples) < perturb_rate
            color_arr[mask] = 1 - color_arr[mask]

        if best_colors is not None:
            rk4, bk5, _, _, _, _ = compute_metrics(best_colors, structs)
            if rk4 == 0 and bk5 == 0:
                return best_colors
        return None

    # Main loop: try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False

        # Strategy: Iterated Local Search with Penalized Energy
        ils_result = iterated_local_search(target_n, structs, deadline, max_restarts=50)
        
        if ils_result is not None:
            colors_list = ils_result.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
                found = True
            else:
                # Should not happen if ils_result returned valid, but check just in case
                break
        
        if not found:
             if target_n >= 28:
                break

    return {'n': best_n, 'colors': best_colors}
