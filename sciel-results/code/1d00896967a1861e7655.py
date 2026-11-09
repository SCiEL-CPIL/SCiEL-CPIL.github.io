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

    def compute_violations(color_arr, structs):
        """Compute violation counts using vectorized numpy."""
        quad_colors = structs['quad_triples_np']
        quad_red_counts = np.sum(color_arr[quad_colors] == 0, axis=1)
        rk4 = int(np.sum(quad_red_counts == 4))

        quint_colors = structs['quint_triples_np']
        quint_blue_counts = np.sum(color_arr[quint_colors] == 1, axis=1)
        bk5 = int(np.sum(quint_blue_counts == 10))

        return rk4, bk5, quad_red_counts, quint_blue_counts

    def local_search(color_arr, structs, max_iters, deadline, temp_start=2.0, temp_end=0.01):
        """Run simulated annealing local search, return (colors, rk4, bk5)."""
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']

        rk4, bk5, quad_red_counts, quint_blue_counts = compute_violations(color_arr, structs)

        if rk4 == 0 and bk5 == 0:
            return color_arr, rk4, bk5

        it = 0
        while it < max_iters:
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                break

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip - bias toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.5:
                violating_quads = np.where(quad_red_counts == 4)[0]
                if len(violating_quads) > 0:
                    qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                    qt = quad_triples_np[qi]
                    tidx = int(qt[random.randint(0, 3)])
            elif bk5 > 0 and r < 0.85:
                violating_quints = np.where(quint_blue_counts == 10)[0]
                if len(violating_quints) > 0:
                    qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                    qt = quint_triples_np[qi]
                    tidx = int(qt[random.randint(0, 9)])

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = int(color_arr[tidx])

            # Compute delta_rk4 incrementally
            delta_rk4 = 0
            for (qi, pos) in triple_in_quads[tidx]:
                qt = quad_triples_np[qi]
                others = [qt[p] for p in range(4) if p != pos]
                others_red = all(int(color_arr[t]) == 0 for t in others)
                if old_color == 0 and others_red:
                    delta_rk4 -= 1
                elif old_color == 1 and others_red:
                    delta_rk4 += 1

            # Compute delta_bk5 incrementally
            delta_bk5 = 0
            for (qi, pos) in triple_in_quints[tidx]:
                qt = quint_triples_np[qi]
                others = [qt[p] for p in range(10) if p != pos]
                others_blue = all(int(color_arr[t]) == 1 for t in others)
                if old_color == 1 and others_blue:
                    delta_bk5 -= 1
                elif old_color == 0 and others_blue:
                    delta_bk5 += 1

            delta_total = delta_rk4 + delta_bk5

            # Acceptance criterion
            if delta_total <= 0:
                color_arr[tidx] = 1 - old_color
                rk4 += delta_rk4
                bk5 += delta_bk5
            else:
                if random.random() < np.exp(-delta_total / max(temp, 1e-10)):
                    color_arr[tidx] = 1 - old_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5

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

    def spectral_local_search(color_arr, structs, max_iters, deadline):
        """
        Spectral embedding approach:
        1. Build hypergraph adjacency structure
        2. Project coloring into low-dim eigenspace
        3. Use projected gradient descent in continuous space
        4. Round to binary and refine with local search
        """
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']

        rk4, bk5, quad_red_counts, quint_blue_counts = compute_violations(color_arr, structs)
        if rk4 == 0 and bk5 == 0:
            return color_arr, rk4, bk5

        # Build adjacency: two triples are connected if they share 2 vertices
        # For efficiency, we'll use a simplified spectral approach
        # Create a weighted graph where edges represent conflict potential
        
        # Compute conflict potential for each triple
        # A triple has high conflict if it's in many violating structures
        conflict_scores = np.zeros(num_triples, dtype=np.float64)
        
        for tidx in range(num_triples):
            # Count how many violating quads this triple is in
            for (qi, pos) in triple_in_quads[tidx]:
                if quad_red_counts[qi] == 4:
                    conflict_scores[tidx] += 1
            # Count how many violating quints this triple is in
            for (qi, pos) in triple_in_quints[tidx]:
                if quint_blue_counts[qi] == 10:
                    conflict_scores[tidx] += 1
        
        # Normalize conflict scores
        max_conflict = np.max(conflict_scores)
        if max_conflict > 0:
            conflict_scores = conflict_scores / max_conflict
        
        # Spectral embedding: use conflict scores as a feature
        # Project into a 2D space using PCA-like approach
        # For simplicity, use the conflict score and a random orthogonal component
        
        # Create continuous representation
        x_continuous = np.zeros((num_triples, 2), dtype=np.float64)
        x_continuous[:, 0] = (color_arr.astype(np.float64) * 2 - 1)  # Map 0->-1, 1->+1
        x_continuous[:, 1] = conflict_scores
        
        # Projected gradient descent in the 2D space
        # Minimize a penalty function based on violations
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                break
            
            # Compute gradient of penalty w.r.t. continuous variables
            # Penalty = sum over quads of (sum of red indicators - 3)^2 * indicator(all red)
            # + sum over quints of (sum of blue indicators - 9)^2 * indicator(all blue)
            
            grad = np.zeros(num_triples, dtype=np.float64)
            
            # Gradient contribution from red K4 violations
            for qi in range(len(quad_red_counts)):
                if quad_red_counts[qi] == 4:
                    # All four triples are red, this is a violation
                    # Gradient pushes them away from red (towards blue)
                    for pos in range(4):
                        tidx = int(quad_triples_np[qi, pos])
                        grad[tidx] -= 1.0  # Push towards blue
            
            # Gradient contribution from blue K5 violations
            for qi in range(len(quint_blue_counts)):
                if quint_blue_counts[qi] == 10:
                    # All ten triples are blue, this is a violation
                    # Gradient pushes them away from blue (towards red)
                    for pos in range(10):
                        tidx = int(quint_triples_np[qi, pos])
                        grad[tidx] += 1.0  # Push towards red
            
            # Update continuous variables
            learning_rate = 0.1
            x_continuous[:, 0] -= learning_rate * grad
            x_continuous[:, 0] = np.clip(x_continuous[:, 0], -1.5, 1.5)
            
            # Round to binary
            new_color_arr = (x_continuous[:, 0] > 0).astype(np.int8)
            
            # Check if we improved
            new_rk4, new_bk5, new_quad_red_counts, new_quint_blue_counts = compute_violations(new_color_arr, structs)
            new_violations = new_rk4 + new_bk5
            old_violations = rk4 + bk5
            
            if new_violations < old_violations:
                color_arr = new_color_arr
                rk4, bk5 = new_rk4, new_bk5
                quad_red_counts, quint_blue_counts = new_quad_red_counts, new_quint_blue_counts
            
            it += 1
        
        # Final refinement with standard local search
        color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=10000, deadline=deadline)
        
        return color_arr, rk4, bk5

    def hybrid_search(target_n, structs, deadline):
        """
        Combine multiple search strategies:
        1. Standard local search with diverse restarts
        2. Spectral embedding approach
        3. Iterated local search with perturbation
        """
        num_triples = structs['num_triples']
        triple_list = structs['triple_list']
        
        best_result = None
        best_violations = float('inf')
        
        # Strategy 1: Standard local search with diverse restarts
        for restart in range(30):
            if time.time() > deadline - 3:
                break
            
            color_arr = generate_initial_coloring(target_n, restart, num_triples, triple_list)
            color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=20000, deadline=deadline)
            
            violations = rk4 + bk5
            if violations < best_violations:
                best_violations = violations
                best_result = color_arr.copy()
            
            if rk4 == 0 and bk5 == 0:
                return color_arr
        
        # Strategy 2: Spectral embedding from best found so far
        if best_result is not None:
            if time.time() < deadline - 5:
                spectral_result = spectral_local_search(best_result.copy(), structs, max_iters=5000, deadline=deadline)
                rk4, bk5 = spectral_result[1], spectral_result[2]
                violations = rk4 + bk5
                if violations < best_violations:
                    best_violations = violations
                    best_result = spectral_result[0].copy()
                if rk4 == 0 and bk5 == 0:
                    return spectral_result[0]
        
        # Strategy 3: Iterated local search with perturbation from best
        if best_result is not None:
            for iteration in range(20):
                if time.time() > deadline - 3:
                    break
                
                # Perturb the best solution
                perturbed = best_result.copy()
                num_flips = random.randint(5, 20)
                flip_indices = random.sample(range(num_triples), min(num_flips, num_triples))
                perturbed[flip_indices] = 1 - perturbed[flip_indices]
                
                # Apply local search
                result = local_search(perturbed, structs, max_iters=15000, deadline=deadline)
                rk4, bk5 = result[1], result[2]
                violations = rk4 + bk5
                
                if violations < best_violations:
                    best_violations = violations
                    best_result = result[0].copy()
                
                if rk4 == 0 and bk5 == 0:
                    return result[0]
        
        return best_result

    # Main loop: try increasing n with hybrid search
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False
        result = hybrid_search(target_n, structs, deadline)
        
        if result is not None:
            rk4, bk5, _, _ = compute_violations(result, structs)
            if rk4 == 0 and bk5 == 0:
                found = True
                colors_list = result.tolist()
            else:
                # Try a few more targeted restarts
                for restart in range(50):
                    if time.time() > deadline - 3:
                        break
                    color_arr = generate_initial_coloring(target_n, restart + 100, num_triples, structs['triple_list'])
                    color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=30000, deadline=deadline)
                    if rk4 == 0 and bk5 == 0:
                        found = True
                        colors_list = color_arr.tolist()
                        break

        if found:
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                break
        else:
            if target_n >= 28:
                break

    return {'n': best_n, 'colors': best_colors}
