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

    def constraint_backtrack(target_n, structs, deadline, max_nodes=50000):
        """
        Constraint propagation backtracking solver.
        
        Key idea: Instead of blind local search, use constraint propagation
        to identify forced assignments. When a quad has 3 red triples, the 4th
        is FORCED to be blue. When a quint has 9 blue triples, the 10th is
        FORCED to be red. This dramatically reduces branching factor.
        
        Uses MRV (Minimum Remaining Values) heuristic for variable ordering.
        """
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']
        num_quads = structs['num_quads']
        num_quints = structs['num_quints']

        # State: -1=unassigned, 0=red, 1=blue
        colors = np.full(num_triples, -1, dtype=np.int8)
        
        # Track constraint satisfaction counts
        # For each quad: count of red triples (0-4), count of blue triples (0-4)
        quad_red = np.zeros(num_quads, dtype=np.int8)
        quad_blue = np.zeros(num_quads, dtype=np.int8)
        # For each quint: count of blue triples (0-10), count of red triples (0-10)
        quint_blue = np.zeros(num_quints, dtype=np.int8)
        quint_red = np.zeros(num_quints, dtype=np.int8)

        # Track which quads/quints are satisfied (violation)
        quad_violated = np.zeros(num_quads, dtype=np.bool_)
        quint_violated = np.zeros(num_quints, dtype=np.bool_)

        # Forced assignment queue
        forced_queue = []
        # Track assignments made to allow backtracking
        assignments = []  # list of (triple_idx, value, quad_red_delta, quad_blue_delta, quint_blue_delta, quint_red_delta)

        node_count = 0
        max_depth = 0
        solution = None

        def propagate():
            """Apply constraint propagation until fixpoint."""
            nonlocal forced_queue
            changed = True
            while changed:
                changed = False
                # Process forced assignments
                while forced_queue:
                    tidx, forced_val = forced_queue.pop()
                    if colors[tidx] != -1:
                        continue
                    if colors[tidx] == forced_val:
                        continue
                    # Check if this assignment is consistent
                    if colors[tidx] != -1 and colors[tidx] != forced_val:
                        return False  # Conflict
                    
                    colors[tidx] = forced_val
                    changed = True
                    assignments.append((tidx, forced_val))
                    
                    # Update quad counts
                    for (qi, pos) in triple_in_quads[tidx]:
                        if forced_val == 0:
                            quad_red[qi] += 1
                            if quad_red[qi] == 4:
                                quad_violated[qi] = True
                        else:
                            quad_blue[qi] += 1
                            if quad_blue[qi] == 4:
                                # All blue in quad - that's fine, no constraint violated
                                pass
                    
                    # Update quint counts
                    for (qi, pos) in triple_in_quints[tidx]:
                        if forced_val == 1:
                            quint_blue[qi] += 1
                            if quint_blue[qi] == 10:
                                quint_violated[qi] = True
                        else:
                            quint_red[qi] += 1
                
                # Check for new forced assignments
                # Rule 1: In a quad, if 3 are red, the 4th must be blue
                for qi in range(num_quads):
                    if quad_violated[qi]:
                        continue
                    qt = quad_triples_np[qi]
                    red_count = quad_red[qi]
                    blue_count = quad_blue[qi]
                    unassigned = 0
                    unassigned_idx = -1
                    
                    for pos in range(4):
                        tidx = int(qt[pos])
                        if colors[tidx] == -1:
                            unassigned += 1
                            unassigned_idx = tidx
                    
                    if unassigned == 1:
                        if red_count == 3:
                            # 4th must be blue
                            forced_queue.append((unassigned_idx, 1))
                            changed = True
                        elif blue_count == 3:
                            # 4th must be red (to avoid all-blue... wait, no constraint on all-blue quads)
                            # Actually there's no constraint against all-blue quads!
                            # Only red K4 is forbidden. So all-blue quads are fine.
                            pass

                # Rule 2: In a quint, if 9 are blue, the 10th must be red
                for qi in range(num_quints):
                    if quint_violated[qi]:
                        continue
                    qt = quint_triples_np[qi]
                    blue_count = quint_blue[qi]
                    red_count = quint_red[qi]
                    unassigned = 0
                    unassigned_idx = -1
                    
                    for pos in range(10):
                        tidx = int(qt[pos])
                        if colors[tidx] == -1:
                            unassigned += 1
                            unassigned_idx = tidx
                    
                    if unassigned == 1:
                        if blue_count == 9:
                            # 10th must be red
                            forced_queue.append((unassigned_idx, 0))
                            changed = True
                        elif red_count == 9:
                            # 10th must be blue... wait, no constraint against all-red quints
                            # Only blue K5 is forbidden. So all-red quints are fine.
                            pass

                # Check for conflicts
                if np.any(quad_violated):
                    return False
                if np.any(quint_violated):
                    return False

            return True

        def undo_assignments(count):
            """Undo the last 'count' assignments."""
            nonlocal colors, quad_red, quad_blue, quint_blue, quint_red, quad_violated, quint_violated
            for _ in range(count):
                tidx, val = assignments.pop()
                colors[tidx] = -1
                
                for (qi, pos) in triple_in_quads[tidx]:
                    if val == 0:
                        quad_red[qi] -= 1
                        if quad_red[qi] < 4:
                            quad_violated[qi] = False
                    else:
                        quad_blue[qi] -= 1
                
                for (qi, pos) in triple_in_quints[tidx]:
                    if val == 1:
                        quint_blue[qi] -= 1
                        if quint_blue[qi] < 10:
                            quint_violated[qi] = False
                    else:
                        quint_red[qi] -= 1

        def backtrack(depth):
            """Recursive backtracking with constraint propagation."""
            nonlocal node_count, max_depth, solution, colors, quad_red, quad_blue
            nonlocal quint_blue, quint_red, quad_violated, quint_violated, forced_queue
            
            if time.time() > deadline - 2:
                return None
            
            node_count += 1
            if node_count > max_nodes:
                return None
            
            max_depth = max(max_depth, depth)

            # Check if all assigned
            if np.all(colors != -1):
                # Verify no violations
                if not np.any(quad_violated) and not np.any(quint_violated):
                    return colors.copy()
                return None

            # Propagate constraints
            forced_queue = []
            if not propagate():
                return None  # Conflict

            # After propagation, check if all assigned
            if np.all(colors != -1):
                if not np.any(quad_violated) and not np.any(quint_violated):
                    return colors.copy()
                return None

            # Find unassigned triple with MRV heuristic
            # Prefer triples that appear in the most constrained structures
            unassigned_idx = np.where(colors == -1)[0]
            if len(unassigned_idx) == 0:
                return None

            # Score each unassigned triple by how constrained it is
            # (number of quads/quints where it's the last unassigned)
            best_tidx = -1
            best_score = -1
            for tidx in unassigned_idx:
                tidx = int(tidx)
                score = 0
                # Count quads where this is the only unassigned
                for (qi, pos) in triple_in_quads[tidx]:
                    qt = quad_triples_np[qi]
                    unassigned_in_quad = 0
                    for p in range(4):
                        if colors[int(qt[p])] == -1:
                            unassigned_in_quad += 1
                    if unassigned_in_quad == 1:
                        score += 10  # Highly constrained
                    elif unassigned_in_quad == 2:
                        score += 1
                
                # Count quints where this is the only unassigned
                for (qi, pos) in triple_in_quints[tidx]:
                    qt = quint_triples_np[qi]
                    unassigned_in_quint = 0
                    for p in range(10):
                        if colors[int(qt[p])] == -1:
                            unassigned_in_quint += 1
                    if unassigned_in_quint == 1:
                        score += 5
                    elif unassigned_in_quint == 2:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_tidx = tidx

            if best_tidx == -1:
                best_tidx = int(unassigned_idx[0])

            # Try both values
            for val in [0, 1]:
                if time.time() > deadline - 2:
                    return None
                
                # Assign
                colors[best_tidx] = val
                assignments.append((best_tidx, val))
                
                # Update counts
                for (qi, pos) in triple_in_quads[best_tidx]:
                    if val == 0:
                        quad_red[qi] += 1
                        if quad_red[qi] == 4:
                            quad_violated[qi] = True
                    else:
                        quad_blue[qi] += 1
                
                for (qi, pos) in triple_in_quints[best_tidx]:
                    if val == 1:
                        quint_blue[qi] += 1
                        if quint_blue[qi] == 10:
                            quint_violated[qi] = True
                    else:
                        quint_red[qi] += 1

                # Check for immediate conflict
                if np.any(quad_violated) or np.any(quint_violated):
                    # Undo
                    undo_assignments(1)
                    continue

                # Recurse
                result = backtrack(depth + 1)
                if result is not None:
                    return result

                # Undo
                undo_assignments(1)

            return None

        # Initial propagation
        forced_queue = []
        if not propagate():
            return None

        result = backtrack(0)
        return result

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

    def hybrid_search(target_n, structs, deadline):
        """
        Hybrid search: use constraint propagation to guide local search.
        
        Strategy:
        1. Start with a random/modular coloring
        2. Use constraint propagation to identify "almost forced" assignments
        3. Guide local search to prefer flips that satisfy propagation rules
        4. Periodically try full constraint backtracking on the current state
        """
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']
        num_quads = structs['num_quads']
        num_quints = structs['num_quints']
        triple_list = structs['triple_list']

        best_colors = None
        best_violations = float('inf')

        # Try multiple restarts with different strategies
        for restart in range(10):
            if time.time() > deadline - 5:
                break

            # Generate initial coloring
            color_arr = generate_initial_coloring(target_n, restart, num_triples, triple_list)
            
            # Phase 1: Quick local search to reduce violations
            color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=5000, deadline=deadline)
            
            if rk4 + bk5 < best_violations:
                best_violations = rk4 + bk5
                best_colors = color_arr.copy()
            
            if rk4 == 0 and bk5 == 0:
                return color_arr

            # Phase 2: Constraint-guided refinement
            # Identify triples that are "close to forced" and flip them preferentially
            quad_red_counts = np.sum(color_arr[quad_triples_np] == 0, axis=1)
            quint_blue_counts = np.sum(color_arr[quint_triples_np] == 1, axis=1)

            # Build priority list of triples to flip
            # Prioritize triples in quads with 3 red (flip to blue) or quints with 9 blue (flip to red)
            priority_triples = {}
            
            for qi in range(num_quads):
                if quad_red_counts[qi] == 3:
                    qt = quad_triples_np[qi]
                    for pos in range(4):
                        tidx = int(qt[pos])
                        if color_arr[tidx] == 0:
                            # This red triple in a 3-red quad: flipping it to blue helps
                            priority_triples[tidx] = priority_triples.get(tidx, 0) + 10
                elif quad_red_counts[qi] == 2:
                    qt = quad_triples_np[qi]
                    for pos in range(4):
                        tidx = int(qt[pos])
                        if color_arr[tidx] == 0:
                            priority_triples[tidx] = priority_triples.get(tidx, 0) + 2

            for qi in range(num_quints):
                if quint_blue_counts[qi] == 9:
                    qt = quint_triples_np[qi]
                    for pos in range(10):
                        tidx = int(qt[pos])
                        if color_arr[tidx] == 1:
                            # This blue triple in a 9-blue quint: flipping it to red helps
                            priority_triples[tidx] = priority_triples.get(tidx, 0) + 5
                elif quint_blue_counts[qi] == 8:
                    qt = quint_triples_np[qi]
                    for pos in range(10):
                        tidx = int(qt[pos])
                        if color_arr[tidx] == 1:
                            priority_triples[tidx] = priority_triples.get(tidx, 0) + 1

            # Phase 3: Guided local search using priority
            it = 0
            max_guided_iters = 20000
            while it < max_guided_iters:
                if time.time() > deadline - 3:
                    break
                if rk4 == 0 and bk5 == 0:
                    break

                # Choose triple: 70% from priority list, 30% random
                if priority_triples and random.random() < 0.7:
                    # Weighted random from priority list
                    items = list(priority_triples.items())
                    total_weight = sum(w for _, w in items)
                    if total_weight > 0:
                        r = random.random() * total_weight
                        cumulative = 0
                        tidx = items[0][0]
                        for t, w in items:
                            cumulative += w
                            if r <= cumulative:
                                tidx = t
                                break
                    else:
                        tidx = random.randint(0, num_triples - 1)
                else:
                    tidx = random.randint(0, num_triples - 1)

                old_color = int(color_arr[tidx])
                
                # Compute delta
                delta_rk4 = 0
                for (qi, pos) in triple_in_quads[tidx]:
                    qt = quad_triples_np[qi]
                    others = [qt[p] for p in range(4) if p != pos]
                    others_red = all(int(color_arr[t]) == 0 for t in others)
                    if old_color == 0 and others_red:
                        delta_rk4 -= 1
                    elif old_color == 1 and others_red:
                        delta_rk4 += 1

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

                # Accept if improving, or with small probability if worsening
                if delta_total <= 0:
                    color_arr[tidx] = 1 - old_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5
                elif random.random() < 0.1:
                    color_arr[tidx] = 1 - old_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5

                it += 1

            # Update best
            if rk4 + bk5 < best_violations:
                best_violations = rk4 + bk5
                best_colors = color_arr.copy()

            # Phase 4: Try constraint backtracking from current state
            if time.time() < deadline - 10 and best_violations <= 10:
                # Seed the backtracker with current assignment
                bt_result = constraint_backtrack(target_n, structs, deadline, max_nodes=20000)
                if bt_result is not None:
                    return bt_result

        return best_colors

    # Main loop: try increasing n with hybrid search
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False
        result_colors = None

        # Strategy 1: Hybrid constraint-guided search
        result_colors = hybrid_search(target_n, structs, deadline)
        if result_colors is not None:
            rk4, bk5, _, _ = compute_violations(result_colors, structs)
            if rk4 == 0 and bk5 == 0:
                found = True

        # Strategy 2: If hybrid failed, try pure constraint backtracking
        if not found and time.time() < deadline - 5:
            bt_result = constraint_backtrack(target_n, structs, deadline, max_nodes=30000)
            if bt_result is not None:
                result_colors = bt_result
                found = True

        # Strategy 3: If still failed, try more local search restarts
        if not found:
            max_restarts = 20
            for restart in range(max_restarts):
                if time.time() > deadline - 3:
                    break

                color_arr = generate_initial_coloring(target_n, restart, num_triples, structs['triple_list'])
                color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=50000, deadline=deadline)

                if rk4 == 0 and bk5 == 0:
                    result_colors = color_arr
                    found = True
                    break

        if found:
            colors_list = result_colors.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                break
        else:
            if target_n >= 25:
                break

    return {'n': best_n, 'colors': best_colors}
