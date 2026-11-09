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

    def run_local_search(n, init_colors, time_limit):
        """
        Run local search on a coloring for n vertices.
        Returns (colors, success)
        """
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        # Precompute quads and quints
        all_quads = list(combinations(range(n), 4))
        num_quads = len(all_quads)
        quad_triples = []
        for a, b, c, d in all_quads:
            quad_triples.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])

        all_quints = list(combinations(range(n), 5))
        num_quints = len(all_quints)
        quint_triples = []
        for q5 in all_quints:
            quint_triples.append([triple_to_idx[t] for t in combinations(q5, 3)])

        # Precompute incidence: triple -> list of quad indices, triple -> list of quint indices
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        # Initialize colors
        color_arr = np.array(init_colors, dtype=np.int8)
        
        # Initialize counts
        # quad_red_counts[q] = number of red triples in quad q
        quad_red_counts = np.zeros(num_quads, dtype=np.int8)
        for qi, qt in enumerate(quad_triples):
            for tidx in qt:
                if color_arr[tidx] == 0:
                    quad_red_counts[qi] += 1
        
        # quint_blue_counts[f] = number of blue triples in quint f
        quint_blue_counts = np.zeros(num_quints, dtype=np.int8)
        for qi, qt in enumerate(quint_triples):
            for tidx in qt:
                if color_arr[tidx] == 1:
                    quint_blue_counts[qi] += 1

        # Identify violations
        violating_quads = set(np.where(quad_red_counts == 4)[0].tolist())
        violating_quints = set(np.where(quint_blue_counts == 10)[0].tolist())
        
        total_violations = len(violating_quads) + len(violating_quints)
        
        if total_violations == 0:
            return color_arr.tolist(), True

        max_iters = 500000
        temp_start = 2.0
        temp_end = 0.01
        it = 0
        best_violations = total_violations
        best_colors_copy = color_arr.copy()
        
        while it < max_iters:
            if time.time() > time_limit:
                break
            if total_violations == 0:
                return color_arr.tolist(), True
            
            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)
            
            # Select a triple to flip
            tidx = -1
            if total_violations > 0:
                # Prioritize flipping a triple involved in a violation
                if random.random() < 0.8:
                    # Pick from violating structures
                    if violating_quads and (not violating_quints or random.random() < 0.5):
                        qi = random.choice(list(violating_quads))
                        qt = quad_triples[qi]
                        tidx = random.choice(qt)
                    elif violating_quints:
                        qi = random.choice(list(violating_quints))
                        qt = quint_triples[qi]
                        tidx = random.choice(qt)
            
            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)
            
            old_color = color_arr[tidx]
            new_color = 1 - old_color
            
            # Calculate delta in violations
            # We need to check if the flip changes the violation status of any quad/quint containing tidx
            
            delta_violations = 0
            
            # Process quads
            for qi in triple_in_quads[tidx]:
                qt = quad_triples[qi]
                # Find position of tidx in qt
                # Since qt is small (4), linear search is fine
                pos = -1
                for p in range(4):
                    if qt[p] == tidx:
                        pos = p
                        break
                
                old_count = quad_red_counts[qi]
                # Check if this quad was violating before
                was_violating = (old_count == 4)
                
                # Update count
                if old_color == 0:
                    new_count = old_count - 1
                else:
                    new_count = old_count + 1
                
                # Check if this quad is violating after
                is_violating = (new_count == 4)
                
                if was_violating and not is_violating:
                    delta_violations -= 1
                elif not was_violating and is_violating:
                    delta_violations += 1
                
                quad_red_counts[qi] = new_count
            
            # Process quints
            for qi in triple_in_quints[tidx]:
                qt = quint_triples[qi]
                # Find position of tidx in qt
                pos = -1
                for p in range(10):
                    if qt[p] == tidx:
                        pos = p
                        break
                
                old_count = quint_blue_counts[qi]
                was_violating = (old_count == 10)
                
                # Update count
                if old_color == 1:
                    new_count = old_count - 1
                else:
                    new_count = old_count + 1
                
                is_violating = (new_count == 10)
                
                if was_violating and not is_violating:
                    delta_violations -= 1
                elif not was_violating and is_violating:
                    delta_violations += 1
                
                quint_blue_counts[qi] = new_count

            # Acceptance criterion (Simulated Annealing)
            if delta_violations <= 0 or random.random() < np.exp(-delta_violations / max(temp, 1e-10)):
                # Accept move
                color_arr[tidx] = new_color
                total_violations += delta_violations
                
                # Update violation sets
                # We don't strictly need to maintain the sets for correctness of the count,
                # but it helps for selection. However, updating sets incrementally is tricky.
                # For simplicity and speed, we can just re-evaluate the sets occasionally or
                # just rely on the count for acceptance and random selection if sets are stale.
                # But we used the sets for selection! So we must update them.
                
                # To update sets efficiently, we need to know which quads/quints changed status.
                # We can track this during the loop above.
                
                # Let's refine: we will track changed_quads and changed_quints
                # Actually, the current implementation of selection uses `violating_quads` set.
                # If we don't update it, we might pick non-violating quads or miss new ones.
                # For a robust implementation, we should update the sets.
                
                # Re-computing sets is O(num_quads + num_quints) which is ~300k.
                # Doing this every step is too slow.
                # We should update the sets incrementally.
                
                # Let's modify the loop to collect changes.
                # This requires a refactor of the inner loop.
                
                # For now, to keep it simple and fast, we will occasionally recompute the sets.
                # Or better: we can just pick a random triple if the sets are too stale.
                # But let's try to update them.
                
                # Actually, let's just recompute the violating sets every 100 steps or when total_violations changes significantly?
                # No, that's risky.
                
                # Alternative: Just pick a random triple always?
                # That is slower to converge.
                
                # Let's stick to the incremental count for acceptance, and for selection,
                # we will use a hybrid: if we have a list of "recently changed" structures, use that.
                # Otherwise, random.
                
                # Given the time constraint, let's simplify:
                # We will just pick a random triple. The acceptance criterion (SA) will guide the search.
                # The "bias" towards violations is helpful but not strictly required if SA is tuned well.
                # However, biasing is much faster.
                
                # Let's try a different approach for selection:
                # Maintain a list of "hot" triples. A triple is hot if it is in a violating quad or quint.
                # We can approximate this by just picking from the violating sets if they are not empty.
                # To keep the sets updated, we will just recompute them every N steps.
                
                if it % 50 == 0:
                    violating_quads = set(np.where(quad_red_counts == 4)[0].tolist())
                    violating_quints = set(np.where(quint_blue_counts == 10)[0].tolist())
                    total_violations = len(violating_quads) + len(violating_quints)
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors_copy = color_arr.copy()
            
            it += 1

        # After loop, check if best found is valid
        if best_violations == 0:
            return best_colors_copy.tolist(), True
        
        # If not found, return the best we have (might be invalid, but we check later)
        return best_colors_copy.tolist(), False

    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate initial coloring
            # Strategy 0: Random
            if restart == 0:
                init_colors = np.random.randint(0, 2, size=len(list(combinations(range(target_n), 3))), dtype=np.int8).tolist()
            # Strategy 1: Modular 3
            elif restart == 1:
                init_colors = [0 if (a+b+c) % 3 == 0 else 1 for a, b, c in combinations(range(target_n), 3)]
            # Strategy 2: Modular 4
            elif restart == 2:
                init_colors = [0 if (a+b+c) % 4 == 0 else 1 for a, b, c in combinations(range(target_n), 3)]
            # Strategy 3: Random with bias
            else:
                init_colors = np.random.randint(0, 2, size=len(list(combinations(range(target_n), 3))), dtype=np.int8).tolist()
                # Flip 10% randomly
                for i in range(len(init_colors)):
                    if random.random() < 0.1:
                        init_colors[i] = 1 - init_colors[i]

            colors, success = run_local_search(target_n, init_colors, deadline - 2)
            
            if success:
                # Verify
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break

        if found:
            # Continue to next n
            continue
        else:
            # If we failed for this n, we might still try higher n?
            # No, if we can't find n, we likely can't find n+1 easily without a better seed.
            # But the task says "Maximize n". We should try to find the largest n.
            # If we fail for n, we should probably stop or try harder.
            # Given the time limit, if we fail for n, we return the previous best.
            # But maybe we can try a few more restarts for n if we have time?
            # The loop above already does 50 restarts.
            # If we fail, we break out and return best_n.
            break

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
