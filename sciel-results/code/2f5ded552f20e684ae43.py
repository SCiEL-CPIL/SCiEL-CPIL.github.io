import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120

    # Edge index lookup
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx

    # Precompute all K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)],
            edge_index[(c, d)]
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)

    def is_valid(colors):
        for e_indices in k4s:
            c_set = set()
            for idx in e_indices:
                c_set.add(colors[idx])
            if len(c_set) < 5:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Baseline coloring: (i + j) % 17
    # This uses colors from 1 to 15 (since i+j ranges from 1 to 29, mod 17 gives 1..16,0..12)
    # Actually let's just use (i+j) % 17
    best_colors = [(i + j) % 17 for i, j in edges]
    best_count = count_colors(best_colors)
    
    current_colors = best_colors[:]
    current_count = best_count

    # Helper to get used colors
    def get_used(colors):
        return list(set(colors))
    
    used = get_used(current_colors)

    max_iter = 10000
    iter_count = 0

    # Randomize seed for variety
    random.seed(int(time.time() * 1000) % 1000000)

    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # If current is invalid (shouldn't happen if we manage states carefully), reset
        if not is_valid(current_colors):
            current_colors = best_colors[:]
            used = get_used(current_colors)
            current_count = best_count
            continue

        # Try to merge a pair of colors
        if len(used) <= 2:
            break
            
        c1, c2 = random.sample(used, 2)
        
        # Create merged version
        merged_colors = [c1 if c == c2 else c for c in current_colors]
        
        # Check validity
        if is_valid(merged_colors):
            # Success
            current_colors = merged_colors
            used = get_used(current_colors)
            current_count = len(used)
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current_colors[:]
                
        else:
            # Failure. Try to repair.
            # Find a violating K4
            viol_k4_idx = -1
            for k_idx, e_indices in enumerate(k4s):
                c_set = set(merged_colors[idx] for idx in e_indices)
                if len(c_set) < 5:
                    viol_k4_idx = k_idx
                    break
                    
            if viol_k4_idx != -1:
                e_indices = k4s[viol_k4_idx]
                # The violation is due to merging c2 into c1.
                # So the K4 now has colors from a set of size < 5.
                # One of these colors is c1 (which was c1 or c2 before).
                # To fix, we need to change one edge to a color NOT in the current K4's color set.
                # But we want to keep the merge (c2->c1) globally, so ideally we don't reintroduce c2.
                # We should pick a color from `used` (which no longer contains c2, but contains others)
                # that is not in the current K4's color set.
                
                c_set = set(merged_colors[idx] for idx in e_indices)
                candidate_colors = [c for c in used if c not in c_set]
                
                if candidate_colors:
                    # Pick a random edge from the K4
                    # Prefer edges that were c2 (now c1) or c1?
                    # Any edge works for the constraint.
                    edge_to_change = random.choice(e_indices)
                    
                    # Try each candidate color
                    found_repair = False
                    for new_c in candidate_colors:
                        test_colors = merged_colors[:]
                        test_colors[edge_to_change] = new_c
                        if is_valid(test_colors):
                            current_colors = test_colors
                            used = get_used(current_colors)
                            current_count = len(used)
                            
                            if current_count < best_count:
                                best_count = current_count
                                best_colors = current_colors[:]
                                
                            found_repair = True
                            break
                            
                    # If no repair found, we revert to pre-merge state (which is effectively skipping this attempt)
                    # Note: current_colors is still the pre-merge valid state.

    return best_colors
