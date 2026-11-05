import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            # Simple product code fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    # balls[idx] is a list of 17 indices
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed=None, deterministic=False):
        if deterministic:
            rng = None
        else:
            rng = np.random.default_rng(seed)
        
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            # Count uncovered points in each ball
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                if deterministic:
                    best_idx = int(candidates[0])
                else:
                    best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
            
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True

        return selected

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def prune(selected_list, coverage_count, selected_set):
        """Remove redundant codewords. Returns new list, count, set."""
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # Check if any point in ball(idx) has count 1
                if np.any(coverage_count[balls[idx]] == 1):
                    continue
                # Redundant
                coverage_count[balls[idx]] -= 1
                selected_list.pop(i)
                selected_set.discard(idx)
                changed = True
                break
        return selected_list, coverage_count, selected_set

    def local_search(initial_code, deadline, seed):
        selected_list = list(initial_code)
        selected_set = set(selected_list)
        coverage_count = rebuild_coverage(selected_list)
        
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(seed)
        
        # Pre-compute private points for each selected codeword?
        # It's dynamic. We will compute on the fly for candidates.
        
        while time.time() < deadline:
            # 1. Prune
            selected_list, coverage_count, selected_set = prune(selected_list, coverage_count, selected_set)
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            if len(selected_list) == 0:
                break

            # 2. Find candidates for swap
            # We look for codewords with small private sets.
            # Private(c) = { p in Ball(c) | coverage_count[p] == 1 }
            
            candidates = []
            for c in selected_list:
                ball_c = balls[c]
                priv_mask = (coverage_count[ball_c] == 1)
                priv_count = np.sum(priv_mask)
                if priv_count == 0:
                    # Should have been pruned, but just in case
                    continue
                if priv_count <= 3: # Limit to small private sets for efficiency
                    candidates.append((priv_count, c, priv_mask))
            
            if not candidates:
                # No small private sets. Try a random swap or kick.
                # Kick: Add a random point, then prune.
                if rng.random() < 0.1:
                    # Add random
                    new_c = rng.randrange(total_points)
                    if new_c not in selected_set:
                        selected_list.append(new_c)
                        selected_set.add(new_c)
                        coverage_count[balls[new_c]] += 1
                        # Loop will prune
                    continue
                else:
                    # Try random swap
                    i = rng.randrange(len(selected_list))
                    c = selected_list[i]
                    # Pick random target
                    target = rng.randrange(total_points)
                    if target in selected_set:
                        continue
                    # Check validity: Private(c) subset Ball(target)
                    ball_c = balls[c]
                    priv_mask = (coverage_count[ball_c] == 1)
                    if np.any(priv_mask):
                        # Check if all private points are in Ball(target)
                        # i.e., for all p in Ball(c) where count==1, p in Ball(target)
                        # This is equivalent to: HammingDist(p, target) <= 1 for all such p
                        # We can check if Ball(target) covers the private points.
                        # Since we don't have a fast set intersection, we check:
                        # For each p in Ball(c) with count 1, is p in Ball(target)?
                        # Ball(target) is a list.
                        # Optimization: Precompute a boolean array for Ball(target)? No, too slow.
                        # Just iterate.
                        valid = True
                        for p in ball_c:
                            if coverage_count[p] == 1:
                                # Check if p is in neighbors[target]
                                # neighbors[target] is a list of 17 ints
                                if p not in neighbors[target]:
                                    valid = False
                                    break
                        if not valid:
                            continue
                    
                    # Perform swap
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[target]] += 1
                    selected_list[i] = target
                    selected_set.discard(c)
                    selected_set.add(target)
                    continue

            # Sort candidates by private count (smallest first)
            candidates.sort(key=lambda x: x[0])
            
            # Try to swap the best candidate
            swapped = False
            for priv_count, c, priv_mask in candidates:
                if time.time() > deadline:
                    break
                
                # Find target c' such that Private(c) subset Ball(c')
                # Private points are indices in ball_c where priv_mask is True
                priv_points = [ball_c[i] for i in range(len(ball_c)) if priv_mask[i]]
                
                # We need c' such that for all p in priv_points, p in Ball(c')
                # This means c' is in Intersection of Ball(p) for p in priv_points
                
                # Start with Ball(priv_points[0])
                potential_targets = set(neighbors[priv_points[0]])
                for p in priv_points[1:]:
                    potential_targets &= set(neighbors[p])
                    if not potential_targets:
                        break
                
                if not potential_targets:
                    continue
                
                # Filter potential targets:
                # 1. Must not be in selected_set (unless it's c itself, but we are moving c)
                # 2. Must be valid (which is guaranteed by construction of intersection)
                
                # We also want to prefer targets that help prune others.
                # For now, pick randomly from potential_targets
                valid_targets = [t for t in potential_targets if t not in selected_set or t == c]
                # If t == c, it's a no-op swap, skip
                valid_targets = [t for t in valid_targets if t != c]
                
                if not valid_targets:
                    continue
                
                # Pick a target
                # Heuristic: Pick target that maximizes coverage of other private points?
                # Or just random.
                target = rng.choice(valid_targets)
                
                # Perform swap
                coverage_count[balls[c]] -= 1
                coverage_count[balls[target]] += 1
                selected_list[selected_list.index(c)] = target
                selected_set.discard(c)
                selected_set.add(target)
                swapped = True
                break
            
            if not swapped:
                # If no swap found, do a kick
                if rng.random() < 0.5:
                    new_c = rng.randrange(total_points)
                    if new_c not in selected_set:
                        selected_list.append(new_c)
                        selected_set.add(new_c)
                        coverage_count[balls[new_c]] += 1
        
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    best_code = None
    best_len = float('inf')

    # Run multiple seeds
    # 1. Deterministic Greedy
    selected = greedy_set_cover(deterministic=True)
    selected = local_search(selected, deadline - 10.0, seed=0)
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected[:]

    # 2. Random Greedy Seeds
    num_seeds = 5
    time_per_seed = (deadline - start_time - 10.0) / num_seeds
    
    for i in range(num_seeds):
        if time.time() > deadline - 10.0:
            break
        seed = i + 1
        selected = greedy_set_cover(seed=seed)
        # Short local search for each seed
        end_time = min(deadline - 10.0, time.time() + time_per_seed)
        selected = local_search(selected, end_time, seed=seed)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    # Final polish on the best code found
    if best_code:
        best_code = local_search(best_code, deadline, seed=999)

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
