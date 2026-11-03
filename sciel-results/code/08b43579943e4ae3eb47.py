"""Ternary covering code K_3(6,1) via greedy independent set + local search."""
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Total space size
    space_size = q ** n  # 729 for q=3, n=6

    # Precompute all indices and their digit representations
    idx_to_digits = []
    for idx in range(space_size):
        digits = []
        x = idx
        for i in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        idx_to_digits.append(tuple(digits))

    def index_of_digits(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    def hamming_dist(d1, d2):
        return sum(1 for i in range(n) if d1[i] != d2[i])

    def ball_indices(center_digits):
        """Return list of indices within Hamming distance <= radius from center_digits."""
        center_idx = index_of_digits(center_digits)
        if radius == 0:
            return [center_idx]
        if radius == 1:
            result = [center_idx]
            for pos in range(n):
                for d in range(q):
                    if d == center_digits[pos]:
                        continue
                    new_digits = list(center_digits)
                    new_digits[pos] = d
                    result.append(index_of_digits(tuple(new_digits)))
            return result
        result = [center_idx]
        for delta_tuple in itertools.product(range(q), repeat=n):
            dist = sum(1 for i in range(n) if delta_tuple[i] != 0)
            if dist == 0 or dist > radius:
                continue
            new_digits = [(center_digits[i] + delta_tuple[i]) % q for i in range(n)]
            result.append(index_of_digits(tuple(new_digits)))
        return result

    all_candidate_digits = idx_to_digits
    all_balls = [ball_indices(digits) for digits in all_candidate_digits]

    # Phase 1: Greedy independent set on distance-2 graph
    # Two codewords at Hamming distance 2 can potentially be merged/reduced
    # We want an initial set where no two codewords are at distance 2
    # Build adjacency: i ~ j if dist(i,j) == 2
    dist2_adj = [[] for _ in range(space_size)]
    for i in range(space_size):
        for j in range(i + 1, space_size):
            d = hamming_dist(idx_to_digits[i], idx_to_digits[j])
            if d == 2:
                dist2_adj[i].append(j)
                dist2_adj[j].append(i)

    # Greedy independent set: pick vertices that don't conflict
    # Sort by degree (descending) to get a better independent set
    degree = [len(dist2_adj[i]) for i in range(space_size)]
    order = sorted(range(space_size), key=lambda x: -degree[x])
    
    in_set = [False] * space_size
    independent_set = []
    for v in order:
        # Check if v conflicts with any already in set
        conflict = False
        for u in dist2_adj[v]:
            if in_set[u]:
                conflict = True
                break
        if not conflict:
            in_set[v] = True
            independent_set.append(v)

    # Use independent set as initial codeword set
    selected = [idx_to_digits[v] for v in independent_set]
    selected_balls = [all_balls[v] for v in independent_set]
    selected_indices = list(independent_set)

    # Check coverage - if not fully covered, add more codewords greedily
    uncovered = [True] * space_size
    for b in selected_balls:
        for w in b:
            uncovered[w] = False
    uncovered_count = sum(uncovered)

    while uncovered_count > 0 and time.time() < deadline:
        best_gain = 0
        best_candidate = -1
        for ci in range(space_size):
            if in_set[ci]:
                continue
            gain = 0
            for w in all_balls[ci]:
                if uncovered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = ci
        if best_candidate == -1 or best_gain == 0:
            break
        in_set[best_candidate] = True
        selected.append(idx_to_digits[best_candidate])
        selected_balls.append(all_balls[best_candidate])
        selected_indices.append(best_candidate)
        for w in all_balls[best_candidate]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1

    # Phase 2: Local search - redundancy removal
    def is_redundant(i, balls_list):
        ball_i = balls_list[i]
        for w in ball_i:
            found = False
            for j in range(len(balls_list)):
                if j == i:
                    continue
                if w in balls_list[j]:
                    found = True
                    break
            if not found:
                return False
        return True

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            if is_redundant(i, selected_balls):
                removed_idx = selected_indices[i]
                in_set[removed_idx] = False
                del selected[i]
                del selected_balls[i]
                del selected_indices[i]
                changed = True
                break

    # Phase 3: Swap-based local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            # Temporarily remove codeword i
            remaining_balls = [b for j, b in enumerate(selected_balls) if j != i]
            covered_set = set()
            for b in remaining_balls:
                covered_set.update(b)
            missing = [w for w in range(space_size) if w not in covered_set]
            if not missing:
                removed_idx = selected_indices[i]
                in_set[removed_idx] = False
                del selected[i]
                del selected_balls[i]
                del selected_indices[i]
                improved = True
                break

            # Try to find candidate j that covers all missing words
            for j in range(space_size):
                if time.time() > deadline:
                    break
                if in_set[j]:
                    continue
                ball_j = all_balls[j]
                if not all(w in ball_j for w in missing):
                    continue

                # New set: remaining + j
                new_balls = remaining_balls + [ball_j]
                # Check if any is redundant
                for k in range(len(new_balls)):
                    if is_redundant(k, new_balls):
                        # Do the swap
                        removed_i_idx = selected_indices[i]
                        removed_k_idx = selected_indices[k] if k < len(selected) else j
                        in_set[removed_i_idx] = False
                        if k < len(selected):
                            in_set[removed_k_idx] = False
                        in_set[j] = True

                        new_selected = [selected[m] for m in range(len(selected)) if m != i]
                        if k < len(new_selected):
                            del new_selected[k]
                        new_selected.append(idx_to_digits[j])

                        new_selected_balls = [selected_balls[m] for m in range(len(selected)) if m != i]
                        if k < len(new_selected_balls):
                            del new_selected_balls[k]
                        new_selected_balls.append(ball_j)

                        new_selected_indices = [selected_indices[m] for m in range(len(selected)) if m != i]
                        if k < len(new_selected_indices):
                            del new_selected_indices[k]
                        new_selected_indices.append(j)

                        selected = new_selected
                        selected_balls = new_selected_balls
                        selected_indices = new_selected_indices
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Final cleanup
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            if is_redundant(i, selected_balls):
                removed_idx = selected_indices[i]
                in_set[removed_idx] = False
                del selected[i]
                del selected_balls[i]
                del selected_indices[i]
                changed = True
                break

    return [list(cw) for cw in selected]
