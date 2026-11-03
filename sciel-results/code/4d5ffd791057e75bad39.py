"""Ternary covering code K_3(6,1) via greedy set cover + WalkSAT-style local search."""
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
        digits.reverse()  # Most significant first
        idx_to_digits.append(tuple(digits))

    def index_of_digits(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

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
        # General radius (not expected for this task)
        result = [center_idx]
        for delta_tuple in itertools.product(range(q), repeat=n):
            dist = sum(1 for i in range(n) if delta_tuple[i] != 0)
            if dist == 0 or dist > radius:
                continue
            new_digits = [(center_digits[i] + delta_tuple[i]) % q for i in range(n)]
            result.append(index_of_digits(tuple(new_digits)))
        return result

    # Precompute balls for all candidates
    all_candidate_digits = idx_to_digits  # list of tuples
    all_balls = [ball_indices(digits) for digits in all_candidate_digits]
    
    # Precompute ball sets for faster membership testing
    all_balls_set = [set(b) for b in all_balls]

    # Greedy set cover: repeatedly pick the candidate that covers the most uncovered words
    uncovered = [True] * space_size
    uncovered_count = space_size
    selected = []
    selected_balls = []
    selected_set = set()

    # For efficiency, maintain for each candidate how many uncovered words it covers
    # and for each uncovered word, which candidates cover it (or just recompute)

    # Since space is small (729 words, 729 candidates, each ball has 19 words),
    # we can afford a reasonable greedy approach.

    # Precompute: for each candidate index, the set of words it covers
    # Actually all_balls[i] is the list of indices covered by candidate i

    # Greedy: at each step, find candidate maximizing number of newly covered words
    # To speed up, we track coverage count per word
    word_cover_count = [0] * space_size  # How many selected codewords cover this word

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        best_gain = 0
        best_candidate = -1

        # For each candidate not yet selected, compute gain
        # Optimization: only consider candidates that cover at least one uncovered word
        for ci in range(space_size):
            if time.time() > deadline:
                break
            gain = 0
            for w in all_balls[ci]:
                if uncovered[w]:
                    gain += 1
                    if gain > best_gain:
                        # Early exit if we can't beat current best
                        pass
            if gain > best_gain:
                best_gain = gain
                best_candidate = ci
            # Early exit: if best_gain is already high, we can stop checking others
            # but we need to check all to be sure (or use a heuristic)

        if best_candidate == -1 or best_gain == 0:
            break

        # Select this candidate
        selected.append(all_candidate_digits[best_candidate])
        selected_balls.append(all_balls[best_candidate])
        selected_set.add(best_candidate)
        for w in all_balls[best_candidate]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1

        # Early exit if fully covered
        if uncovered_count == 0:
            break

    # Phase 2: Local search to reduce the number of codewords
    # Try removing codewords that are redundant
    def is_redundant(i, selected_balls_list, selected_set_local):
        """Check if codeword i is redundant (its ball is covered by others)."""
        ball_i = selected_balls_list[i]
        for w in ball_i:
            found = False
            for j in range(len(selected_balls_list)):
                if j == i:
                    continue
                if w in selected_balls_set[j]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Redundancy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            if is_redundant(i, selected_balls, selected_set):
                selected_set.discard(index_of_digits(selected[i]))
                del selected[i]
                del selected_balls[i]
                changed = True
                break

    # Phase 3: WalkSAT-style local search
    # Treat the problem as a hypergraph 2-SAT reduction where each space word is a clause
    # requiring at least one incident codeword. Apply randomized WalkSAT-style local search
    # that flips the selection state of codewords to satisfy the maximum number of clauses.
    
    rng = random.Random(42)
    
    # Track coverage counts for each word
    cover_count = [0] * space_size
    for ball in selected_balls:
        for w in ball:
            cover_count[w] += 1
    
    # Uncovered words are those with cover_count == 0
    def get_uncovered():
        return [w for w in range(space_size) if cover_count[w] == 0]
    
    # WalkSAT-style local search
    best_code = selected[:]
    best_size = len(selected)
    best_cover_count = cover_count[:]
    
    temp = 1.0
    cool_rate = 0.999
    min_temp = 0.01
    
    def is_removable(idx):
        """Check if codeword at index idx can be removed without creating uncovered words."""
        ball = selected_balls[idx]
        for w in ball:
            if cover_count[w] == 1:
                return False
        return True
    
    def try_swap(old_idx, new_idx):
        """Try swapping codeword at old_idx with new_idx."""
        old_ball = selected_balls[old_idx]
        new_ball = all_balls[new_idx]
        
        # Check if swap maintains coverage
        for w in old_ball:
            if cover_count[w] == 1 and w not in all_balls_set[new_idx]:
                return False
        return True
    
    # Main WalkSAT loop
    while time.time() < deadline:
        if temp < min_temp:
            temp = min_temp
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Remove
            if len(selected) > 0:
                idx = rng.randint(0, len(selected) - 1)
                if is_removable(idx):
                    # Remove codeword
                    for w in selected_balls[idx]:
                        cover_count[w] -= 1
                    selected_set.discard(index_of_digits(selected[idx]))
                    del selected[idx]
                    del selected_balls[idx]
                    
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_code = selected[:]
                        best_cover_count = cover_count[:]
                        
        elif move_type < 0.8:
            # Try Swap
            if len(selected) > 0:
                idx = rng.randint(0, len(selected) - 1)
                new_idx = rng.randint(0, space_size - 1)
                
                if new_idx in selected_set or new_idx == index_of_digits(selected[idx]):
                    continue
                    
                if try_swap(idx, new_idx):
                    # Perform swap
                    old_ball = selected_balls[idx]
                    new_ball = all_balls[new_idx]
                    
                    for w in old_ball:
                        cover_count[w] -= 1
                    for w in new_ball:
                        cover_count[w] += 1
                    
                    selected_set.discard(index_of_digits(selected[idx]))
                    selected[idx] = all_candidate_digits[new_idx]
                    selected_balls[idx] = new_ball
                    selected_set.add(new_idx)
                    
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_code = selected[:]
                        best_cover_count = cover_count[:]
                        
        else:
            # Try Add (Perturbation)
            new_idx = rng.randint(0, space_size - 1)
            if new_idx not in selected_set:
                if rng.random() < 0.1 / temp:
                    # Add codeword
                    for w in all_balls[new_idx]:
                        cover_count[w] += 1
                    selected.append(all_candidate_digits[new_idx])
                    selected_balls.append(all_balls[new_idx])
                    selected_set.add(new_idx)
                    
        temp *= cool_rate
    
    # Use the best code found
    selected = best_code
    selected_balls = [all_balls[index_of_digits(cw)] for cw in selected]
    selected_set = set(index_of_digits(cw) for cw in selected)
    
    # Final cleanup: remove any redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            if is_redundant(i, selected_balls, selected_set):
                selected_set.discard(index_of_digits(selected[i]))
                del selected[i]
                del selected_balls[i]
                changed = True
                break

    # Convert tuples to lists for return
    return [list(cw) for cw in selected]
