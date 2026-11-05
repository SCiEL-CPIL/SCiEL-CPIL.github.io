import time
import random
import heapq
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    Precomputes transitions for all 4096 masks.
    """
    all_states = (1 << N) - 1
    
    # Precompute transitions for all masks using bit manipulation
    # For each state i, precompute which bit it maps to under a and b
    # Then for a mask, the result is OR of all destination bits
    
    # Fast precomputation: for each mask, compute trans_a[mask] and trans_b[mask]
    # Use the fact that trans(mask) = OR over i in mask of (1 << trans[i])
    
    # Build lookup: for each possible 3-bit chunk (0-7), precompute contribution
    # Actually, simplest: iterate bits of mask
    
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        # Get lowest set bit
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        trans_a[mask] = trans_a[rest] | (1 << a_trans[i])
        trans_b[mask] = trans_b[rest] | (1 << b_trans[i])
    
    start = all_states
    
    # Check if already singleton
    if (start & (start - 1)) == 0:
        return 0

    # BFS with visited array (list of -1 for unvisited, distance for visited)
    visited = [-1] * (1 << N)
    visited[start] = 0
    
    queue = deque([start])
    dist = 0
    
    while queue:
        dist += 1
        for _ in range(len(queue)):
            curr = queue.popleft()
            
            # Generate neighbors
            next_a = trans_a[curr]
            next_b = trans_b[curr]
            
            if next_a != next_b:
                # Check next_a
                if visited[next_a] == -1:
                    if (next_a & (next_a - 1)) == 0:
                        return dist
                    visited[next_a] = dist
                    queue.append(next_a)
                # Check next_b
                if visited[next_b] == -1:
                    if (next_b & (next_b - 1)) == 0:
                        return dist
                    visited[next_b] = dist
                    queue.append(next_b)
            else:
                if visited[next_a] == -1:
                    if (next_a & (next_a - 1)) == 0:
                        return dist
                    visited[next_a] = dist
                    queue.append(next_a)
    
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Helper to update best
    def update(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = list(a)
            best_b = list(b)
        return score

    # Strategy: Structured families for initial seeds, then steepest-ascent hill climbing
    
    # 1. 'a' is a full cycle (shift by 1)
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Sub-family 1a: b is identity with 1 defect
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = list(range(N))
            b[k] = j
            if time.time() > deadline:
                return [best_a, best_b]
            update(a_cycle, b)
            
    # Sub-family 1b: Pinsker-like structure
    b_pinsker = [0] * N
    b_pinsker[0] = 0
    for i in range(1, N):
        if i == N - 1:
            b_pinsker[i] = 1
        else:
            b_pinsker[i] = i + 1
    update(a_cycle, b_pinsker)
    
    # 2. 'a' is a shift by d
    for d in range(1, 7):
        a_shift = [(i + d) % N for i in range(N)]
        # Try b as Pinsker-like cycle fixing one state
        for fixed in range(N):
            b = [0] * N
            b[fixed] = fixed
            others = [i for i in range(N) if i != fixed]
            for i in range(len(others)):
                b[others[i]] = others[(i + 1) % len(others)]
            if time.time() > deadline:
                return [best_a, best_b]
            update(a_shift, b)
        
        # Try b as identity with 1 defect
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = list(range(N))
                b[k] = j
                if time.time() > deadline:
                    return [best_a, best_b]
                update(a_shift, b)

    # Ensure we have a valid starting point
    if best_a is None:
        best_a = a_cycle
        best_b = b_pinsker
        update(best_a, best_b)
    
    # 3. Steepest-ascent hill climbing from multiple starting points
    # Create a list of starting points for hill climbing
    starting_points = [(best_a, best_b)]
    
    # Add some other structured starting points
    for d in [1, 2, 3, 4, 5, 6]:
        a_shift = [(i + d) % N for i in range(N)]
        b_pinsker_d = [0] * N
        b_pinsker_d[0] = 0
        for i in range(1, N):
            if i == N - 1:
                b_pinsker_d[i] = 1
            else:
                b_pinsker_d[i] = i + 1
        starting_points.append((a_shift, b_pinsker_d))
    
    # Random starting points
    random.seed(42)
    for _ in range(30):
        a_rand = [random.randint(0, N-1) for _ in range(N)]
        b_rand = [random.randint(0, N-1) for _ in range(N)]
        starting_points.append((a_rand, b_rand))
    
    # Steepest-ascent hill climbing from each starting point
    for start_a, start_b in starting_points:
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        
        current_a = list(start_a)
        current_b = list(start_b)
        current_score = calculate_threshold(current_a, current_b)
        
        # If this starting point is better than our best, update
        if current_score > best_score:
            best_score = current_score
            best_a = list(current_a)
            best_b = list(current_b)
        
        # Steepest ascent: try all single-position changes, take the best improvement
        no_improve_count = 0
        max_no_improve = 50  # Stop after 50 consecutive no-improvement rounds
        
        while no_improve_count < max_no_improve:
            if time.time() > deadline - 3:  # Leave 3s buffer
                break
            
            best_neighbor_score = current_score
            best_neighbor_a = None
            best_neighbor_b = None
            
            # Try all single-position changes in a (N positions x N values = N*(N-1) changes)
            for pos in range(N):
                orig_val = current_a[pos]
                for new_val in range(N):
                    if new_val == orig_val:
                        continue
                    candidate_a = list(current_a)
                    candidate_a[pos] = new_val
                    score = calculate_threshold(candidate_a, current_b)
                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_a = candidate_a
                        best_neighbor_b = list(current_b)
            
            # Try all single-position changes in b
            for pos in range(N):
                orig_val = current_b[pos]
                for new_val in range(N):
                    if new_val == orig_val:
                        continue
                    candidate_b = list(current_b)
                    candidate_b[pos] = new_val
                    score = calculate_threshold(current_a, candidate_b)
                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_a = list(current_a)
                        best_neighbor_b = candidate_b
            
            if best_neighbor_a is not None:
                current_a = best_neighbor_a
                current_b = best_neighbor_b
                current_score = best_neighbor_score
                no_improve_count = 0
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)
            else:
                no_improve_count += 1
        
        # After steepest ascent, try some 2-position mutations for escape
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            
            a_new = list(current_a)
            b_new = list(current_b)
            
            # Mutate 2 positions
            for _ in range(2):
                if random.random() < 0.5:
                    idx = random.randint(0, N-1)
                    val = random.randint(0, N-1)
                    a_new[idx] = val
                else:
                    idx = random.randint(0, N-1)
                    val = random.randint(0, N-1)
                    b_new[idx] = val
            
            score = calculate_threshold(a_new, b_new)
            if score > current_score:
                current_a = a_new
                current_b = b_new
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)
                
                # Continue steepest ascent from new point
                no_improve_count = 0
                while no_improve_count < 20:
                    if time.time() > deadline - 3:
                        break
                    best_neighbor_score = current_score
                    best_neighbor_a = None
                    best_neighbor_b = None
                    
                    for pos in range(N):
                        orig_val = current_a[pos]
                        for new_val in range(N):
                            if new_val == orig_val:
                                continue
                            candidate_a = list(current_a)
                            candidate_a[pos] = new_val
                            score = calculate_threshold(candidate_a, current_b)
                            if score > best_neighbor_score:
                                best_neighbor_score = score
                                best_neighbor_a = candidate_a
                                best_neighbor_b = list(current_b)
                    
                    for pos in range(N):
                        orig_val = current_b[pos]
                        for new_val in range(N):
                            if new_val == orig_val:
                                continue
                            candidate_b = list(current_b)
                            candidate_b[pos] = new_val
                            score = calculate_threshold(current_a, candidate_b)
                            if score > best_neighbor_score:
                                best_neighbor_score = score
                                best_neighbor_a = list(current_a)
                                best_neighbor_b = candidate_b
                    
                    if best_neighbor_a is not None:
                        current_a = best_neighbor_a
                        current_b = best_neighbor_b
                        current_score = best_neighbor_score
                        no_improve_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_a = list(current_a)
                            best_b = list(current_b)
                    else:
                        no_improve_count += 1

    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
