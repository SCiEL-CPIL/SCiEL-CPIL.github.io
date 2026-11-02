import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]

    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15

    # Precompute all words as index arrays
    all_words = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        temp = i
        for j in range(n - 1, -1, -1):
            all_words[i, j] = temp % q
            temp //= q

    # Precompute base values for index arithmetic
    base = np.zeros(n, dtype=np.int32)
    base[n - 1] = 1
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = []
    for i in range(N):
        w = all_words[i]
        ball = [i]
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball.append(new_idx)
        ball_indices.append(np.array(ball, dtype=np.int32))
    
    # Convert to list of lists for faster iteration in some contexts, or keep as numpy array of arrays
    # We'll use a list of numpy arrays for ball_indices[c]
    
    # Helper: Greedy Set Cover (Fast version using NumPy counts)
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        code = []
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = N
        remaining_candidates = list(range(N))
        rng.shuffle(remaining_candidates)

        while uncovered > 0:
            best_i = -1
            best_gain = 0
            
            # Iterate through remaining candidates to find the one covering most new points
            # Optimization: We can break early if we find a codeword covering all uncovered points in its ball?
            # Max possible gain is ball_size.
            for i in remaining_candidates:
                # Calculate how many points in ball(i) are currently uncovered (count == 0)
                b = ball_indices[i]
                gain = np.sum(cover_count[b] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    if best_gain == ball_size:
                        break
            
            if best_i == -1:
                break # Should not happen if uncovered > 0 and candidates not empty

            code.append(best_i)
            b = ball_indices[best_i]
            cover_count[b] += 1
            uncovered -= np.sum(cover_count[b] == 1) # Only count those that went from 0 to 1
            remaining_candidates.remove(best_i)

        return code

    # Helper: Local Search (Swap and Remove)
    def local_search(code, cover_count, time_limit):
        code_set = set(code)
        improved = True
        no_improvement_count = 0
        max_no_improvement = 20
        current_temp = 1.0
        cooling_rate = 0.95
        rng = random.Random(123)

        while time.time() < time_limit:
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                current_temp *= cooling_rate
                if no_improvement_count > max_no_improvement:
                    # Perturbation
                    if len(code) > 0:
                        # Remove random
                        idx = rng.randint(0, len(code) - 1)
                        cw = code[idx]
                        code.pop(idx)
                        cover_count[ball_indices[cw]] -= 1
                        # Add random
                        new_cw = rng.randint(0, N - 1)
                        if new_cw not in code_set:
                            code.append(new_cw)
                            code_set.add(new_cw)
                            cover_count[ball_indices[new_cw]] += 1
                    no_improvement_count = 0
                    improved = True # Force loop to continue
            
            improved = False
            
            # 1. Try Removals
            # Find a codeword that can be removed
            # We need to check if min(cover_count[ball(cw)]) > 1
            # Doing this for all cw is O(K * ball_size). K is ~186, ball_size 15. 186*15 = 2790 ops. Very fast.
            remove_idx = -1
            for i, cw in enumerate(code):
                b = ball_indices[cw]
                if np.min(cover_count[b]) > 1:
                    remove_idx = i
                    break
            
            if remove_idx != -1:
                cw = code[remove_idx]
                code.pop(remove_idx)
                code_set.discard(cw)
                cover_count[ball_indices[cw]] -= 1
                improved = True
                continue

            # 2. Try Swaps
            # Pick a random codeword to try replacing
            if not code:
                break
                
            idx = rng.randint(0, len(code) - 1)
            old_cw = code[idx]
            b_old = ball_indices[old_cw]
            
            # Points uniquely covered by old_cw
            unique_mask = cover_count[b_old] == 1
            if np.sum(unique_mask) == 0:
                # Should have been removed above, but handle just in case
                code.pop(idx)
                code_set.discard(old_cw)
                cover_count[b_old] -= 1
                improved = True
                continue
                
            unique_points = b_old[unique_mask]
            
            # Find a candidate new_cw that covers all unique_points
            # We can sample some candidates or iterate.
            # To be fast, we'll try a few random candidates from the space, 
            # or specifically look for candidates that cover the unique points.
            # A good heuristic: Pick a random unique point p. Candidates must be in ball(p).
            # But they must cover ALL unique points.
            
            # Let's pick a random unique point to start the search
            p = unique_points[rng.randint(0, len(unique_points)-1)]
            # Candidates are in ball_indices[p]
            candidates = ball_indices[p]
            
            found_swap = False
            for new_cw in candidates:
                if new_cw in code_set:
                    continue
                if new_cw == old_cw:
                    continue
                    
                # Check if new_cw covers all unique points
                b_new = ball_indices[new_cw]
                # We need to check if all unique_points are in b_new
                # Efficient check: 
                # unique_points are indices. b_new is an array of indices.
                # We can create a set of b_new for fast lookup, but that's slow.
                # Alternative: 
                # Since unique_points are few (usually 1-5), we can check membership.
                # Or use numpy: np.all(np.isin(unique_points, b_new))
                
                if np.all(np.isin(unique_points, b_new)):
                    # Valid swap candidate. 
                    # Calculate cost/benefit? 
                    # For now, accept the first valid one found or pick random valid.
                    # To improve, we could pick the one that maximizes redundancy removal or minimizes new redundancy.
                    # Let's just accept it to ensure progress.
                    
                    # Perform swap
                    code[idx] = new_cw
                    code_set.discard(old_cw)
                    code_set.add(new_cw)
                    
                    cover_count[b_old] -= 1
                    cover_count[b_new] += 1
                    
                    improved = True
                    found_swap = True
                    break
                    
            if not found_swap:
                # If we tried and failed, we can try another codeword in the next iteration
                pass

        return code, cover_count

    # Main Logic
    best_code = []
    best_size = float('inf')
    
    # 1. Multiple Greedy Restarts
    # Greedy is now fast enough to run several times
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 60.0:
            break
        code = greedy_build(seed=i)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    # 2. Local Search on the best greedy solution
    if best_code:
        # Initialize cover counts for local search
        cover_count = np.zeros(N, dtype=np.int32)
        for cw in best_code:
            cover_count[ball_indices[cw]] += 1
        
        # Run local search
        code, cover_count = local_search(best_code, cover_count, deadline - 10.0)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code

    # 3. Final Polish: Try to remove redundancies one last time
    if best_code:
        cover_count = np.zeros(N, dtype=np.int32)
        for cw in best_code:
            cover_count[ball_indices[cw]] += 1
            
        # Repeat removal until stable
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(best_code)):
                cw = best_code[i]
                b = ball_indices[cw]
                if np.min(cover_count[b]) > 1:
                    best_code.pop(i)
                    cover_count[b] -= 1
                    changed = True
                    break # Restart loop as indices changed

    # Verify and Return
    if not best_code:
        # Fallback: Return full space (should not happen)
        return [list(w) for w in all_words]
        
    # Final Verification
    cover_count = np.zeros(N, dtype=np.int32)
    for cw in best_code:
        cover_count[ball_indices[cw]] += 1
        
    if np.min(cover_count) > 0:
        return [list(all_words[cw]) for cw in best_code]
    else:
        # If invalid, return a valid but potentially larger code
        # Greedy solution is always valid if it completes
        return [list(all_words[cw]) for cw in greedy_build(seed=0)]
