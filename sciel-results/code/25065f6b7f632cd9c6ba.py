"""Combined program: vectorized greedy + enhanced local search with 2-swaps and restarts for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        # Generic fallback: brute force or simple greedy
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Simple greedy for others
        total_words = q ** n
        word_to_idx = {tuple(w): i for i, w in enumerate(itertools.product(range(q), repeat=n))}
        neighbors = []
        for i in range(total_words):
            w = list(itertools.product(range(q), repeat=n))[i] # This is inefficient, better to use index conversion
            # For generic case, just use a simple greedy
            pass
        # We will focus on the specific case q=3, n=7, radius=1 as per task
        # If not specific, return a valid but potentially suboptimal code
        code = []
        uncovered = set(itertools.product(range(q), repeat=n))
        while uncovered:
            best_word = None
            best_count = -1
            # Sample candidates or iterate
            for word in list(uncovered):
                # Ball around word
                ball = set()
                for pos in range(n):
                    for val in range(q):
                        new_word = list(word)
                        if val == word[pos]:
                            continue
                        new_word[pos] = val
                        ball.add(tuple(new_word))
                ball.add(tuple(word))
                count = len(ball & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = word
            code.append(list(best_word))
            # Remove covered
            ball = set()
            for pos in range(n):
                for val in range(q):
                    new_word = list(best_word)
                    if val == best_word[pos]:
                        continue
                    new_word[pos] = val
                    ball.add(tuple(new_word))
            ball.add(tuple(best_word))
            uncovered -= ball
        return code

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    # Precompute index <-> tuple mappings
    idx_to_word = [None] * N
    word_to_idx = {}
    for i in range(N):
        w = [0] * n
        x = i
        for j in range(n):
            w[j] = x % 3
            x //= 3
        idx_to_word[i] = tuple(w)
        word_to_idx[tuple(w)] = i

    # Precompute neighbors as numpy arrays
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = list(idx_to_word[i])
        nb = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(3):
                if val == orig_val:
                    continue
                w[pos] = val
                nb.append(word_to_idx[tuple(w)])
            w[pos] = orig_val
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    def greedy_construct():
        """Phase 1: Greedy Set Cover with vectorized gain computation."""
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()

        while np.any(uncovered):
            if time.time() > deadline - 30:
                break

            # Compute gains: for each candidate codeword, how many uncovered points does it cover?
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                gains[neighbors_list[idx]] += 1

            best_idx = int(np.argmax(gains))
            max_gain = gains[best_idx]

            if max_gain == 0:
                break

            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors_list[best_idx]] = False

        return code, code_set

    def local_search(code, code_set, time_limit):
        """Phase 2: Local Search with deletion, 1-swap, and 2-swap."""
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

        best_size = len(code)
        best_code = list(code)
        best_set = set(code)
        
        improved = True
        no_improve_count = 0
        last_improve_time = time.time()

        while improved and time.time() < time_limit:
            improved = False

            # --- Deletion Pass ---
            to_remove = []
            for c in code:
                if np.min(counts[neighbors_list[c]]) > 1:
                    to_remove.append(c)

            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                last_improve_time = time.time()

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                best_set = set(code)

            # --- 1-Swap Pass ---
            if time.time() > time_limit - 5:
                break

            random.shuffle(code)
            for c in list(code):
                if time.time() > time_limit - 5:
                    break

                # Check if c is currently removable
                if np.min(counts[neighbors_list[c]]) > 1:
                    continue

                # Get critical points for c: points in neighbors[c] with counts == 1
                crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

                if not crit_points:
                    continue

                # Try to find a replacement w
                # Candidates must cover all critical points
                # We look for w in the union of neighbors of critical points
                candidate_pool = set()
                for p in crit_points:
                    candidate_pool.update(neighbor_sets[p])
                
                # Filter candidates not in code
                cands = [x for x in candidate_pool if x not in code_set]
                
                best_w = -1
                best_score = -1

                for w in cands:
                    # Check if w covers all critical points
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Score: maximize redundancy creation
                    # Points that are currently count 1 and will become count 2 (except critical ones which become 0 then 1? No, c is removed)
                    # If we remove c, crit_points become 0. If we add w, they become 1.
                    # Other points in neighbors[w] that were count 1 become 2.
                    # Other points in neighbors[w] that were count >1 become >1.
                    # Points in neighbors[c] not in crit_points (count >1) become count-1.
                    
                    # Simpler heuristic: Count how many points in neighbors[w] have count == 1.
                    # These will become 2, making them safer for future deletion.
                    score = int(np.sum(counts[neighbors_list[w]] == 1))
                    
                    # Penalize if w doesn't cover some points that c covered? 
                    # We already ensured w covers crit_points.
                    # What about points covered by c that are NOT crit_points?
                    # Those have count > 1. If we remove c, they become count-1.
                    # If w covers them, they become count.
                    # If w doesn't cover them, they become count-1.
                    # This is complex. Let's stick to the redundancy score.
                    
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors_list[best_w]] += 1

                    improved = True
                    last_improve_time = time.time()
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        best_set = set(code)

            # --- 2-Swap Pass (Limited) ---
            # Try to remove two codewords and add one or two new ones.
            # This is O(N^2) so we limit it.
            if time.time() < time_limit - 10 and len(code) > 10:
                # Pick a few random pairs
                pairs_tried = 0
                max_pairs = 50 # Limit to keep time down
                while pairs_tried < max_pairs and time.time() < time_limit - 10:
                    c1, c2 = random.sample(code, 2)
                    pairs_tried += 1
                    
                    # Critical points for c1 and c2
                    # A point is critical for the pair if it is covered ONLY by c1 and c2 (count 2) 
                    # or ONLY by one of them (count 1) and the other doesn't cover it?
                    # Actually, if we remove c1 and c2, a point is uncovered if:
                    # 1. It was covered by c1 only (count 1)
                    # 2. It was covered by c2 only (count 1)
                    # 3. It was covered by c1 and c2 only (count 2)
                    
                    # Let's find points that would become uncovered
                    temp_counts = counts.copy()
                    temp_counts[neighbors_list[c1]] -= 1
                    temp_counts[neighbors_list[c2]] -= 1
                    
                    would_be_uncovered = [i for i in range(N) if temp_counts[i] == 0]
                    
                    if not would_be_uncovered:
                        # We can just remove both? That would be a 2-deletion.
                        # But we need to maintain coverage. If would_be_uncovered is empty, 
                        # it means all points are covered by other codewords.
                        # So we can remove c1 and c2.
                        code.remove(c1)
                        code_set.remove(c1)
                        counts[neighbors_list[c1]] -= 1
                        code.remove(c2)
                        code_set.remove(c2)
                        counts[neighbors_list[c2]] -= 1
                        improved = True
                        last_improve_time = time.time()
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                            best_set = set(code)
                        continue

                    # Find candidates to cover would_be_uncovered
                    candidate_pool = set()
                    for p in would_be_uncovered:
                        candidate_pool.update(neighbor_sets[p])
                    
                    cands = [x for x in candidate_pool if x not in code_set]
                    
                    best_w1 = -1
                    best_w2 = -1
                    best_score = -1
                    
                    # Try adding 1 codeword
                    for w in cands:
                        valid = True
                        for p in would_be_uncovered:
                            if p not in neighbor_sets[w]:
                                valid = False
                                break
                        if not valid:
                            continue
                        score = int(np.sum(counts[neighbors_list[w]] == 1))
                        if score > best_score:
                            best_score = score
                            best_w1 = w
                            best_w2 = -1 # Mark as 1-swap replacement for 2-removals? 
                            # Actually, if we remove 2 and add 1, size decreases by 1.
                            # This is a "2-remove 1-add" move.
                    
                    # Try adding 2 codewords (if 1 wasn't enough or better score)
                    # This is expensive. Skip for now, rely on 1-swap and deletion.
                    
                    if best_w1 != -1:
                        # Remove c1, c2. Add best_w1.
                        code.remove(c1)
                        code_set.remove(c1)
                        counts[neighbors_list[c1]] -= 1
                        code.remove(c2)
                        code_set.remove(c2)
                        counts[neighbors_list[c2]] -= 1
                        
                        code.append(best_w1)
                        code_set.add(best_w1)
                        counts[neighbors_list[best_w1]] += 1
                        
                        improved = True
                        last_improve_time = time.time()
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                            best_set = set(code)

            if not improved:
                no_improve_count += 1
                # If stuck for a while, break to try restart
                if time.time() - last_improve_time > 5.0:
                    break
            else:
                no_improve_count = 0

        return best_code, best_set

    # Main Loop with Restarts
    best_code_overall = None
    best_size_overall = float('inf')

    # First run
    code, code_set = greedy_construct()
    best_code, best_set = local_search(code, code_set, deadline - 20)
    best_code_overall = best_code
    best_size_overall = len(best_code)

    # Try restarts with perturbation if time permits
    while time.time() < deadline - 10:
        # Perturb the best solution
        code, code_set = list(best_code_overall), set(best_code_overall)
        
        # Randomly remove a few codewords and repair
        num_remove = min(5, len(code) // 10)
        to_remove = random.sample(code, num_remove)
        for c in to_remove:
            code.remove(c)
            code_set.remove(c)
        
        # Repair using greedy
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1
        
        uncovered = np.where(counts == 0)[0]
        
        while np.any(uncovered > 0) and time.time() < deadline - 5:
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(counts == 0)[0]
            for idx in uncovered_indices:
                gains[neighbors_list[idx]] += 1
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            counts[neighbors_list[best_idx]] += 1
        
        # Local search on perturbed solution
        best_code, best_set = local_search(code, code_set, min(deadline - 5, time.time() + 10))
        
        if len(best_code) < best_size_overall:
            best_size_overall = len(best_code)
            best_code_overall = best_code
            # If we found a significantly better solution, we might want to restart local search on it
            # But the loop above handles it.

    # Convert best_code_overall back to list of lists
    result = [list(idx_to_word[cw]) for cw in best_code_overall]

    return result
# EVOLVE-BLOCK-END
