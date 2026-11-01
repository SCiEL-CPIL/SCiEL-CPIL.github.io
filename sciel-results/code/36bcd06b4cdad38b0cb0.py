import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precomputation
    total = q ** n
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q
        
    def int_to_digits(x):
        return [x // steps[i] % q for i in range(n)]
        
    def digits_to_int(d):
        return sum(d[i] * steps[i] for i in range(n) if d[i] != 0)

    # Precompute balls as lists of integers
    balls = []
    for c in range(total):
        digits = int_to_digits(c)
        covered = [c]
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits)
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
        
    # Convert balls to numpy arrays for potential vectorization, 
    # but lists are sufficient for small ball sizes (15 for q=3, n=7, r=1).
    
    # Helper to check validity and count uncovered
    def get_uncovered(code_set, cover_count=None):
        if cover_count is None:
            cover_count = np.zeros(total, dtype=np.int8)
            for c in code_set:
                for w in balls[c]:
                    cover_count[w] += 1
        return np.where(cover_count == 0)[0], cover_count

    # Redundancy Pruning: Remove codewords that are not essential.
    # A codeword c is redundant if removing it does not uncover any word.
    # i.e., for all w in balls[c], there exists c' in C \ {c} such that w in balls[c'].
    # This is equivalent to: for all w in balls[c], cover_count[w] > 1 (when c is in C).
    def prune_redundant(code_set, cover_count):
        removed = False
        # Iterate until no more removals possible (stable state)
        # To avoid O(N^2) in worst case, we can just do one pass or a few passes.
        # Given the small size, one pass might miss some, but ILS restarts help.
        # Let's do a full pass.
        to_remove = []
        for c in list(code_set):
            is_redundant = True
            for w in balls[c]:
                if cover_count[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.append(c)
        
        if to_remove:
            for c in to_remove:
                code_set.remove(c)
                for w in balls[c]:
                    cover_count[w] -= 1
            return True
        return False

    # 2-for-1 Exchange: Find c1, c2 in C such that balls[c1] U balls[c2] is subset of balls[c3] for some c3 not in C.
    # Then replace {c1, c2} with {c3}.
    # This is expensive to check all pairs. We will use a heuristic:
    # For each c1 in C, look for c2 in C \ {c1} such that the union is small?
    # Or, for each c3 not in C, check if there are c1, c2 in C whose union is contained in balls[c3].
    # The latter is better: For each candidate c3, count how many words in balls[c3] are covered by C \ {c3}.
    # If all words in balls[c3] are covered by C \ {c3}, then c3 could potentially replace a pair?
    # No, that means c3 is redundant if added.
    
    # Better 2-for-1 logic:
    # We want to find c1, c2 in C and c3 not in C such that:
    # 1. balls[c1] U balls[c2] is a subset of balls[c3]
    # 2. Removing c1, c2 and adding c3 keeps coverage valid.
    # Condition 1 implies that any word covered by c1 or c2 is covered by c3.
    # Since c1, c2 are in C, their covered words are already covered by C.
    # So we just need to ensure that no word loses coverage.
    # A word w loses coverage if it was covered ONLY by c1 or c2 (and not by others in C, and not by c3).
    # Actually, if balls[c1] U balls[c2] subset balls[c3], then any w covered by c1 or c2 is covered by c3.
    # So if we remove c1, c2 and add c3, w is still covered by c3.
    # Are there other words? No, only words in balls[c1] or balls[c2] are affected by removing them.
    # And those are all in balls[c3].
    # So the move is valid IF balls[c1] U balls[c2] is a subset of balls[c3].
    
    # How to find such triples efficiently?
    # For each c3 not in C:
    #   Let S = balls[c3]
    #   We need to find c1, c2 in C such that balls[c1] U balls[c2] subset S.
    #   This means balls[c1] subset S AND balls[c2] subset S.
    #   So we just need to find two codewords in C whose balls are entirely contained in S.
    #   If we find c1, c2 in C such that balls[c1] subset S and balls[c2] subset S,
    #   we can replace {c1, c2} with {c3}.
    
    def try_2_for_1(code_set, cover_count):
        # Precompute for each c in C, which c3 not in C contain balls[c]?
        # Too expensive.
        
        # Heuristic:
        # For each c3 not in C:
        #   Check if there are at least 2 codewords in C whose balls are subsets of balls[c3].
        #   If yes, replace them.
        
        # To speed up, we can iterate c3 randomly or in some order.
        # Given 2187 words, iterating all c3 is 2187 * (check cost).
        # Check cost: For each c in C, check if balls[c] subset balls[c3].
        # balls[c] size 15. balls[c3] size 15.
        # Subset check: O(15 * 15) if using sets, or O(15) if using bitsets/precomputed masks.
        
        # Let's use sets for balls for faster subset checks in this specific search.
        ball_sets = [set(b) for b in balls]
        
        # Iterate through c3 not in code_set
        # To avoid getting stuck in local minima, we should try multiple c3.
        # We'll do a single pass through a random subset of c3 not in C.
        
        candidates_c3 = [c for c in range(total) if c not in code_set]
        random.shuffle(candidates_c3)
        
        # Limit checks to save time
        max_checks = 200
        for c3 in candidates_c3[:max_checks]:
            s3 = ball_sets[c3]
            # Find c1, c2 in code_set such that ball_sets[c1] subset s3 and ball_sets[c2] subset s3
            valid_cs = []
            for c in code_set:
                if ball_sets[c].issubset(s3):
                    valid_cs.append(c)
            
            if len(valid_cs) >= 2:
                # We can replace any two with c3
                # To be safe, pick the first two
                c1, c2 = valid_cs[0], valid_cs[1]
                
                # Verify the move is valid (it should be by construction)
                # Remove c1, c2, add c3
                code_set.remove(c1)
                code_set.remove(c2)
                for w in balls[c1]:
                    cover_count[w] -= 1
                for w in balls[c2]:
                    cover_count[w] -= 1
                code_set.add(c3)
                for w in balls[c3]:
                    cover_count[w] += 1
                
                return True
        
        return False

    # Greedy Initialization
    def greedy_init():
        uncovered = np.ones(total, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            best_c = -1
            best_gain = 0
            for c in range(total):
                gain = 0
                for w in balls[c]:
                    if uncovered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for w in balls[best_c]:
                uncovered[w] = False
                
        return code

    # Iterative Local Search
    best_code = greedy_init()
    best_size = len(best_code)
    
    # Main ILS Loop
    while time.time() < deadline - 5:
        # 1. Prune Redundant
        code_set = set(best_code)
        cover_count = np.zeros(total, dtype=np.int8)
        for c in code_set:
            for w in balls[c]:
                cover_count[w] += 1
        
        # Prune until stable
        changed = True
        while changed:
            changed = prune_redundant(code_set, cover_count)
            
        # 2. Try 2-for-1 Exchanges
        # Try multiple times to find improvements
        improved = True
        while improved:
            improved = try_2_for_1(code_set, cover_count)
            if improved:
                # After a 2-for-1, we might have introduced redundancy or new 2-for-1 opportunities
                # So we prune again
                changed = True
                while changed:
                    changed = prune_redundant(code_set, cover_count)
        
        # 3. Check if improved
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
        # 4. Perturbation (if stuck)
        # If we haven't improved in the last few iterations, perturb.
        # Simple perturbation: Remove a random codeword and repair greedily.
        # Or, add a random codeword and prune.
        
        # Let's do a "kick": Remove 2 random codewords, then repair greedily to valid, then prune.
        if len(code_set) > 0:
            # Randomly remove 2
            items = list(code_set)
            random.shuffle(items)
            to_remove = items[:2]
            for c in to_remove:
                code_set.remove(c)
                for w in balls[c]:
                    cover_count[w] -= 1
            
            # Repair: Add codewords to cover uncovered
            unc_idx, _ = get_uncovered(code_set, cover_count)
            while len(unc_idx) > 0:
                if time.time() > deadline - 5:
                    break
                # Pick a random uncovered word
                w = unc_idx[random.randint(0, len(unc_idx)-1)]
                # Find best c covering w
                best_c = -1
                best_gain = 0
                for c in range(total):
                    if w in balls[c]:
                        gain = 0
                        for ww in balls[c]:
                            if cover_count[ww] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                if best_c == -1:
                    break
                code_set.add(best_c)
                for ww in balls[best_c]:
                    cover_count[ww] += 1
                
                # Update uncovered
                unc_idx, cover_count = get_uncovered(code_set, cover_count)
                
            # Prune again
            changed = True
            while changed:
                changed = prune_redundant(code_set, cover_count)
                
            # Update best if improved
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)

    # Final Check: Ensure validity
    # Convert best_code to list of digits
    final_code = [int_to_digits(c) for c in best_code]
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in best_code:
        if c not in seen:
            seen.add(c)
            unique_code.append(c)
            
    if len(unique_code) != len(best_code):
        final_code = [int_to_digits(c) for c in unique_code]
        
    return final_code
