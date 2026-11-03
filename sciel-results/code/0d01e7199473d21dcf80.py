import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def score_set(U):
        """Compute the score for a set U."""
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def fast_score(U):
        """Faster score using numpy for moderate-sized sets."""
        import numpy as np
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_U = int(U_arr.max())
        if max_U == 0:
            return 0.0
        
        # Compute sumset using outer sum
        sums = U_arr[:, None] + U_arr[None, :]
        sumset_size = len(np.unique(sums))
        
        # Compute diffset using outer diff
        diffs = U_arr[:, None] - U_arr[None, :]
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def binary_independent_set(n, bit_limit=20):
        """
        Construct a set of n nonnegative integers where the binary representations
        are linearly independent over F_2 (i.e., no subset XORs to 0).
        This is achieved by using distinct bits.
        """
        # Use the first n distinct powers of 2
        # This gives a set where all sums are distinct (no carries in sumset sense for unique representation)
        # Actually, for binary basis, the sumset has size n(n+1)/2 (all subset sums are distinct)
        # and the diffset also has size n(n+1) (all differences are distinct in magnitude and sign)
        # Wait, if U = {0, 1, 2, 4, 8, ...}, then:
        # Sumset: all sums of distinct elements -> 2^n distinct sums (subset sums)
        # Diffset: all differences a-b. Since elements are powers of 2, differences are unique?
        # No, 4-2 = 2, but 2 is in the set. 8-4=4. 
        # Actually, for U = {0, 1, 2, 4, ..., 2^(n-1)}, the sumset is {0, ..., 2^n - 1} (all integers up to 2^n - 1).
        # Size of sumset = 2^n.
        # The diffset is {-(2^n-1), ..., 0, ..., 2^n-1}. Size = 2^(n+1) - 1.
        # Ratio = (2^(n+1)-1) / 2^n = 2 - 1/2^n.
        # Max U = 2^(n-1). 2*max_U + 1 = 2^n + 1.
        # Score = 1 + log(2 - 1/2^n) / log(2^n + 1).
        # As n grows, log(2) / (n * log(2)) = 1/n. So score -> 1.
        # This is not good for large n. We want the score to be high.
        # The score is 1 + log(ratio) / log(2*max_U + 1).
        # We want ratio to be large and max_U to be small.
        
        # Let's try a different algebraic structure.
        # What if we use a Golomb ruler?
        # Golomb ruler: all differences are distinct.
        # If U is a Golomb ruler, |U-U| = n(n-1) + 1 (including 0).
        # What about |U+U|?
        # For a Golomb ruler, sums are not necessarily distinct.
        # Example: {0, 1, 4}. Diffs: 0, 1, 3, 4. (0-0, 0-1, 0-4, 1-0, 1-4, 4-0, 4-1). Unique diffs: 0, 1, 3, 4. Size 4.
        # Sums: 0, 1, 4, 2, 5, 8. Unique sums: 0, 1, 2, 4, 5, 8. Size 6.
        # Ratio = 4/6 = 0.66. Log(0.66) < 0. Score < 1.
        # We want ratio > 1, so |U-U| > |U+U|.
        # This means we want many distinct differences and few distinct sums.
        
        # To have few sums, we want many collisions in a+b.
        # To have many differences, we want few collisions in a-b.
        
        # Consider U = {0, 1, M, M+1} for large M.
        # Sums: 0, 1, M, M+1, 2, 2M, 2M+1, 2M+2. 
        # If M is large, these are all distinct. Size 8.
        # Diffs: 0, 1, M, M+1, -1, -M, -(M+1), 2M, 2M+1, -2M, -(2M+1).
        # Unique diffs: 0, 1, M, M+1, 2M, 2M+1 and their negatives.
        # If M is large, size ~ 11.
        # Ratio ~ 11/8 > 1.
        # Max U = M+1. 2*max_U+1 ~ 2M.
        # Score = 1 + log(11/8) / log(2M).
        # As M grows, score -> 1.
        
        # We need to balance M and n.
        
        # Let's try a set with small diameter but many elements.
        # But if diameter is small, max_U is small, denominator is small, which is good.
        # But if elements are close, sums and diffs overlap a lot.
        
        # Let's go back to the quadratic set idea from the inspiration, but maybe refine it.
        # U = {i^2 for i in range(n)}.
        # Sums: i^2 + j^2.
        # Diffs: i^2 - j^2 = (i-j)(i+j).
        # For n=10, max_U = 81. 2*max_U+1 = 163. log(163) ~ 5.1.
        # Let's compute score for quadratic sets for various n.
        
        return [i * i for i in range(n)]

    def greedy_golomb(n):
        """
        Construct a Golomb ruler with n marks using a greedy algorithm.
        """
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        marks = [0]
        diffs = set()
        
        for i in range(1, n):
            best_pos = None
            max_search = 100 + i * i * 2 
            
            for pos in range(marks[-1] + 1, max_search):
                valid = True
                for m in marks:
                    d = abs(pos - m)
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    best_pos = pos
                    break
            
            if best_pos is None:
                best_pos = marks[-1] + i * i + 1
            
            marks.append(best_pos)
            for m in marks[:-1]:
                diffs.add(abs(best_pos - m))
        
        return marks

    def local_search(initial_set, time_limit):
        """
        Perform local search to improve the score.
        """
        current_set = sorted(set(initial_set))
        current_score = fast_score(current_set) if len(current_set) > 20 else score_set(current_set)
        best_set = current_set[:]
        best_score = current_score
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                break
                
            if len(current_set) <= 1:
                break
                
            indices = list(range(1, len(current_set)))
            idx = random.choice(indices)
            old_val = current_set[idx]
            
            improved = False
            # Try a few perturbations
            for _ in range(20):
                strategy = random.random()
                new_val = old_val
                
                if strategy < 0.4:
                    # Small shift
                    shift = random.randint(-10, 10)
                    new_val = old_val + shift
                elif strategy < 0.7:
                    # Random replacement within bounds
                    max_val = max(current_set)
                    new_val = random.randint(1, max_val + 50)
                else:
                    # Large jump
                    new_val = random.randint(1, 500)
                
                if new_val == 0 or new_val in current_set:
                    continue
                
                # Apply perturbation
                candidate_set = current_set[:]
                candidate_set[idx] = new_val
                candidate_set = sorted(set(candidate_set))
                
                new_score = fast_score(candidate_set) if len(candidate_set) > 20 else score_set(candidate_set)
                
                if new_score > current_score:
                    current_set = candidate_set
                    current_score = new_score
                    no_improve_count = 0
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                    break # Move to next iteration with new current
                else:
                    # Keep current_set unchanged
                    pass
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

        return best_set, best_score

    # Strategy:
    # 1. Try quadratic sets for n=2 to 100.
    # 2. Try triangular sets.
    # 3. Try Golomb rulers.
    # 4. Run local search on the best candidates.

    best_overall_score = 0.0
    best_overall_set = [0]

    # Phase 1: Algebraic constructions
    # Quadratic
    for n in range(2, 150):
        if time.time() > deadline:
            break
        U = binary_independent_set(n)
        s = fast_score(U) if len(U) > 20 else score_set(U)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_set = U[:]

    # Triangular
    for n in range(2, 150):
        if time.time() > deadline:
            break
        U = [i * (i + 1) // 2 for i in range(n)]
        s = fast_score(U) if len(U) > 20 else score_set(U)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_set = U[:]

    # Cubic
    for n in range(2, 100):
        if time.time() > deadline:
            break
        U = [i * i * i for i in range(n)]
        s = fast_score(U) if len(U) > 20 else score_set(U)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_set = U[:]

    # Phase 2: Golomb rulers
    for n in range(2, 50):
        if time.time() > deadline:
            break
        U = greedy_golomb(n)
        s = fast_score(U) if len(U) > 20 else score_set(U)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_set = U[:]

    # Phase 3: Local search on best candidates
    # Take the top few candidates from Phase 1 and 2 and refine them
    candidates = [best_overall_set[:]]
    
    # Add a few diverse starting points
    for n in [10, 20, 30, 40, 50]:
        candidates.append(binary_independent_set(n))
        candidates.append(greedy_golomb(n))
        
    # Run local search on each candidate for a short time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        time_per_candidate = remaining_time / max(1, len(candidates))
        for cand in candidates:
            if time.time() > deadline - 2:
                break
            sub_deadline = time.time() + time_per_candidate
            improved_set, improved_score = local_search(cand, sub_deadline)
            if improved_score > best_overall_score:
                best_overall_score = improved_score
                best_overall_set = improved_set[:]

    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    print(f"Score: {score_set(U) if False else 0}") # Just for debugging if run directly, but score_set is not defined globally in this scope if not careful.
    # Actually, let's just print the set.
