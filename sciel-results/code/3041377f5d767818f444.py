import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def exact_score(U):
        """
        Compute exact score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        sums = set()
        diffs = set()
        for a in U_set:
            for b in U_set:
                sums.add(a + b)
                diffs.add(a - b)
        
        if not sums:
            return 0.0
            
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def fast_score_numpy(U):
        """
        Fast score using numpy for sets up to size ~500.
        """
        if len(U) < 2:
            return 0.0
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_U = int(U_arr.max())
        if max_U == 0:
            return 0.0
        
        # For very large sets, we might need to be careful with memory, 
        # but for n < 100, n^2 is 10000 which is fine.
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def greedy_constructive_search(time_limit):
        """
        Deterministic greedy construction.
        Start with {0}.
        Repeatedly add the candidate element x that maximizes the score of U ∪ {x}.
        Candidates are generated from a pool of "interesting" numbers.
        """
        current_set = [0]
        current_score = fast_score_numpy(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        # Candidate pool:
        # 1. Small integers
        # 2. Powers of 2, 3
        # 3. Random integers
        # We will maintain a list of candidates and pick the best one.
        
        max_val_bound = 200  # We won't go much higher initially, maybe expand later
        candidates = set()
        
        # Initial candidates
        for i in range(1, 50):
            candidates.add(i)
        for p in [2, 3, 4, 5]:
            v = 1
            while v < max_val_bound:
                candidates.add(v)
                v *= p
        for _ in range(100):
            candidates.add(random.randint(1, max_val_bound))
            
        while time.time() < time_limit:
            improved = False
            # Evaluate each candidate
            best_candidate = None
            best_candidate_score = current_score
            
            # To speed up, we can sample if the candidate list is huge, 
            # but 200-300 candidates is fine.
            # Sort candidates for determinism? Or just iterate.
            for x in candidates:
                if x in current_set:
                    continue
                new_set = current_set + [x]
                new_set = sorted(set(new_set))
                s = fast_score_numpy(new_set)
                if s > best_candidate_score + 1e-12:
                    best_candidate_score = s
                    best_candidate = x
            
            if best_candidate is not None:
                current_set.append(best_candidate)
                current_set = sorted(current_set)
                current_score = best_candidate_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                
                # Expand candidate pool if we haven't hit a good score
                if best_score < 1.02:
                    max_val_bound += 20
                    for _ in range(20):
                        candidates.add(random.randint(1, max_val_bound))
                    # Add neighbors of existing elements
                    for u in current_set:
                        for d in [-2, -1, 1, 2, 3, 5]:
                            nx = u + d
                            if nx > 0:
                                candidates.add(nx)
                
                # If we have a very high score, we might stop adding
                if len(current_set) > 150:
                    break
            else:
                # No improvement found in current pool
                # Try expanding the pool significantly
                if len(candidates) < 5000:
                    for _ in range(50):
                        candidates.add(random.randint(1, max_val_bound + 50))
                        candidates.add(random.randint(1, max(current_set) + 50))
                    # If still no improvement, we might be stuck.
                    # For now, break if we've tried a few times? 
                    # Let's just break for now to avoid wasting time, 
                    # as the local search will handle refinements.
                    break

        return best_set, best_score

    def local_search_refinement(initial_set, time_limit):
        """
        Coordinate descent / Hill Climbing.
        1. Try removing each element.
        2. Try replacing each element with a nearby value.
        """
        current_set = sorted(set(initial_set))
        current_score = fast_score_numpy(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        max_iter = 0
        while time.time() < time_limit and max_iter < 100:
            max_iter += 1
            improved = False
            
            # 1. Try removals
            n = len(current_set)
            if n > 1:
                # Evaluate all removals
                # Optimization: If n is large, this is O(n^3) roughly.
                # For n=100, 100 removals * 100^2 ops = 1M ops. Fast in numpy.
                best_removal = None
                best_removal_score = current_score
                for i in range(1, n):  # Don't remove 0
                    new_set = current_set[:i] + current_set[i+1:]
                    s = fast_score_numpy(new_set)
                    if s > best_removal_score + 1e-12:
                        best_removal_score = s
                        best_removal = i
                if best_removal is not None:
                    current_set = current_set[:best_removal] + current_set[best_removal+1:]
                    current_score = best_removal_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]

            # 2. Try replacements
            # For each element, try a few nearby values
            n = len(current_set)
            if n > 1:
                best_replace_idx = None
                best_replace_val = None
                best_replace_score = current_score
                
                # Limit replacements to save time if n is large
                # Sample indices if n > 20
                indices_to_check = list(range(1, n))
                if n > 20:
                    indices_to_check = random.sample(indices_to_check, min(10, len(indices_to_check)))
                
                for i in indices_to_check:
                    old_val = current_set[i]
                    # Try neighbors
                    for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -10, 10, -20, 20]:
                        new_val = old_val + delta
                        if new_val <= 0:
                            continue
                        if new_val in current_set:
                            continue
                        
                        # Create new set
                        # Efficiently update
                        # Since we are just checking, we can make a copy
                        new_set = current_set[:]
                        new_set[i] = new_val
                        new_set = sorted(set(new_set)) # Ensure uniqueness and sort
                        
                        s = fast_score_numpy(new_set)
                        if s > best_replace_score + 1e-12:
                            best_replace_score = s
                            best_replace_idx = i
                            best_replace_val = new_val
                
                if best_replace_idx is not None:
                    current_set[best_replace_idx] = best_replace_val
                    current_set = sorted(set(current_set))
                    current_score = best_replace_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]

            # 3. Try additions (if we removed some or if we want to grow)
            if n < 50:
                # Try adding a random candidate
                candidates = set()
                for u in current_set:
                    for d in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                        nx = u + d
                        if nx > 0:
                            candidates.add(nx)
                candidates.update(random.sample(range(1, max(current_set)+10), min(10, max(current_set)+10)))
                
                best_add_val = None
                best_add_score = current_score
                for x in candidates:
                    if x in current_set:
                        continue
                    new_set = current_set + [x]
                    s = fast_score_numpy(new_set)
                    if s > best_add_score + 1e-12:
                        best_add_score = s
                        best_add_val = x
                
                if best_add_val is not None:
                    current_set.append(best_add_val)
                    current_set = sorted(current_set)
                    current_score = best_add_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]

            if not improved:
                break

        return best_set, best_score

    # Strategy:
    # 1. Run Greedy Constructive Search for a portion of the time.
    # 2. Run Local Search Refinement on the result.
    # 3. If time remains, run a second round of Local Search or a different Greedy start.

    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Phase 1: Greedy Construction
    greedy_time = 60.0
    greedy_set, greedy_score = greedy_constructive_search(start_time + greedy_time)
    
    if greedy_score > best_overall_score:
        best_overall_score = greedy_score
        best_overall_set = greedy_set[:]

    # Phase 2: Local Search Refinement
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        # Split remaining time: 50% for LS on greedy result, 50% for LS on random starts?
        # Let's do LS on greedy result first.
        ls_time_1 = remaining_time * 0.5
        ls_set, ls_score = local_search_refinement(greedy_set, time.time() + ls_time_1)
        
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_set = ls_set[:]

        # If time permits, try a different random start for LS
        if time.time() < deadline - 10:
            # Generate a random start
            rand_size = random.randint(5, 15)
            rand_set = [0] + random.sample(range(1, 100), rand_size - 1)
            rand_set = sorted(set(rand_set))
            
            ls_time_2 = deadline - time.time() - 5
            if ls_time_2 > 5:
                ls_set2, ls_score2 = local_search_refinement(rand_set, time.time() + ls_time_2)
                if ls_score2 > best_overall_score:
                    best_overall_score = ls_score2
                    best_overall_set = ls_set2[:]

    # Final verification
    final_score = exact_score(best_overall_set)
    # Note: The evaluator uses exact score. 
    # If our fast score was optimistic, we might return a set that scores lower.
    # However, local search usually converges to a local optimum in the fast metric,
    # which correlates well with the exact metric.
    
    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    if U:
        max_U = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a+b)
                diffs.add(a-b)
        if sums:
            ratio = len(diffs)/len(sums)
            score = 1 + math.log(ratio)/math.log(2*max_U+1)
            print(f"Score: {score}")
