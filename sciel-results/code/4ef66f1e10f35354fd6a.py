import time
import math
import random

def construct():
        start_time = time.time()
        timeout = 115
        
        def calc_score(U):
            if not U or max(U) == 0:
                return 0.0
            max_u = max(U)
            sum_set = set()
            diff_set = set()
            # Optimization: use list of tuples? No, sets of ints are fine.
            for a in U:
                for b in U:
                    sum_set.add(a + b)
                    diff_set.add(a - b)
            n_sum = len(sum_set)
            n_diff = len(diff_set)
            if n_sum == 0:
                return 0.0
            ratio = n_diff / n_sum
            if ratio <= 1:
                return 1.0
            score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
            return score

        def greedy_golomb(n):
            """Generates a greedy Golomb ruler of size n."""
            U = [0]
            used_diffs = set()
            for _ in range(n - 1):
                candidate = U[-1] + 1
                while True:
                    # Check if adding candidate creates duplicate differences
                    valid = True
                    for u in U:
                        d = candidate - u
                        if d in used_diffs:
                            valid = False
                            break
                        # Note: we also need to check if d is 0? No, u != candidate.
                        # We add both d and -d? 
                        # Difference set contains a-b.
                        # If we add candidate, new diffs are candidate-u and u-candidate.
                        # So we need to check if candidate-u is in used_diffs OR u-candidate is in used_diffs.
                        # Since used_diffs should contain all diffs so far, we just check if candidate-u is in used_diffs.
                        # Wait, used_diffs should contain both positive and negative?
                        # Let's store all diffs in used_diffs.
                    if valid:
                        # Add new diffs to used_diffs
                        for u in U:
                            used_diffs.add(candidate - u)
                            used_diffs.add(u - candidate)
                        U.append(candidate)
                        break
                    candidate += 1
            return U

        best_set = greedy_golomb(5)
        best_score = calc_score(best_set)
        
        # Local search
        def local_search(U, max_iters=500):
            current_score = calc_score(U)
            U = list(U)
            for _ in range(max_iters):
                if time.time() - start_time > timeout:
                    break
                improved = False
                # Try moving each element
                for i in range(1, len(U)):
                    original = U[i]
                    # Try nearby values
                    for delta in range(-5, 6):
                        if delta == 0: continue
                        new_val = original + delta
                        if new_val < 0 or new_val == 0: continue
                        # Check uniqueness
                        if new_val in U: continue
                        U[i] = new_val
                        new_score = calc_score(U)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            break
                        U[i] = original
                    if improved:
                        break
                if not improved:
                    # Try adding an element
                    if len(U) < 20:
                        max_u = max(U)
                        # Try adding a random element
                        for _ in range(10):
                            new_val = random.randint(1, max_u + 10)
                            if new_val in U: continue
                            U.append(new_val)
                            new_score = calc_score(U)
                            if new_score > current_score:
                                current_score = new_score
                                improved = True
                                break
                            U.pop()
                        if improved:
                            continue
                    # Try removing an element
                    if len(U) > 3:
                        for i in range(1, len(U)):
                            original = U[i]
                            U.pop(i)
                            new_score = calc_score(U)
                            if new_score > current_score:
                                current_score = new_score
                                improved = True
                                break
                            U.insert(i, original)
                    if not improved:
                        break
            return U, current_score

        # Run local search on several initial guesses
        for n in range(4, 12):
            if time.time() - start_time > timeout:
                break
            init_U = greedy_golomb(n)
            # Also try random sets
            random.seed(n)
            random_U = [0] + sorted(random.sample(range(1, 50), n-1))
            
            for start_U in [init_U, random_U]:
                if time.time() - start_time > timeout:
                    break
                improved_U, score = local_search(start_U)
                if score > best_score:
                    best_score = score
                    best_set = improved_U

        return best_set
