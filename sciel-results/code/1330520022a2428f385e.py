import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Parameters for the specific task: q=3, n=6, radius=1
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets (neighbors including self)
    # coverage[i] is a set of indices j such that dist(i, j) <= radius
    # For efficiency in ILP, we need the constraint matrix A.
    # A[j, i] = 1 if i covers j.
    
    # Since total_words = 729, the matrix is 729x729.
    # We can build it as a sparse matrix or dense.
    # scipy.sparse is better for ILP.
    
    try:
        import scipy.sparse as sp
        import scipy.optimize as opt
        
        # Build the constraint matrix A
        # A is total_words x total_words
        # A[j, i] = 1 if word i is within distance radius of word j
        
        # It's easier to iterate over i (codeword candidates) and mark what they cover
        # But A is defined by rows j (constraints) and cols i (variables).
        # So for each i, we find all j covered by i, and set A[j, i] = 1.
        
        rows = []
        cols = []
        data = []
        
        for i, w_i in enumerate(all_words):
            # Find all words w_j within distance radius of w_i
            # For radius 1, this is w_i itself and all words differing in exactly 1 position.
            # Count: 1 + n * (q - 1) = 1 + 6 * 2 = 13.
            
            covered = [i]
            if radius == 1:
                for pos in range(n):
                    orig = w_i[pos]
                    for nv in range(q):
                        if nv != orig:
                            lst = list(w_i)
                            lst[pos] = nv
                            j = word_to_idx[tuple(lst)]
                            covered.append(j)
            
            for j in covered:
                rows.append(j)
                cols.append(i)
                data.append(1)
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
        
        # Objective: minimize sum(x_i)
        c = np.ones(total_words)
        
        # Bounds: x_i in [0, 1]
        # Symmetry breaking: fix x_0 = 1 (the all-zero word)
        # This is valid because the Hamming graph is vertex-transitive.
        
        # We can handle fixed variables by adjusting bounds.
        # x_0 = 1
        # Others: 0 <= x_i <= 1
        
        bounds = []
        for i in range(total_words):
            if i == 0:
                bounds.append((1, 1))
            else:
                bounds.append((0, 1))
        
        # Constraints: A @ x >= b, where b is vector of 1s
        # scipy.optimize.milp expects constraints in the form:
        # A @ x <= b or A @ x >= b or A @ x == b
        # We have A @ x >= 1
        
        b = np.ones(total_words)
        
        # Define constraints
        # In scipy.optimize.milp, constraints are defined with LinearConstraint
        # A @ x >= 1  =>  -A @ x <= -1
        
        constraint = opt.LinearConstraint(-A, lb=-np.inf, ub=-b)
        
        # Solve
        # integrality: 1 for integer variables
        integrality = np.ones(total_words)
        
        # Run solver
        # Use a timeout to prevent hanging
        # milp doesn't have a direct timeout parameter in all versions, 
        # but we can check time after.
        # Actually, scipy.optimize.milp uses HiGHS or CBC. 
        # We should wrap it in a thread or just hope it's fast.
        # 729 variables is small for HiGHS.
        
        res = opt.milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # x should be 0 or 1
            selected_indices = [i for i in range(total_words) if x[i] > 0.5]
            
            # Verify coverage just in case
            # (Optional but good for safety)
            # selected_set = set(selected_indices)
            # covered = set()
            # for i in selected_indices:
            #    # add all neighbors
            #    # ...
            
            return [list(all_words[i]) for i in selected_indices]
        
        else:
            # Fallback if ILP fails or is not optimal
            pass
            
    except Exception as e:
        pass

    # Fallback: Greedy + Local Search
    # This is the logic from the previous program, refined.
    
    # Precompute coverage masks as Python ints
    coverage_masks = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        lst = list(w)
                        lst[pos] = nv
                        j = word_to_idx[tuple(lst)]
                        mask |= (1 << j)
        coverage_masks[i] = mask

    all_ones = (1 << total_words) - 1

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def greedy_cover(seed_random):
        uncovered = all_ones
        chosen = []
        chosen_set = set()
        indices = list(range(total_words))
        rng = random.Random(seed_random)
        rng.shuffle(indices)

        while uncovered and time.time() < deadline - 5:
            best_count = -1
            best_candidates = []
            for i in indices:
                if i in chosen_set:
                    continue
                common = coverage_masks[i] & uncovered
                cnt = popcount(common)
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [i]
                elif cnt == best_count:
                    best_candidates.append(i)
            if not best_candidates:
                break
            pick = rng.choice(best_candidates)
            chosen.append(pick)
            chosen_set.add(pick)
            uncovered &= ~coverage_masks[pick]

        return chosen, chosen_set

    def local_search(chosen, chosen_set, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            k = len(chosen)
            if k == 0:
                break

            masks = [coverage_masks[c] for c in chosen]
            total_cov = 0
            for m in masks:
                total_cov |= m

            if total_cov != all_ones:
                break

            # Try removal
            prefix_or = [0] * (k + 1)
            for i in range(k):
                prefix_or[i + 1] = prefix_or[i] | masks[i]
            suffix_or = [0] * (k + 1)
            for i in range(k - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | masks[i]

            removable_indices = []
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                if without_i == all_ones:
                    removable_indices.append(i)

            if removable_indices:
                removed_idx = removable_indices[0]
                c = chosen.pop(removed_idx)
                chosen_set.discard(c)
                improved = True
                continue

            # Try 2-for-1 swap (remove 2, add 1)
            # Find pairs of "important" codewords
            missing_if_remove = [0] * k
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                missing_if_remove[i] = all_ones & ~without_i

            important = [i for i in range(k) if missing_if_remove[i] != 0]
            
            if len(important) < 2:
                break

            found_swap = False
            # Limit checks
            max_pairs = 200
            pair_count = 0
            for idx_i in range(len(important)):
                if time.time() > deadline - 1:
                    break
                i = important[idx_i]
                for idx_j in range(idx_i + 1, len(important)):
                    if pair_count >= max_pairs:
                        break
                    pair_count += 1
                    j = important[idx_j]
                    
                    # Compute OR without i and j
                    or_without_ij = 0
                    for m_idx in range(k):
                        if m_idx != i and m_idx != j:
                            or_without_ij |= masks[m_idx]
                    
                    missing_ij = all_ones & ~or_without_ij
                    if missing_ij == 0:
                        # Both redundant
                        if i > j: i, j = j, i
                        chosen.pop(i)
                        chosen_set.discard(chosen[i])
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Find replacement
                        for cp in range(total_words):
                            if cp in chosen_set:
                                continue
                            cm = coverage_masks[cp]
                            if (missing_ij & ~cm) == 0:
                                # Replace i, j with cp
                                if i > j: i, j = j, i
                                word_i = chosen[i]
                                word_j = chosen[j]
                                chosen.pop(j)
                                chosen_set.discard(word_j)
                                chosen.pop(i)
                                chosen_set.discard(word_i)
                                chosen.append(cp)
                                chosen_set.add(cp)
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                if found_swap:
                    break
            
            if not found_swap:
                break

        return chosen, chosen_set

    best_chosen = None
    best_size = total_words + 1

    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        seed = 42 + r * 7
        chosen, chosen_set = greedy_cover(seed)
        
        if time.time() > deadline - 5:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
            continue

        chosen, chosen_set = local_search(chosen, chosen_set, deadline - 2)

        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)

    if best_chosen is None:
        chosen, _ = greedy_cover(0)
        best_chosen = chosen

    return [list(all_words[i]) for i in best_chosen]
