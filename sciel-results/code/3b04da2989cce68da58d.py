import time
import random
import sys
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific task: K_3(6,1)
    # The evaluator passes q, n, radius.
    # If not the specific case, fall back to a simple greedy or direct sum.
    if not (q == 3 and n == 6 and radius == 1):
        # Fallback: Direct sum of tetracode if q=3, n>=4
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy for other cases (slow, but safe)
            all_words = list(itertools.product(range(q), repeat=n))
            code = []
            covered = set()
            for w in all_words:
                if w not in covered:
                    code.append(list(w))
                    for i in range(n):
                        for v in range(q):
                            if v == w[i]:
                                continue
                            new_w = list(w)
                            new_w[i] = v
                            covered.add(tuple(new_w))
                    covered.add(w)
            return code

    # Specific case: q=3, n=6, radius=1
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    total = q ** n
    powers = [q ** p for p in range(n)]

    # Precompute neighbors: neighbors[i] is a list of indices j such that dist(i,j) <= 1
    # There are 1 + n*(q-1) = 1 + 6*2 = 13 neighbors per word.
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        digits = [(i // powers[p]) % q for p in range(n)]
        neighbors[i].append(i)
        for p in range(n):
            for alt in range(q):
                if alt != digits[p]:
                    new_i = i + (alt - digits[p]) * powers[p]
                    neighbors[i].append(new_i)

    # Convert to sets for fast membership testing
    neighbor_sets = [set(nei) for nei in neighbors]

    # Phase 1: Greedy Set Cover to get an initial code
    uncovered = np.ones(total, dtype=np.int8)
    code_indices = set()
    counts = np.zeros(total, dtype=np.int32)

    # Precompute neighbor arrays as numpy for faster scoring
    neighbor_arrays = [np.array(nei, dtype=np.int32) for nei in neighbors]

    while uncovered.sum() > 0:
        if time.time() > deadline:
            break

        best_candidate = -1
        best_score = 0

        # Use numpy to score all candidates at once
        for c in range(total):
            score = np.sum(uncovered[neighbor_arrays[c]])
            if score > best_score:
                best_score = score
                best_candidate = c

        if best_score == 0:
            break

        code_indices.add(best_candidate)
        counts[neighbor_arrays[best_candidate]] += 1
        uncovered[neighbor_arrays[best_candidate]] = 0

    # Check if we have a valid cover
    if uncovered.sum() != 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_size = len(code_indices)
    best_code = list(code_indices)

    def can_remove(c_idx):
        return np.all(counts[neighbor_arrays[c_idx]] > 1)

    def get_max_removability_score():
        max_score = 0
        for c in code_indices:
            min_count = np.min(counts[neighbor_arrays[c]])
            if min_count > max_score:
                max_score = min_count
        return max_score

    # Local Search Loop with Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995

    # Helper for smooth surrogate: log-sum-exp based soft-min
    def soft_min(values, beta=10.0):
        if len(values) == 0:
            return 0
        max_v = np.max(values)
        if max_v == 0:
            return 0
        exp_vals = np.exp(beta * (values - max_v))
        log_sum = np.log(np.sum(exp_vals))
        return max_v - np.log(len(exp_vals)) / beta

    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        for c in list(code_indices):
            if can_remove(c):
                counts[neighbor_arrays[c]] -= 1
                code_indices.remove(c)
                removed_any = True
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                break

        if removed_any:
            continue

        # Phase 2: Swap with Simulated Annealing
        current_score = get_max_removability_score()

        # Identify critical words
        critical_mask = (counts == 1)
        critical_words = np.where(critical_mask)[0]

        if len(critical_words) == 0:
            break

        found_improvement = False
        code_list = list(code_indices)

        for c_out in code_list:
            if time.time() > deadline:
                break

            # Find promising c_in candidates
            # Score c_in by how many critical words it covers that c_out doesn't
            c_out_neighbors_arr = neighbor_arrays[c_out]
            c_out_neighbors_set = neighbor_sets[c_out]
            
            # Vectorized scoring for all c_in
            candidate_scores = []
            for c_in in range(total):
                if c_in in code_indices:
                    continue
                c_in_neighbors_arr = neighbor_arrays[c_in]
                # Count critical words covered by c_in but not by c_out
                score = np.sum(critical_mask[c_in_neighbors_arr] & ~np.isin(c_in_neighbors_arr, c_out_neighbors_arr))
                if score > 0:
                    candidate_scores.append((score, c_in))

            if not candidate_scores:
                continue

            # Sort by score and take top candidates
            candidate_scores.sort(reverse=True)
            top_candidates = [c for _, c in candidate_scores[:30]]

            # Shuffle for diversity
            random.shuffle(top_candidates)

            for c_in in top_candidates:
                if time.time() > deadline:
                    break

                # Check validity of swap
                c_out_neighbors_arr = neighbor_arrays[c_out]
                c_in_neighbors_arr = neighbor_arrays[c_in]
                
                # Words covered by c_out but not c_in must have count >= 2
                only_in_out = c_out_neighbors_arr[~np.isin(c_out_neighbors_arr, c_in_neighbors_arr)]
                if len(only_in_out) > 0:
                    if np.min(counts[only_in_out]) < 2:
                        continue

                # Apply swap temporarily
                counts[neighbor_arrays[c_out]] -= 1
                counts[neighbor_arrays[c_in]] += 1
                code_indices.remove(c_out)
                code_indices.add(c_in)

                new_score = get_max_removability_score()

                # Use smooth surrogate for acceptance
                soft_curr = soft_min(counts[neighbor_arrays[c_out]], beta=5.0)
                soft_new = soft_min(counts[neighbor_arrays[c_in]], beta=5.0)

                accept = False
                if new_score > current_score:
                    accept = True
                elif new_score == current_score:
                    # Neutral move: accept if it improves the soft surrogate
                    if soft_new > soft_curr + 0.1:
                        accept = True
                    else:
                        accept = random.random() < 0.3
                else:
                    # Worse discrete score: accept with low probability (annealing)
                    delta = current_score - new_score
                    accept_prob = np.exp(-delta / T) * 0.5
                    accept = random.random() < accept_prob

                if accept:
                    found_improvement = True
                    # Try immediate removal
                    for c in list(code_indices):
                        if can_remove(c):
                            counts[neighbor_arrays[c]] -= 1
                            code_indices.remove(c)
                            if len(code_indices) < best_size:
                                best_size = len(code_indices)
                                best_code = list(code_indices)
                            break
                    break
                else:
                    # Revert swap
                    counts[neighbor_arrays[c_in]] -= 1
                    counts[neighbor_arrays[c_out]] += 1
                    code_indices.remove(c_in)
                    code_indices.add(c_out)

            if found_improvement:
                break

        if not found_improvement:
            T *= cooling_rate
            if T < 0.05:
                # Perturbation: apply a random valid swap
                perturbed = False
                for _ in range(20):
                    c_out = random.choice(list(code_indices))
                    c_in = random.randint(0, total - 1)
                    if c_in in code_indices:
                        continue
                    
                    c_out_neighbors_arr = neighbor_arrays[c_out]
                    c_in_neighbors_arr = neighbor_arrays[c_in]
                    
                    # Words covered by c_out but not c_in must have count >= 2
                    only_in_out = c_out_neighbors_arr[~np.isin(c_out_neighbors_arr, c_in_neighbors_arr)]
                    if len(only_in_out) > 0:
                        if np.min(counts[only_in_out]) < 2:
                            continue
                    
                    counts[neighbor_arrays[c_out]] -= 1
                    counts[neighbor_arrays[c_in]] += 1
                    code_indices.remove(c_out)
                    code_indices.add(c_in)
                    perturbed = True
                    break
                if not perturbed:
                    # Restore best and break
                    code_indices = set(best_code)
                    counts = np.zeros(total, dtype=np.int32)
                    for c in code_indices:
                        counts[neighbor_arrays[c]] += 1
                    break
                T = 1.0

    # Convert best_code indices back to words
    return [list((best_code[i] // powers[p]) % q for p in range(n)) for i in range(len(best_code))]
