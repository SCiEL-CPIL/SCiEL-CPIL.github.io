"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s/90s limit

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square_from_dense(dense):
        if len(dense) == 0:
            return np.array([0], dtype=np.int64)
        return np.convolve(dense, dense, mode='full')

    def terms_to_dense(terms):
        if not terms:
            return np.array([0], dtype=np.int64)
        max_deg = max(pos for pos, _ in terms)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in terms:
            dense[pos] = val
        return dense

    def evaluate(terms):
        if len(terms) != 12:
            return 1000
        positions = [pos for pos, _ in terms]
        if len(set(positions)) != 12:
            return 1000
        for pos, val in terms:
            if val == 0 or abs(val) > 64 or pos > 40:
                return 1000
        dense = terms_to_dense(terms)
        sq = poly_square_from_dense(dense)
        return count_nonzero(sq)

    def sum_collision_count(positions):
        """Count distinct sum positions p_i + p_j for i <= j"""
        sums = set()
        for i in range(12):
            for j in range(i, 12):
                sums.add(positions[i] + positions[j])
        return len(sums)

    def generate_structural_candidates():
        """Generate position sets with high sum-collision density."""
        candidates = []
        
        # Arithmetic progressions
        for start in range(0, 41):
            for step in range(1, 6):
                pos = [start + i * step for i in range(12)]
                if pos[-1] <= 40:
                    candidates.append(tuple(pos))
        
        # Two interleaved arithmetic progressions
        for s1 in range(0, 25):
            for s2 in range(s1 + 1, 30):
                for step in range(1, 5):
                    pos = []
                    for i in range(6):
                        pos.append(s1 + i * step)
                        pos.append(s2 + i * step)
                    pos.sort()
                    if len(set(pos)) == 12 and pos[-1] <= 40:
                        candidates.append(tuple(pos))
        
        # Three interleaved APs of 4
        for s1 in range(0, 20):
            for s2 in range(s1 + 1, 25):
                for s3 in range(s2 + 1, 30):
                    for step in range(1, 4):
                        pos = []
                        for i in range(4):
                            pos.extend([s1 + i * step, s2 + i * step, s3 + i * step])
                        pos.sort()
                        if len(set(pos)) == 12 and pos[-1] <= 40:
                            candidates.append(tuple(pos))
        
        # Clusters
        for n_clusters in [2, 3, 4, 6]:
            if 12 % n_clusters != 0:
                continue
            cluster_size = 12 // n_clusters
            for _ in range(300):
                bases = sorted(random.sample(range(0, 41 - cluster_size + 1), n_clusters))
                pos = []
                for b in bases:
                    offsets = random.sample(range(0, cluster_size + 3), cluster_size)
                    pos.extend(b + o for o in offsets)
                pos = sorted(set(pos))
                if len(pos) == 12 and pos[-1] <= 40:
                    candidates.append(tuple(pos))
        
        # Random sets
        for _ in range(1000):
            pos = tuple(sorted(random.sample(range(41), 12)))
            candidates.append(pos)
        
        # Deduplicate and sort by collision count
        candidates = list(set(candidates))
        candidates.sort(key=lambda p: sum_collision_count(p))
        
        return candidates

    def find_best_values_for_positions(positions, time_limit=1.0):
        """For fixed positions, find values minimizing square nonzero count."""
        start_t = time.time()
        best_score = 1000
        best_values = None
        
        # Build sum-to-pairs mapping
        sum_to_pairs = defaultdict(list)
        for i in range(12):
            for j in range(i, 12):
                s = positions[i] + positions[j]
                sum_to_pairs[s].append((i, j))
        
        # Try random value assignments with bias toward small values and signs
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 80000:
            n_trials += 1
            # Use small values to increase chance of cancellation
            vals = []
            for _ in range(12):
                sign = random.choice([1, -1])
                # Bias toward small values
                r = random.random()
                if r < 0.35:
                    mag = 1
                elif r < 0.65:
                    mag = random.randint(1, 3)
                elif r < 0.85:
                    mag = random.randint(1, 8)
                else:
                    mag = random.randint(1, 64)
                vals.append(sign * mag)
            
            dense = np.zeros(max(positions) + 1, dtype=np.int64)
            for p, v in zip(positions, vals):
                dense[p] = v
            sq = np.convolve(dense, dense, mode='full')
            score = count_nonzero(sq)
            if score < best_score:
                best_score = score
                best_values = vals[:]
                if best_score <= 12:
                    break
        
        return best_score, best_values

    def local_search_from(terms, time_limit=2.0):
        """Local search starting from given terms."""
        start_t = time.time()
        current_terms = list(terms)
        current_score = evaluate(current_terms)
        
        def neighbor(terms):
            terms = list(terms)
            move_type = random.random()
            if move_type < 0.35:
                # Change position of one term
                idx = random.randint(0, 11)
                old_pos = terms[idx][0]
                new_pos = random.randint(0, 40)
                if new_pos == old_pos:
                    return terms
                for i, (pos, val) in enumerate(terms):
                    if i != idx and pos == new_pos:
                        return terms
                terms[idx] = (new_pos, terms[idx][1])
            elif move_type < 0.65:
                # Change value of one term
                idx = random.randint(0, 11)
                new_val = random.choice([1, -1]) * random.randint(1, 64)
                terms[idx] = (terms[idx][0], new_val)
            elif move_type < 0.85:
                # Change sign of one term
                idx = random.randint(0, 11)
                terms[idx] = (terms[idx][0], -terms[idx][1])
            else:
                # Swap positions of two terms
                i, j = random.sample(range(12), 2)
                terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])
            return terms
        
        while time.time() - start_t < time_limit:
            improved = False
            for _ in range(100):
                if time.time() - start_t > time_limit:
                    break
                new_terms = neighbor(current_terms)
                new_score = evaluate(new_terms)
                if new_score < current_score:
                    current_terms = new_terms
                    current_score = new_score
                    improved = True
                    if current_score <= 12:
                        return current_terms, current_score
            if not improved:
                break
        
        return current_terms, current_score

    def generate_factor_candidates():
        """Generate candidate factor pairs (A, B) where P = A*B has 12 terms."""
        candidates = []
        
        # Factorization 3x4: A has 3 terms, B has 4 terms
        # A positions: 3 values from 0..40
        # B positions: 4 values from 0..40
        # P positions: all sums a_i + b_j, must be 12 distinct
        
        # Try small position ranges for factors
        max_pos_a = 15
        max_pos_b = 25
        
        # Systematic enumeration of factor position sets
        for n_a in [2, 3, 4, 6]:
            n_b = 12 // n_a
            if n_a * n_b != 12:
                continue
            
            # Generate random position sets for A and B
            for _ in range(500):
                pos_a = sorted(random.sample(range(0, max_pos_a + 1), n_a))
                pos_b = sorted(random.sample(range(0, max_pos_b + 1), n_b))
                
                # Compute product positions
                prod_pos = set()
                for pa in pos_a:
                    for pb in pos_b:
                        prod_pos.add(pa + pb)
                
                if len(prod_pos) == 12 and max(prod_pos) <= 40:
                    candidates.append((tuple(pos_a), tuple(pos_b)))
        
        # Also try structured factor positions
        # A as arithmetic progression
        for start_a in range(0, 10):
            for step_a in range(1, 4):
                for n_a in [3, 4, 6]:
                    pos_a = tuple(start_a + i * step_a for i in range(n_a))
                    if pos_a[-1] > max_pos_a:
                        continue
                    n_b = 12 // n_a
                    for start_b in range(0, 15):
                        for step_b in range(1, 4):
                            pos_b = tuple(start_b + i * step_b for i in range(n_b))
                            if pos_b[-1] > max_pos_b:
                                continue
                            
                            prod_pos = set()
                            for pa in pos_a:
                                for pb in pos_b:
                                    prod_pos.add(pa + pb)
                            
                            if len(prod_pos) == 12 and max(prod_pos) <= 40:
                                candidates.append((pos_a, pos_b))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        return candidates

    def optimize_factor_values(pos_a, pos_b, time_limit=0.5):
        """Find values for factors A and B that minimize the square of P = A*B."""
        start_t = time.time()
        best_score = 1000
        best_vals_a = None
        best_vals_b = None
        
        n_a = len(pos_a)
        n_b = len(pos_b)
        
        # Try template values first
        templates_a = [
            [1] * n_a,
            [1 if i % 2 == 0 else -1 for i in range(n_a)],
        ]
        templates_b = [
            [1] * n_b,
            [1 if i % 2 == 0 else -1 for i in range(n_b)],
        ]
        
        def evaluate_factor_pair(vals_a, vals_b):
            # Compute P = A * B
            dense_a = np.zeros(max(pos_a) + 1, dtype=np.int64)
            for p, v in zip(pos_a, vals_a):
                dense_a[p] = v
            
            dense_b = np.zeros(max(pos_b) + 1, dtype=np.int64)
            for p, v in zip(pos_b, vals_b):
                dense_b[p] = v
            
            dense_p = np.convolve(dense_a, dense_b, mode='full')
            sq_p = np.convolve(dense_p, dense_p, mode='full')
            return count_nonzero(sq_p)
        
        # Try template combinations
        for ta in templates_a:
            for tb in templates_b:
                score = evaluate_factor_pair(ta, tb)
                if score < best_score:
                    best_score = score
                    best_vals_a = ta[:]
                    best_vals_b = tb[:]
                if best_score <= 12:
                    return best_score, best_vals_a, best_vals_b
        
        # Random search
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 5000:
            n_trials += 1
            vals_a = []
            for _ in range(n_a):
                sign = random.choice([1, -1])
                r = random.random()
                if r < 0.4:
                    mag = 1
                elif r < 0.7:
                    mag = random.randint(1, 3)
                else:
                    mag = random.randint(1, 16)
                vals_a.append(sign * mag)
            
            vals_b = []
            for _ in range(n_b):
                sign = random.choice([1, -1])
                r = random.random()
                if r < 0.4:
                    mag = 1
                elif r < 0.7:
                    mag = random.randint(1, 3)
                else:
                    mag = random.randint(1, 16)
                vals_b.append(sign * mag)
            
            score = evaluate_factor_pair(vals_a, vals_b)
            if score < best_score:
                best_score = score
                best_vals_a = vals_a[:]
                best_vals_b = vals_b[:]
                if best_score <= 12:
                    return best_score, best_vals_a, best_vals_b
        
        return best_score, best_vals_a, best_vals_b

    def factor_to_terms(pos_a, pos_b, vals_a, vals_b):
        """Convert factor representation to full polynomial terms."""
        dense_a = np.zeros(max(pos_a) + 1, dtype=np.int64)
        for p, v in zip(pos_a, vals_a):
            dense_a[p] = v
        
        dense_b = np.zeros(max(pos_b) + 1, dtype=np.int64)
        for p, v in zip(pos_b, vals_b):
            dense_b[p] = v
        
        dense_p = np.convolve(dense_a, dense_b, mode='full')
        
        # Extract nonzero terms
        terms = []
        for i, v in enumerate(dense_p):
            if v != 0:
                terms.append((i, int(v)))
        
        return terms

    def main_search():
        best_terms = None
        best_score = 1000
        
        # Phase 1: Structural search over high-collision position sets
        candidates = generate_structural_candidates()
        
        # Try top candidates by collision count
        for idx, positions in enumerate(candidates[:200]):
            if time.time() > deadline - 15:
                break
            
            remaining = deadline - time.time()
            n_remaining = min(len(candidates) - idx, 200)
            if n_remaining > 0:
                time_per = max(0.02, remaining / max(n_remaining, 1))
            else:
                time_per = 0.2
            
            score, values = find_best_values_for_positions(list(positions), time_limit=time_per)
            
            if values is not None and score < best_score:
                best_score = score
                best_terms = list(zip(positions, values))
                if best_score <= 12:
                    break
        
        # Phase 2: Factor-based search
        if time.time() < deadline - 10:
            factor_candidates = generate_factor_candidates()
            # Sort by expected collision density (heuristic: prefer factors with more internal collisions)
            factor_candidates.sort(key=lambda x: len(x[0]) * len(x[1]))
            
            for idx, (pos_a, pos_b) in enumerate(factor_candidates[:100]):
                if time.time() > deadline - 8:
                    break
                
                remaining = deadline - time.time() - 8
                time_per = max(0.05, remaining / max(100 - idx, 1))
                
                score, vals_a, vals_b = optimize_factor_values(pos_a, pos_b, time_limit=time_per)
                
                if vals_a is not None and vals_b is not None and score < best_score:
                    best_score = score
                    terms = factor_to_terms(pos_a, pos_b, vals_a, vals_b)
                    if len(terms) == 12:
                        best_terms = terms
                        if best_score <= 12:
                            break
        
        # Phase 3: Local search refinement from best result
        if best_terms is not None and time.time() < deadline - 3:
            refined_terms, refined_score = local_search_from(best_terms, time_limit=min(3.0, max(0.5, deadline - time.time() - 2)))
            if refined_score < best_score:
                best_score = refined_score
                best_terms = refined_terms
        
        # Phase 4: Additional random restarts with local search
        while time.time() < deadline - 2:
            # Random initial
            positions = random.sample(range(41), 12)
            values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
            terms = list(zip(sorted(positions), values))
            
            # Quick local search
            ls_time = min(0.8, max(0.2, deadline - time.time() - 2))
            refined_terms, refined_score = local_search_from(terms, time_limit=ls_time)
            
            if refined_score < best_score:
                best_score = refined_score
                best_terms = refined_terms
                if best_score <= 12:
                    break
        
        return best_terms, best_score

    # Run main search
    best_terms, best_score = main_search()
    
    # Fallback if no valid terms found
    if best_terms is None or len(best_terms) != 12:
        positions = random.sample(range(41), 12)
        values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
        best_terms = list(zip(sorted(positions), values))
        best_score = evaluate(best_terms)
    
    # Convert best terms back to coefficient list
    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = val
    
    # Ensure length <= 41 and nonzero final coefficient
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    if coeff_list[-1] == 0:
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]
    
    return coeff_list
# EVOLVE-BLOCK-END
