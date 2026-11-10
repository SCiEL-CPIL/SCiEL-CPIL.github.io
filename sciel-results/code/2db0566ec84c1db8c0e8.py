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

    def main_search():
        best_terms = None
        best_score = 1000
        
        # Phase 1: Structural search over high-collision position sets
        candidates = generate_structural_candidates()
        
        # Try top candidates by collision count
        for idx, positions in enumerate(candidates[:300]):
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time()
            n_remaining = min(len(candidates) - idx, 300)
            if n_remaining > 0:
                time_per = max(0.03, remaining / max(n_remaining, 1))
            else:
                time_per = 0.3
            
            score, values = find_best_values_for_positions(list(positions), time_limit=time_per)
            
            if values is not None and score < best_score:
                best_score = score
                best_terms = list(zip(positions, values))
                if best_score <= 12:
                    break
        
        # Phase 2: Local search refinement from best structural result
        if best_terms is not None:
            refined_terms, refined_score = local_search_from(best_terms, time_limit=min(3.0, max(0.5, deadline - time.time() - 2)))
            if refined_score < best_score:
                best_score = refined_score
                best_terms = refined_terms
        
        # Phase 3: Additional random restarts with local search
        while time.time() < deadline - 2:
            # Random initial
            positions = random.sample(range(41), 12)
            values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
            terms = list(zip(sorted(positions), values))
            
            # Quick local search
            ls_time = min(1.0, max(0.2, deadline - time.time() - 2))
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
