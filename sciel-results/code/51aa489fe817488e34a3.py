"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def fast_eval(positions, values):
        """
        Evaluate the number of non-zero terms in the square of the polynomial
        defined by positions and values.
        Returns (score, dense_array) to avoid recomputation if needed, but here we just need score.
        """
        max_deg = max(positions)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for p, v in zip(positions, values):
            dense[p] = v
        sq = np.convolve(dense, dense, mode='full')
        return int(np.count_nonzero(sq))

    def generate_initial_candidates():
        """
        Generate a diverse set of initial position candidates.
        Focus on structures that maximize sum collisions.
        """
        candidates = set()
        
        # 1. Arithmetic Progressions (AP)
        for start in range(0, 41):
            for step in range(1, 4):
                pos = tuple(start + i * step for i in range(12))
                if pos[-1] <= 40:
                    candidates.add(pos)
        
        # 2. Two interleaved APs (e.g., even/odd offsets)
        for s1 in range(0, 30):
            for s2 in range(s1 + 1, 40):
                for step in range(1, 4):
                    pos1 = [s1 + i * step for i in range(6)]
                    pos2 = [s2 + i * step for i in range(6)]
                    pos = sorted(pos1 + pos2)
                    if len(set(pos)) == 12 and pos[-1] <= 40:
                        candidates.add(tuple(pos))
        
        # 3. "Block" structures: clusters of consecutive integers
        # e.g., 3 blocks of 4
        for _ in range(2000):
            n_blocks = random.choice([2, 3, 4, 6])
            block_sizes = [12 // n_blocks] * n_blocks
            # Randomly adjust block sizes slightly if possible, but keep simple for now
            pos = []
            current_base = 0
            for size in block_sizes:
                # Random gap
                gap = random.randint(0, 10)
                current_base += gap
                block = list(range(current_base, current_base + size))
                pos.extend(block)
                current_base += size
            if len(set(pos)) == 12 and max(pos) <= 40:
                candidates.add(tuple(sorted(pos)))
            
        # 4. Random sets
        for _ in range(2000):
            pos = tuple(sorted(random.sample(range(41), 12)))
            candidates.add(pos)
            
        return list(candidates)

    def optimize_values_for_positions(positions, time_limit=0.5):
        """
        For a fixed set of positions, find values that minimize the non-zero count.
        Uses a mix of template-based initialization and local search.
        """
        start_t = time.time()
        best_score = 1000
        best_values = None
        
        # Define some value templates that promote cancellation
        # 1. All 1s
        # 2. Alternating 1, -1
        # 3. Small random integers
        # 4. Signs only
        
        def try_values(vals):
            nonlocal best_score, best_values
            score = fast_eval(positions, vals)
            if score < best_score:
                best_score = score
                best_values = vals[:]
                return True
            return False
        
        # 1. Templates
        try_values([1] * 12)
        try_values([1 if i % 2 == 0 else -1 for i in range(12)])
        try_values([1 if i % 3 == 0 else -1 for i in range(12)])
        
        # 2. Random search with bias toward small integers
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 2000:
            n_trials += 1
            vals = []
            for _ in range(12):
                r = random.random()
                if r < 0.5:
                    mag = 1
                elif r < 0.8:
                    mag = random.randint(1, 3)
                else:
                    mag = random.randint(1, 8)
                sign = 1 if random.random() < 0.5 else -1
                vals.append(sign * mag)
            
            if try_values(vals):
                if best_score <= 12:
                    break
        
        # 3. Local search on values (if we have a good start)
        if best_values:
            current_vals = best_values[:]
            current_score = best_score
            improved = True
            while improved and time.time() - start_t < time_limit:
                improved = False
                for _ in range(50):
                    if time.time() - start_t > time_limit:
                        break
                    idx = random.randint(0, 11)
                    old_val = current_vals[idx]
                    # Mutate value
                    r = random.random()
                    if r < 0.4:
                        new_val = -old_val
                    elif r < 0.7:
                        new_val = random.choice([1, -1]) * random.randint(1, 4)
                    else:
                        new_val = random.choice([1, -1]) * random.randint(1, 16)
                    
                    if new_val == old_val:
                        continue
                    
                    current_vals[idx] = new_val
                    score = fast_eval(positions, current_vals)
                    if score < current_score:
                        current_score = score
                        best_score = score
                        best_values = current_vals[:]
                        improved = True
                    else:
                        current_vals[idx] = old_val
                    if current_score <= 12:
                        return best_score, best_values
        
        return best_score, best_values

    def main_search():
        best_terms = None
        best_score = 1000
        
        candidates = generate_initial_candidates()
        # Sort candidates by sum-collision density (lower is better) to prioritize promising structures
        def sum_collisions(pos):
            sums = set()
            for i in range(12):
                for j in range(i, 12):
                    sums.add(pos[i] + pos[j])
            return len(sums)
            
        candidates.sort(key=sum_collisions)
        
        # Phase 1: Evaluate top structural candidates
        # We have ~50s. Let's spend 40s on top 500 candidates.
        # 500 candidates * 0.08s/candidate = 40s.
        # Adjust time per candidate dynamically.
        
        n_candidates = min(500, len(candidates))
        time_per_candidate = 40.0 / n_candidates
        
        for idx, pos in enumerate(candidates[:n_candidates]):
            if time.time() > deadline - 10:
                break
                
            remaining_time = deadline - time.time() - 10
            if remaining_time <= 0:
                break
            
            # Allocate time based on rank: better structures get more time
            rank_factor = 1.0 / (1.0 + idx * 0.1)
            t_limit = min(1.0, max(0.02, time_per_candidate * rank_factor))
            
            score, values = optimize_values_for_positions(pos, time_limit=t_limit)
            
            if values is not None and score < best_score:
                best_score = score
                best_terms = list(zip(pos, values))
                if best_score <= 12:
                    break
        
        # Phase 2: Local search on the best found solution (positions + values)
        if best_terms:
            current_terms = best_terms[:]
            current_score = best_score
            start_ls = time.time()
            ls_time_limit = min(10.0, max(1.0, deadline - time.time() - 2))
            
            def mutate_terms(terms):
                terms = terms[:]
                move_type = random.random()
                if move_type < 0.4:
                    # Change position
                    idx = random.randint(0, 11)
                    old_pos = terms[idx][0]
                    new_pos = random.randint(0, 40)
                    # Check collision
                    for i, (p, v) in enumerate(terms):
                        if i != idx and p == new_pos:
                            return terms
                    terms[idx] = (new_pos, terms[idx][1])
                elif move_type < 0.8:
                    # Change value
                    idx = random.randint(0, 11)
                    old_val = terms[idx][1]
                    r = random.random()
                    if r < 0.3:
                        new_val = -old_val
                    else:
                        new_val = random.choice([1, -1]) * random.randint(1, 16)
                    terms[idx] = (terms[idx][0], new_val)
                else:
                    # Swap positions
                    i, j = random.sample(range(12), 2)
                    terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])
                return terms
            
            improved = True
            while improved and time.time() - start_ls < ls_time_limit:
                improved = False
                for _ in range(100):
                    if time.time() - start_ls > ls_time_limit:
                        break
                    new_terms = mutate_terms(current_terms)
                    # Validate
                    positions = [p for p, v in new_terms]
                    values = [v for p, v in new_terms]
                    if len(set(positions)) != 12:
                        continue
                    if any(abs(v) > 64 or v == 0 for v in values):
                        continue
                    if any(p > 40 for p in positions):
                        continue
                    
                    score = fast_eval(positions, values)
                    if score < current_score:
                        current_terms = new_terms
                        current_score = score
                        best_score = score
                        best_terms = current_terms
                        improved = True
                        if current_score <= 12:
                            return best_terms, best_score
        
        return best_terms, best_score

    # Run main search
    best_terms, best_score = main_search()
    
    # Fallback
    if best_terms is None or len(best_terms) != 12:
        positions = random.sample(range(41), 12)
        values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
        best_terms = list(zip(sorted(positions), values))
        best_score = fast_eval([p for p, v in best_terms], [v for p, v in best_terms])
    
    # Convert to coefficient list
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
