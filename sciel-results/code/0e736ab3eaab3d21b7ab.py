import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with tetracode direct sum (243 words for q=3, n=8, radius=1)
    if q == 3 and n == 8 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [list(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    else:
        # Generic greedy construction
        initial = []
    
    if not initial:
        # Greedy set cover
        all_words = np.array(np.meshgrid(*[np.arange(q)]*n)).reshape(n, -1).T
        uncovered = set(map(tuple, all_words))
        code = []
        while uncovered and time.time() < deadline:
            best_word = None
            best_count = -1
            candidates = list(uncovered)
            random.shuffle(candidates)
            candidates = candidates[:min(500, len(candidates))]
            for w in candidates:
                w = tuple(w)
                count = 0
                for i in range(n):
                    for val in range(q):
                        if val == w[i]:
                            continue
                        nw = list(w)
                        nw[i] = val
                        if tuple(nw) in uncovered:
                            count += 1
                if w in uncovered:
                    count += 1
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_word is None:
                break
            code.append(list(best_word))
            bw = list(best_word)
            for i in range(n):
                for val in range(q):
                    nw = bw[:]
                    nw[i] = val
                    uncovered.discard(tuple(nw))
            uncovered.discard(tuple(best_word))
        initial = code
    
    # Simulated annealing to minimize codeword count with penalty for uncovered words
    code = initial[:]
    code_set = set(map(tuple, code))
    
    # Precompute coverage for each codeword
    def get_covered(cw, q, n, radius):
        """Get all words within Hamming distance radius of cw."""
        covered = set()
        for d in range(radius + 1):
            if d == 0:
                covered.add(tuple(cw))
            else:
                positions = list(itertools.combinations(range(n), d))
                for pos in positions:
                    for vals in itertools.product(range(q), repeat=d):
                        nw = cw[:]
                        for p, v in zip(pos, vals):
                            if v != cw[p]:
                                nw[p] = v
                        covered.add(tuple(nw))
        return covered
    
    # Compute initial coverage
    all_words = np.array(np.meshgrid(*[np.arange(q)]*n)).reshape(n, -1).T
    all_words_list = [tuple(w) for w in all_words]
    all_words_set = set(all_words_list)
    
    # Coverage count: how many codewords cover each word
    coverage_count = np.zeros(len(all_words_list), dtype=np.int32)
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    
    # Build coverage map
    code_covered = []
    for cw in code:
        cov = get_covered(cw, q, n, radius)
        code_covered.append(cov)
        for w in cov:
            if w in word_to_idx:
                coverage_count[word_to_idx[w]] += 1
    
    # Track uncovered words
    uncovered_mask = coverage_count == 0
    uncovered_count = int(np.sum(uncovered_mask))
    
    # Objective: minimize len(code) + penalty * uncovered_count
    penalty = 100  # High penalty to ensure coverage
    
    def objective(len_code, uncov_count):
        return len_code + penalty * uncov_count
    
    current_obj = objective(len(code), uncovered_count)
    best_code = code[:]
    best_obj = current_obj
    
    # Simulated annealing parameters
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.9995
    max_iterations = 50000
    
    # Precompute single-position neighbors for efficiency
    # For each position and value, compute the effect on coverage
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        if T < T_min:
            break
        
        # Choose a random move: remove, add, or swap a codeword
        move_type = random.random()
        
        if move_type < 0.4 and len(code) > 1:
            # Remove a codeword
            idx = random.randint(0, len(code) - 1)
            cw = code[idx]
            cov = code_covered[idx]
            
            # Check if removal keeps coverage
            new_uncovered = 0
            for w in cov:
                if w in word_to_idx:
                    wi = word_to_idx[w]
                    if coverage_count[wi] == 1:
                        new_uncovered += 1
            
            delta_obj = 1 + penalty * new_uncovered  # Removing 1 codeword, adding new_uncovered uncovered words
            
            if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
                # Accept removal
                code.pop(idx)
                code_covered.pop(idx)
                for w in cov:
                    if w in word_to_idx:
                        coverage_count[word_to_idx[w]] -= 1
                
                uncovered_count += new_uncovered
                current_obj += delta_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code = code[:]
        
        elif move_type < 0.7:
            # Add a codeword
            # Choose a word that is uncovered or near-uncovered
            if uncovered_count > 0:
                uncov_indices = np.where(coverage_count == 0)[0]
                if len(uncov_indices) > 0:
                    wi = random.choice(uncov_indices)
                    new_cw = list(all_words_list[wi])
                else:
                    wi = random.randint(0, len(all_words_list) - 1)
                    new_cw = list(all_words_list[wi])
            else:
                wi = random.randint(0, len(all_words_list) - 1)
                new_cw = list(all_words_list[wi])
            
            new_cw_tuple = tuple(new_cw)
            if new_cw_tuple in code_set:
                continue
            
            cov = get_covered(new_cw, q, n, radius)
            
            # Count how many currently uncovered words this would cover
            newly_covered = 0
            for w in cov:
                if w in word_to_idx:
                    wi = word_to_idx[w]
                    if coverage_count[wi] == 0:
                        newly_covered += 1
            
            delta_obj = 1 - penalty * newly_covered
            
            if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
                # Accept addition
                code.append(new_cw)
                code_covered.append(cov)
                code_set.add(new_cw_tuple)
                
                for w in cov:
                    if w in word_to_idx:
                        coverage_count[word_to_idx[w]] += 1
                
                uncovered_count -= newly_covered
                current_obj += delta_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code = code[:]
        
        else:
            # Swap: remove one codeword and add another
            if len(code) > 1:
                idx = random.randint(0, len(code) - 1)
                old_cw = code[idx]
                old_cov = code_covered[idx]
                
                # Choose replacement
                if uncovered_count > 0:
                    uncov_indices = np.where(coverage_count == 0)[0]
                    if len(uncov_indices) > 0:
                        wi = random.choice(uncov_indices)
                        new_cw = list(all_words_list[wi])
                    else:
                        wi = random.randint(0, len(all_words_list) - 1)
                        new_cw = list(all_words_list[wi])
                else:
                    wi = random.randint(0, len(all_words_list) - 1)
                    new_cw = list(all_words_list[wi])
                
                new_cw_tuple = tuple(new_cw)
                if new_cw_tuple in code_set and new_cw_tuple != tuple(old_cw):
                    continue
                
                new_cov = get_covered(new_cw, q, n, radius)
                
                # Compute delta
                # Words that were covered only by old_cw
                lost_coverage = 0
                for w in old_cov:
                    if w in word_to_idx:
                        wi = word_to_idx[w]
                        if coverage_count[wi] == 1 and w not in new_cov:
                            lost_coverage += 1
                
                # Words that were uncovered and now covered by new_cw
                gained_coverage = 0
                for w in new_cov:
                    if w in word_to_idx:
                        wi = word_to_idx[w]
                        if coverage_count[wi] == 0:
                            gained_coverage += 1
                
                delta_obj = penalty * (lost_coverage - gained_coverage)
                
                if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
                    # Accept swap
                    for w in old_cov:
                        if w in word_to_idx:
                            coverage_count[word_to_idx[w]] -= 1
                    for w in new_cov:
                        if w in word_to_idx:
                            coverage_count[word_to_idx[w]] += 1
                    
                    code_set.discard(tuple(old_cw))
                    code_set.add(new_cw_tuple)
                    
                    code[idx] = new_cw
                    code_covered[idx] = new_cov
                    
                    uncovered_count += lost_coverage - gained_coverage
                    current_obj += delta_obj
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_code = code[:]
        
        T *= cooling_rate
    
    # Use best solution found
    code = best_code
    
    # Final verification and cleanup
    # Check if valid
    coverage_count_final = np.zeros(len(all_words_list), dtype=np.int32)
    for cw in code:
        cov = get_covered(cw, q, n, radius)
        for w in cov:
            if w in word_to_idx:
                coverage_count_final[word_to_idx[w]] += 1
    
    if np.all(coverage_count_final > 0):
        return [list(w) for w in code]
    else:
        # If invalid, return the initial valid solution
        return [list(w) for w in initial]
