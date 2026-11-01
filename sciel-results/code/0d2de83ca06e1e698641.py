import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses simulated annealing with NumPy-based coverage deficit to minimize codewords.
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Precompute neighbor lists for each word
    neighbors = []
    for idx in range(total_words):
        word = []
        tmp = idx
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        
        covered = set([idx])
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val != orig:
                        word[pos] = new_val
                        val = 0
                        for i in range(n - 1, -1, -1):
                            val = val * q + word[i]
                        covered.add(val)
                word[pos] = orig
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1, orig2 = word[pos1], word[pos2]
                    for v1 in range(q):
                        if v1 != orig1:
                            word[pos1] = v1
                            for v2 in range(q):
                                if v2 != orig2:
                                    word[pos2] = v2
                                    val = 0
                                    for i in range(n - 1, -1, -1):
                                        val = val * q + word[i]
                                    covered.add(val)
                            word[pos1] = orig1
        neighbors.append(covered)
    
    # Build coverage matrix: cov_matrix[i, j] = 1 if word j is covered by codeword i
    # For memory efficiency with 729 words, use int8
    cov_matrix = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in neighbors[i]:
            cov_matrix[i, j] = 1
    
    # Greedy initialization
    def greedy_init():
        uncovered_mask = np.ones(total_words, dtype=np.int8)
        selected = []
        
        # Try to seed with tetracode for q=3, n=6, radius=1
        if q == 3 and n == 6 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            for tail_a in range(3):
                for tail_b in range(3):
                    for h in tetracode:
                        word = h + [tail_a, tail_b]
                        idx = 0
                        for i in range(n - 1, -1, -1):
                            idx = idx * q + word[i]
                        selected.append(idx)
                        uncovered_mask[list(neighbors[idx])] = 0
            # Remove duplicates
            selected = list(set(selected))
        
        while True:
            if time.time() > deadline:
                break
            uncovered_count = uncovered_mask.sum()
            if uncovered_count == 0:
                break
            
            # Find candidate that covers most uncovered words
            # coverage[i] = number of uncovered words covered by candidate i
            coverage = (cov_matrix * uncovered_mask).sum(axis=1)
            best = np.argmax(coverage)
            if coverage[best] == 0:
                break
            selected.append(best)
            uncovered_mask[neighbors[best]] = 0
        
        return list(set(selected))
    
    # Simulated annealing
    def simulated_annealing(initial_codewords, time_limit):
        if not initial_codewords:
            return []
        
        # State: binary vector of which codewords are included
        # We'll work with a fixed pool of candidates to keep the state space manageable
        # Pool = initial codewords + some random candidates
        pool = set(initial_codewords)
        # Add some random candidates to allow swaps
        for _ in range(min(200, total_words)):
            pool.add(random.randint(0, total_words - 1))
        pool = list(pool)
        pool_idx = {p: i for i, p in enumerate(pool)}
        num_pool = len(pool)
        
        # Precompute coverage for pool candidates
        pool_cov = cov_matrix[np.array(pool), :]  # shape (num_pool, total_words)
        
        # Initial state: include all initial codewords
        state = np.zeros(num_pool, dtype=np.int8)
        for c in initial_codewords:
            state[pool_idx[c]] = 1
        
        # Compute current coverage and cost
        def compute_coverage(state):
            # coverage = pool_cov.T @ state, shape (total_words,)
            cov = pool_cov.T.dot(state)
            uncovered = (cov == 0).sum()
            num_codewords = state.sum()
            return uncovered, num_codewords
        
        uncovered, num_cw = compute_coverage(state)
        best_state = state.copy()
        best_uncovered = uncovered
        best_num_cw = num_cw
        best_cost = best_num_cw + best_uncovered * 100  # Heavy penalty for uncovered
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() < time_limit:
            # Pick a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a codeword
                active = np.where(state == 1)[0]
                if len(active) == 0:
                    continue
                idx_remove = random.choice(active.tolist())
                new_state = state.copy()
                new_state[idx_remove] = 0
                
            elif move_type < 0.6:
                # Add a codeword
                inactive = np.where(state == 0)[0]
                if len(inactive) == 0:
                    continue
                idx_add = random.choice(inactive.tolist())
                new_state = state.copy()
                new_state[idx_add] = 1
                
            else:
                # Swap: remove one, add another
                active = np.where(state == 1)[0]
                inactive = np.where(state == 0)[0]
                if len(active) == 0 or len(inactive) == 0:
                    continue
                idx_remove = random.choice(active.tolist())
                idx_add = random.choice(inactive.tolist())
                new_state = state.copy()
                new_state[idx_remove] = 0
                new_state[idx_add] = 1
            
            # Compute new coverage
            new_uncovered, new_num_cw = compute_coverage(new_state)
            new_cost = new_num_cw + new_uncovered * 100
            old_cost = num_cw + uncovered * 100
            
            delta = new_cost - old_cost
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                state = new_state
                uncovered = new_uncovered
                num_cw = new_num_cw
                
                if new_uncovered < best_uncovered or (new_uncovered == best_uncovered and new_num_cw < best_num_cw):
                    best_state = state.copy()
                    best_uncovered = new_uncovered
                    best_num_cw = new_num_cw
                    best_cost = best_num_cw + best_uncovered * 100
            
            T = max(T_min, T * cooling)
        
        # Extract codewords from best state
        result = []
        for i in range(num_pool):
            if best_state[i] == 1:
                result.append(pool[i])
        return result
    
    # Try multiple restarts
    best_result = None
    best_size = float('inf')
    
    # First greedy solution
    greedy_sol = greedy_init()
    if greedy_sol:
        sa_sol = simulated_annealing(greedy_sol, deadline - 10)
        if sa_sol:
            # Verify coverage
            covered_all = np.zeros(total_words, dtype=np.int8)
            for c in sa_sol:
                covered_all[list(neighbors[c])] = 1
            if covered_all.sum() == total_words:
                if len(sa_sol) < best_size:
                    best_size = len(sa_sol)
                    best_result = sa_sol
    
    # More restarts if time permits
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        greedy_sol = greedy_init()
        if greedy_sol:
            sa_sol = simulated_annealing(greedy_sol, deadline - 5)
            if sa_sol:
                covered_all = np.zeros(total_words, dtype=np.int8)
                for c in sa_sol:
                    covered_all[list(neighbors[c])] = 1
                if covered_all.sum() == total_words:
                    if len(sa_sol) < best_size:
                        best_size = len(sa_sol)
                        best_result = sa_sol
    
    # Final local search: try to remove redundant codewords
    if best_result:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(best_result) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Check if best_result[i] is redundant
                redundant = True
                for w in neighbors[best_result[i]]:
                    covered_by_other = False
                    for j in range(len(best_result)):
                        if j == i:
                            continue
                        if w in neighbors[best_result[j]]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                if redundant:
                    best_result.pop(i)
                    changed = True
                    break
    
    # Convert to lists of integers
    result = []
    if best_result:
        for c in best_result:
            word = []
            tmp = c
            for _ in range(n):
                word.append(tmp % q)
                tmp //= q
            word.reverse()
            result.append(word)
    else:
        # Fallback: return all words to ensure validity (though score will be poor)
        result = []
        for c in range(total_words):
            word = []
            tmp = c
            for _ in range(n):
                word.append(tmp % q)
                tmp //= q
            word.reverse()
            result.append(word)
    
    return result
