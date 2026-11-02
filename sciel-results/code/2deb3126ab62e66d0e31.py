import numpy as np
import itertools
import time
import random

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: for each codeword c, ball_indices[c] = array of indices within distance 1
ball_indices = []
for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices[-1].append(idx)

ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    best_code = get_tetracode_code()
    best_size = len(best_code)

    def greedy_with_reduction():
        """Run greedy set cover with randomization, then reduction."""
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        while not np.all(covered):
            # Compute scores: number of uncovered words in each ball
            scores = np.zeros(NUM_WORDS, dtype=np.int32)
            for c in range(NUM_WORDS):
                scores[c] = np.sum(~covered[ball_indices[c]])
            
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            # Randomized selection from top candidates
            threshold = max_score * 0.85
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[ball_indices[c]] = True
            
        # Reduction Phase
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
            
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
                    
        return code

    def simulated_annealing(initial_code, max_iterations=50000):
        """Simulated annealing on the binary codeword selection problem."""
        code = list(initial_code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
        
        uncovered = np.sum(cov_counts == 0)
        current_energy = 1000.0 * uncovered + 1.0 * len(code)
        
        best_code = list(code)
        best_energy = current_energy
        
        T_start = 50.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
        T = T_start
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Choose a random move
            move_type = np.random.randint(3)
            
            if move_type == 0 and len(code) > 0:
                # Remove a codeword
                idx = np.random.randint(len(code))
                c_remove = code[idx]
                
                # Check if removal keeps coverage valid
                # Words that would become uncovered
                would_lose = cov_counts[ball_indices[c_remove]] == 1
                new_uncovered = np.sum(would_lose)
                delta_energy = 1000.0 * new_uncovered - 1.0
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code.pop(idx)
                    cov_counts[ball_indices[c_remove]] -= 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
                        
            elif move_type == 1:
                # Add a codeword
                c_add = np.random.randint(NUM_WORDS)
                if c_add in code:
                    continue
                    
                # Check how many new words it covers
                new_covered = np.sum(cov_counts[ball_indices[c_add]] == 0)
                delta_energy = -1000.0 * new_covered + 1.0
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code.append(c_add)
                    cov_counts[ball_indices[c_add]] += 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
                        
            elif move_type == 2 and len(code) > 0:
                # Swap: remove one, add one
                idx = np.random.randint(len(code))
                c_remove = code[idx]
                c_add = np.random.randint(NUM_WORDS)
                if c_add == c_remove or c_add in code:
                    continue
                
                # Compute delta for removal
                would_lose = cov_counts[ball_indices[c_remove]] == 1
                lose_count = np.sum(would_lose)
                
                # Compute delta for addition (considering the removal)
                # After removal, cov_counts would decrease by 1 in ball[c_remove]
                # So for c_add, new_covered = words with cov_count==0 after removal
                # This is complex, so approximate:
                # new_covered = words in ball[c_add] with cov_count==0, minus overlap with ball[c_remove]
                # Simplified: just compute both independently and take min
                new_covered_add = np.sum(cov_counts[ball_indices[c_add]] == 0)
                # Subtract overlap that might be affected
                overlap = np.intersect1d(ball_indices[c_remove], ball_indices[c_add])
                overlap_affected = np.sum(cov_counts[overlap] == 1)
                
                delta_energy = 1000.0 * lose_count - 1000.0 * (new_covered_add - overlap_affected)  # net change in uncovered
                # The +1 and -1 for codeword count cancel out in a swap
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code[idx] = c_add
                    cov_counts[ball_indices[c_remove]] -= 1
                    cov_counts[ball_indices[c_add]] += 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
            
            T *= cooling_rate
        
        return best_code

    # Phase 1: Find good initial solutions with greedy + reduction
    initial_solutions = []
    for restart in range(50):
        if time.time() > deadline:
            break
        sol = greedy_with_reduction()
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]
        initial_solutions.append(sol)
    
    # Phase 2: Simulated annealing from best initial solutions
    # Sort initial solutions by size, take the best few
    initial_solutions.sort(key=len)
    for sol in initial_solutions[:10]:
        if time.time() > deadline:
            break
        
        # Run SA with different temperature schedules
        for trial in range(3):
            if time.time() > deadline:
                break
            
            sa_result = simulated_annealing(sol, max_iterations=30000)
            
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_code = [list(all_words[c]) for c in sa_result]
                
                if best_size <= 73:
                    # Found the best known, try to improve further
                    pass

    return best_code
