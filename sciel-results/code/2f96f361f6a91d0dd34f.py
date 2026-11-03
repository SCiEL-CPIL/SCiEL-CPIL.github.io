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
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    ball_matrix[i, i] = 1
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

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

    # Precompute pairwise ball intersection sizes for faster swap evaluation
    # ball_inter[i,j] = number of words in ball_i ∩ ball_j
    # This is expensive to precompute (729x729), so we compute on the fly but cache
    
    def compute_swap_gain_loss(c_remove, c_add, cov_counts):
        """Compute the net coverage change when swapping c_remove for c_add."""
        br = ball_indices[c_remove]
        ba = ball_indices[c_add]
        
        # Use set operations for efficiency
        br_set = set(br.tolist())
        ba_set = set(ba.tolist())
        
        # Words only in br (lost if c_remove is removed and not covered by c_add)
        br_only = br_set - ba_set
        # Words only in ba (gained if c_add is added and not covered by c_remove)
        ba_only = ba_set - br_set
        # Words in both (no net change in coverage count)
        
        lose_count = 0
        for w in br_only:
            if cov_counts[w] == 1:
                lose_count += 1
        
        gain_count = 0
        for w in ba_only:
            if cov_counts[w] == 0:
                gain_count += 1
        
        return lose_count, gain_count

    def greedy_with_reduction():
        """Run vectorized greedy set cover with randomization, then reduction."""
        covered = np.zeros(NUM_WORDS, dtype=bool)
        uncovered_count = NUM_WORDS
        code = []
        
        while uncovered_count > 0:
            # Compute scores: number of uncovered words in each ball
            # Use int8 dot product for speed
            scores = ball_matrix.dot((~covered).astype(np.int8))
            
            max_score = scores.max()
            if max_score == 0:
                break
                
            # Randomized selection from top candidates
            threshold = max_score * 0.9
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            # Update covered in-place
            bi = ball_indices[c]
            new_uncovered = (~covered[bi]).sum()
            uncovered_count -= new_uncovered
            covered[bi] = True
            
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
                # Check if all words in this codeword's ball are covered by >1 codeword
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
                    
        return code

    def local_search_reduce(code, max_iterations=100000):
        """Local search that tries to reduce the code size via removals and swaps."""
        code = list(code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
        
        best_code = list(code)
        best_size = len(code)
        
        # Build a set for fast membership testing
        code_set = set(code)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            made_progress = False
            
            # Try removals
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                c = code[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    code_set.discard(c)
                    made_progress = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    break
            
            if made_progress:
                continue
            
            # Try swaps: replace one codeword with another to enable further removals
            if len(code) > 0:
                # Sample a few random swaps
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    i = np.random.randint(len(code))
                    c_remove = code[i]
                    c_add = np.random.randint(NUM_WORDS)
                    
                    if c_add == c_remove or c_add in code_set:
                        continue
                    
                    lose_count, gain_count = compute_swap_gain_loss(c_remove, c_add, cov_counts)
                    
                    # Accept if it doesn't increase uncovered count (gain >= lose)
                    # and prefer swaps that increase gain
                    if gain_count >= lose_count:
                        # Apply swap
                        cov_counts[ball_indices[c_remove]] -= 1
                        cov_counts[ball_indices[c_add]] += 1
                        code[i] = c_add
                        code_set.discard(c_remove)
                        code_set.add(c_add)
                        made_progress = True
                        
                        # Try to remove now-redundant codewords
                        for j in list(range(len(code))):
                            if np.all(cov_counts[ball_indices[code[j]]] > 1):
                                cov_counts[ball_indices[code[j]]] -= 1
                                code.pop(j)
                                code_set.discard(code[j])
                                if len(code) < best_size:
                                    best_size = len(code)
                                    best_code = list(code)
                                break
                        
                        break
        
        return best_code

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
                
                br = ball_indices[c_remove]
                ba = ball_indices[c_add]
                
                # Compute lose and gain
                in_br = np.zeros(NUM_WORDS, dtype=bool)
                in_br[br] = True
                in_ba = np.zeros(NUM_WORDS, dtype=bool)
                in_ba[ba] = True
                
                # lose: in br, not in ba, cov==1
                lose_mask = in_br & ~in_ba
                lose_count = np.sum(cov_counts[lose_mask] == 1)
                
                # gain: in ba, not in br, cov==0
                gain_mask = in_ba & ~in_br
                gain_count = np.sum(cov_counts[gain_mask] == 0)
                
                delta_energy = 1000.0 * (lose_count - gain_count)
                # Codeword count unchanged in swap
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code[idx] = c_add
                    cov_counts[br] -= 1
                    cov_counts[ba] += 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
            
            T *= cooling_rate
        
        return best_code

    # Phase 1: Find good initial solutions with greedy + reduction
    initial_solutions = []
    restart_count = 0
    while time.time() < deadline - 30: # Reserve 30s for local search and safety
        sol = greedy_with_reduction()
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]
        initial_solutions.append(sol)
        restart_count += 1
        
    # Phase 2: Local search on the best solutions
    initial_solutions.sort(key=len)
    
    num_ls_trials = min(5, len(initial_solutions))
    for i in range(num_ls_trials):
        if time.time() > deadline - 10:
            break
        sol = initial_solutions[i]
        
        ls_result = local_search_reduce(sol, max_iterations=50000)
        
        if len(ls_result) < best_size:
            best_size = len(ls_result)
            best_code = [list(all_words[c]) for c in ls_result]
            
            if best_size <= 73:
                pass

    # Phase 3: Final polish - aggressive local search on the best solution
    if time.time() < deadline - 3:
        # Convert best_code back to indices
        best_idx = []
        for cw in best_code:
            best_idx.append(word_to_idx[tuple(cw)])
        
        # Try removals in random order, multiple passes
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in best_idx:
            cov_counts[ball_indices[c]] += 1
        
        polish_changed = True
        while polish_changed and time.time() < deadline - 3:
            polish_changed = False
            idx_list = list(range(len(best_idx)))
            np.random.shuffle(idx_list)
            for idx in idx_list:
                if time.time() > deadline - 3:
                    break
                c = best_idx[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    best_idx.pop(idx)
                    polish_changed = True
                    break
        
        # Try swaps to reduce size
        if time.time() < deadline - 3:
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                if len(best_idx) == 0:
                    break
                i = np.random.randint(len(best_idx))
                j = np.random.randint(NUM_WORDS)
                if j == best_idx[i] or j in best_idx:
                    continue
                bi = ball_indices[best_idx[i]]
                bj = ball_indices[j]
                # Check if swap helps: remove i, add j
                # Lose: words in bi with cov==1 and not in bj
                # Gain: words in bj with (cov==0 if not in bi) or (cov==1 if in bi)
                in_bi = np.zeros(NUM_WORDS, dtype=bool)
                in_bi[bi] = True
                # Words in bi not in bj: lose if cov==1
                # Words in bj not in bi: gain if cov==0
                # Words in both: no net change
                bi_only = bi[np.isin(bi, bj, invert=True)]
                bj_only = bj[np.isin(bj, bi, invert=True)]
                lose_count = np.sum(cov_counts[bi_only] == 1)
                gain_count = np.sum(cov_counts[bj_only] == 0)
                if gain_count > lose_count:
                    cov_counts[bi] -= 1
                    cov_counts[bj] += 1
                    best_idx[i] = j
                    polish_changed = True
        
        if len(best_idx) < best_size:
            best_size = len(best_idx)
            best_code = [list(all_words[c]) for c in best_idx]

    return best_code
