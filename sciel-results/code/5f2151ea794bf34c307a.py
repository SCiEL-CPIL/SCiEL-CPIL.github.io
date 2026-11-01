import time
import random
import itertools
import numpy as np

# Precompute all words and their neighbors for the specific case q=3, n=7, radius=1
Q = 3
N_LEN = 7
RADIUS = 1
TOTAL_WORDS = Q ** N_LEN  # 2187

# Map word tuple to index
word_to_idx = {}
idx_to_word = []
index = 0
for w in itertools.product(range(Q), repeat=N_LEN):
    word_to_idx[w] = index
    idx_to_word.append(w)
    index += 1

# Precompute neighbors (ball of radius 1) as numpy arrays for fast operations
neighbors = []
for i in range(TOTAL_WORDS):
    w = idx_to_word[i]
    ball = [i]
    for pos in range(N_LEN):
        for val in range(Q):
            if val == w[pos]:
                continue
            new_w = list(w)
            new_w[pos] = val
            ball.append(word_to_idx[tuple(new_w)])
    neighbors.append(np.array(ball, dtype=np.int32))

# Precompute neighbor sets as frozensets for faster set operations
neighbor_sets = [frozenset(nb) for nb in neighbors]

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Fallback to the known product code if time is tight or for initialization
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_sol = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    base_size = len(base_sol)
    
    best_size = base_size
    best_solution = base_sol
    
    seed = 0
    while time.time() < deadline:
        sol, size = run_greedy_search(seed)
        if size < best_size:
            best_size = size
            best_solution = sol
            if best_size <= 156:
                break
        seed += 1
        
    return best_solution

def run_greedy_search(seed_val):
    rng = random.Random(seed_val)
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    uncovered = TOTAL_WORDS
    selected = set()
    uncovered_set = set(range(TOTAL_WORDS))
    
    # Phase 1: Greedy construction
    while uncovered_set and time.time() < deadline - 2:
        best_gain = 0
        best_candidates = []
        
        # Sample a subset of candidates if search space is large
        # But 2187 is small, so check all
        for c in range(TOTAL_WORDS):
            if c in selected:
                continue
            gain = len(uncovered_set & neighbor_sets[c])
            if gain > best_gain:
                best_gain = gain
                best_candidates = [c]
            elif gain == best_gain and gain > 0:
                best_candidates.append(c)
        
        if best_gain == 0:
            break
            
        # Randomly pick from best candidates to diversify
        c = rng.choice(best_candidates)
        selected.add(c)
        ball = neighbors[c]
        for v in ball:
            if v in uncovered_set:
                uncovered_set.remove(v)
                uncovered -= 1
            cov[v] += 1
    
    # Phase 2: Deletion - remove redundant codewords
    sel_list = list(selected)
    changed = True
    while changed:
        changed = False
        rng.shuffle(sel_list)
        for c in sel_list:
            if c not in selected:
                continue
            # Check if c is essential
            is_essential = False
            for v in neighbors[c]:
                if cov[v] == 1:
                    is_essential = True
                    break
            
            if not is_essential:
                selected.remove(c)
                for v in neighbors[c]:
                    cov[v] -= 1
                changed = True
    
    # Phase 3: Swap-based local search
    # Try to replace one codeword with another to improve coverage
    improved = True
    max_swaps = 50  # Limit swaps to avoid infinite loops
    swaps_done = 0
    while improved and swaps_done < max_swaps and time.time() < deadline - 5:
        improved = False
        sel_list = list(selected)
        rng.shuffle(sel_list)
        
        for old_c in sel_list:
            if old_c not in selected:
                continue
            
            # Calculate which words are only covered by old_c
            old_ball = neighbors[old_c]
            critical_words = set()
            for v in old_ball:
                if cov[v] == 1:
                    critical_words.add(v)
            
            # Try to find a new codeword that covers all critical words
            # and potentially adds new coverage
            best_new_c = None
            best_new_gain = 0
            
            for new_c in range(TOTAL_WORDS):
                if new_c in selected or new_c == old_c:
                    continue
                
                new_ball = neighbors[new_c]
                new_ball_set = neighbor_sets[new_c]
                
                # Check if new_c covers all critical words
                if not critical_words.issubset(new_ball_set):
                    continue
                
                # Calculate additional coverage (words covered by new_c but not by old_c)
                additional = len(new_ball_set - set(old_ball))
                
                if additional > best_new_gain:
                    best_new_gain = additional
                    best_new_c = new_c
            
            if best_new_c is not None and best_new_gain > 0:
                # Perform the swap
                selected.remove(old_c)
                selected.add(best_new_c)
                
                # Update coverage
                for v in neighbors[old_c]:
                    cov[v] -= 1
                for v in neighbors[best_new_c]:
                    cov[v] += 1
                
                improved = True
                swaps_done += 1
                break  # Restart the loop after a successful swap
    
    # Final deletion pass
    sel_list = list(selected)
    changed = True
    while changed:
        changed = False
        rng.shuffle(sel_list)
        for c in sel_list:
            if c not in selected:
                continue
            is_essential = False
            for v in neighbors[c]:
                if cov[v] == 1:
                    is_essential = True
                    break
            
            if not is_essential:
                selected.remove(c)
                for v in neighbors[c]:
                    cov[v] -= 1
                changed = True
    
    # Convert back to list of words
    solution = [list(idx_to_word[c]) for c in selected]
    return solution, len(solution)

# Original construct function with fallback
_original_construct = construct

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _original_construct(q, n, radius)
    else:
        # Fallback for other parameters
        start_time = time.time()
        deadline = 115.0
        
        if n < 4:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
        base_sol = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        
        return base_sol
