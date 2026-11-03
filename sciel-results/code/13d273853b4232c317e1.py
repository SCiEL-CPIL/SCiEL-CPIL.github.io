import time
import random
import itertools
import numpy as np

# Precomputation for the specific case q=3, n=8
Q = 3
N = 8
R = 1
TOTAL_WORDS = Q ** N  # 6561

# Convert base-3 vector to int
def vec_to_int(vec):
    res = 0
    for v in vec:
        res = res * Q + v
    return res

# Convert int to base-3 vector
def int_to_vec(num, n=N, q=Q):
    vec = [0] * n
    for i in range(n - 1, -1, -1):
        vec[i] = num % q
        num //= q
    return vec

# Precompute balls for all words
# ball[i] is a list of ints j such that dist(i, j) <= R
BALLS = []
for i in range(TOTAL_WORDS):
    vec_i = int_to_vec(i)
    ball = [i]
    for pos in range(N):
        orig = vec_i[pos]
        for new_val in range(Q):
            if new_val == orig:
                continue
            vec_j = vec_i[:]
            vec_j[pos] = new_val
            j = vec_to_int(vec_j)
            ball.append(j)
    BALLS.append(ball)

# Precompute tetracode-based structured initial code
def _tetracode_product():
    """Generate the tetracode product code: 81 codewords that cover a large portion."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=N - 4):
            code.append(vec_to_int(h + list(tail)))
    return code

def _greedy_cover_from_uncovered(uncovered_set, counts_arr, seed):
    """Greedy cover starting from a given uncovered set and counts array."""
    rng = random.Random(seed)
    uncovered = set(uncovered_set)
    counts = counts_arr.copy()
    code = []
    
    while uncovered:
        # Find candidate with max count among all words
        max_count = -1
        best_candidates = []
        for i in range(TOTAL_WORDS):
            if counts[i] > max_count:
                max_count = counts[i]
                best_candidates = [i]
            elif counts[i] == max_count:
                best_candidates.append(i)
                
        if not best_candidates:
            break
            
        c_star = rng.choice(best_candidates)
        code.append(c_star)
        
        # Update uncovered and counts
        for w in BALLS[c_star]:
            if w in uncovered:
                uncovered.remove(w)
                for c in BALLS[w]:
                    counts[c] -= 1
                    
    return code, uncovered, counts

def _local_search_hypergraph(code_set, uncovered_set, counts_arr, deadline, seed):
    """
    Simulated annealing local search on the hypergraph vertex cover problem.
    Vertices: 6561 words. Hyperedges: 17-word spheres.
    Objective: minimize |code_set| subject to uncovered_set being empty.
    
    Moves:
    - Remove a codeword if it doesn't uncover any new vertices (or accept with probability)
    - Add a codeword to reduce uncovered set
    - Swap: remove one, add another
    """
    rng = random.Random(seed)
    code = set(code_set)
    uncovered = set(uncovered_set)
    counts = counts_arr.copy()
    
    best_code = set(code)
    best_uncovered = set(uncovered)
    best_size = len(code)
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    T_current = T_start
    
    # Precompute for each codeword, how many words it uniquely covers
    # unique_count[c] = number of w in ball(c) that are NOT covered by any other codeword in code
    
    def compute_unique_counts(code_set, counts_arr):
        """For each codeword, count how many words in its ball are only covered by it."""
        unique = [0] * TOTAL_WORDS
        # For each word, find how many codewords cover it
        # Then a word is uniquely covered by c if count==1 and c covers it
        cover_counts = np.zeros(TOTAL_WORDS, dtype=np.int32)
        for c in code_set:
            for w in BALLS[c]:
                cover_counts[w] += 1
        for c in code_set:
            for w in BALLS[c]:
                if cover_counts[w] == 1:
                    unique[c] += 1
        return unique
    
    # Main SA loop
    steps = 0
    max_steps = 50000
    
    while True:
        elapsed = time.time() - _local_search_hypergraph._start_time
        if elapsed > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        T_current = T_start * (T_end / T_start) ** (steps / max_steps)
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.3 and code:
            # Try to remove a codeword
            c_remove = rng.choice(list(code))
            
            # Check if removing c_remove would uncover any words
            # A word w in ball(c_remove) becomes uncovered if no other codeword covers it
            # i.e., cover_count[w] == 1 before removal
            would_uncover = 0
            for w in BALLS[c_remove]:
                if w in uncovered:
                    continue  # already uncovered, removing c doesn't change this
                # Check if any other codeword covers w
                covered_by_other = False
                for c2 in code:
                    if c2 == c_remove:
                        continue
                    if w in BALLS[c2]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    would_uncover += 1
            
            if would_uncover == 0:
                # Safe removal
                code.discard(c_remove)
                for w in BALLS[c_remove]:
                    if w not in uncovered:
                        for c2 in BALLS[w]:
                            if c2 in code:
                                counts[c2] -= 1
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = set(code)
                    best_uncovered = set(uncovered)
            else:
                # Accept with probability exp(-would_uncover / T_current)
                prob = np.exp(-would_uncover / max(T_current, 1e-9))
                if rng.random() < prob:
                    code.discard(c_remove)
                    for w in BALLS[c_remove]:
                        if w not in uncovered:
                            uncovered.add(w)
                            for c2 in BALLS[w]:
                                if c2 in code:
                                    counts[c2] -= 1
        
        elif move_type < 0.6:
            # Try to add a codeword that covers the most uncovered words
            if uncovered:
                # Find codeword that covers most uncovered words
                best_gain = 0
                best_add = None
                for c in range(TOTAL_WORDS):
                    if c in code:
                        continue
                    gain = 0
                    for w in BALLS[c]:
                        if w in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add = c
                
                if best_add is not None and best_gain > 0:
                    code.add(best_add)
                    for w in BALLS[best_add]:
                        if w in uncovered:
                            uncovered.discard(w)
                            for c2 in BALLS[w]:
                                if c2 in code:
                                    counts[c2] += 1
        
        else:
            # Swap: remove one, add another
            if code:
                c_remove = rng.choice(list(code))
                
                # Find best replacement
                best_gain = 0
                best_add = None
                for c in range(TOTAL_WORDS):
                    if c in code:
                        continue
                    # Net gain = words covered by c that are currently uncovered
                    gain = 0
                    for w in BALLS[c]:
                        if w in uncovered:
                            gain += 1
                    # Penalty = words that would become uncovered if we remove c_remove
                    # (approximate: count words in ball(c_remove) not in ball(c) and not covered by others)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = c
                
                if best_add is not None and best_gain > 0:
                    # Perform swap
                    code.discard(c_remove)
                    code.add(best_add)
                    # Update uncovered
                    for w in BALLS[c_remove]:
                        if w not in uncovered:
                            # Check if still covered
                            still_covered = False
                            for c2 in code:
                                if w in BALLS[c2]:
                                    still_covered = True
                                    break
                            if not still_covered:
                                uncovered.add(w)
                    for w in BALLS[best_add]:
                        if w in uncovered:
                            uncovered.discard(w)
    
    return best_code, best_uncovered, best_size

def _trim_redundant(code_list):
    """Remove codewords that are redundant (their ball is covered by others)."""
    code_set = set(code_list)
    changed = True
    while changed:
        changed = False
        for c in list(code_set):
            # Check if all words in ball(c) are covered by code_set \ {c}
            redundant = True
            for w in BALLS[c]:
                covered = False
                for c2 in code_set:
                    if c2 == c:
                        continue
                    if w in BALLS[c2]:
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                code_set.discard(c)
                changed = True
    return list(code_set)

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # 110s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Phase 1: Structured initialization using tetracode product
    initial_code = _tetracode_product()
    
    # Compute initial uncovered set and counts
    covered = set()
    for c in initial_code:
        for w in BALLS[c]:
            covered.add(w)
    uncovered = set(range(TOTAL_WORDS)) - covered
    
    # Compute counts: for each word, how many codewords cover it
    counts = [0] * TOTAL_WORDS
    for c in initial_code:
        for w in BALLS[c]:
            counts[w] += 1
    
    # Phase 2: Greedy completion if not fully covered
    if uncovered:
        code, uncovered, counts = _greedy_cover_from_uncovered(uncovered, counts, 42)
        # Combine: initial_code + code (but remove duplicates)
        all_code = list(set(initial_code) | set(code))
    else:
        all_code = list(initial_code)
    
    # Trim redundant codewords
    all_code = _trim_redundant(all_code)
    
    best_code = list(all_code)
    best_size = len(best_code)
    
    # Phase 3: Local search with simulated annealing
    _local_search_hypergraph._start_time = time.time()
    
    seed_val = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Run local search with current best
        code_set = set(best_code)
        # Recompute uncovered and counts
        covered = set()
        for c in code_set:
            for w in BALLS[c]:
                covered.add(w)
        uncovered = set(range(TOTAL_WORDS)) - covered
        counts = [0] * TOTAL_WORDS
        for c in code_set:
            for w in BALLS[c]:
                counts[w] += 1
        
        result_code, result_uncovered, result_size = _local_search_hypergraph(
            code_set, uncovered, counts, deadline - elapsed - 5.0, seed_val
        )
        
        # Verify coverage
        if not result_uncovered:
            if result_size < best_size:
                best_size = result_size
                best_code = list(result_code)
        
        seed_val += 1
        
        # If we've reached a very good size, do a few more tries
        if best_size <= 490:
            # Try a few more seeds quickly
            for extra_seed in range(seed_val, seed_val + 3):
                elapsed = time.time() - start_time
                if elapsed > deadline - 10:
                    break
                code_set = set(best_code)
                covered = set()
                for c in code_set:
                    for w in BALLS[c]:
                        covered.add(w)
                uncovered = set(range(TOTAL_WORDS)) - covered
                counts = [0] * TOTAL_WORDS
                for c in code_set:
                    for w in BALLS[c]:
                        counts[w] += 1
                
                result_code, result_uncovered, result_size = _local_search_hypergraph(
                    code_set, uncovered, counts, deadline - elapsed - 15.0, extra_seed
                )
                if not result_uncovered and result_size < best_size:
                    best_size = result_size
                    best_code = list(result_code)
            break
    
    # Final trim
    best_code = _trim_redundant(best_code)
    
    # Convert ints back to lists of ints
    if not best_code:
        return []
        
    return [int_to_vec(c) for c in best_code]
