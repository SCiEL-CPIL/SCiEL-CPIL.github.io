import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    # Handle non-standard cases with a simple greedy
    if q != 3 or n != 8 or radius != 1:
        return _general_greedy(q, n, radius, deadline)
    
    # Specialized for q=3, n=8, radius=1
    return _construct_ternary_8_1(deadline)


def _general_greedy(q, n, radius, deadline):
    """Simple greedy for general cases."""
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors within radius
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
    
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    covered = np.zeros(total_words, dtype=bool)
    code = []
    uncovered_count = total_words
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        scores = np.sum(~covered[neighbors], axis=1)
        max_score = np.max(scores)
        candidates = np.where(scores == max_score)[0]
        best_idx = candidates[0]
        code.append(int(best_idx))
        nbs = neighbors[best_idx]
        newly = np.sum(~covered[nbs])
        covered[nbs] = True
        uncovered_count -= newly
    
    # Convert to digits
    result = []
    for w in code:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
    return result


def _construct_ternary_8_1(deadline):
    """
    Multi-start stochastic local search for K_3(8,1).
    
    Strategy:
    1. Get initial solutions via greedy with different tie-breaking
    2. Apply delete-and-repair local search to improve
    3. Keep the best solution found
    """
    total_words = 6561
    n = 8
    q = 3
    powers = [3 ** i for i in range(n)]
    
    # Precompute neighbors (each word + its 16 single-digit variations)
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
    
    neighbors = np.array(neighbors_list, dtype=np.int32)
    neighbor_set = [set(nb) for nb in neighbors_list]
    
    best_code = None
    best_size = float('inf')
    
    # Phase 1: Generate initial solutions via greedy with different strategies
    initial_codes = []
    
    # Deterministic greedy (lowest index)
    code = _greedy(neighbors, total_words, strategy='first')
    initial_codes.append(code)
    
    # Greedy with random tie-breaking
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        code = _greedy(neighbors, total_words, strategy='random')
        initial_codes.append(code)
    
    # Phase 2: Local search on each initial code
    for init_code in initial_codes:
        if time.time() > deadline - 5:
            break
        
        code = init_code.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to delete each codeword
            for i in range(len(code) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                # Check if removing code[i] leaves all words covered
                # A word is uncovered if it was only covered by code[i]
                # Efficient check: for each neighbor of code[i], count how many
                # other codewords cover it
                
                removed = code.pop(i)
                
                # Check coverage
                uncovered = _find_uncovered(code, neighbors, total_words)
                
                if len(uncovered) == 0:
                    improved = True
                    continue
                elif len(uncovered) <= 3:
                    # Try to repair by adding one codeword
                    repair = _find_repair(uncovered, neighbors, total_words)
                    if repair is not None and repair not in code:
                        code.append(repair)
                        improved = True
                        continue
                elif len(uncovered) <= 10:
                    # Try to repair by adding one or two codewords
                    repair = _find_repair_multi(uncovered, neighbors, total_words, max_add=2)
                    if repair is not None:
                        for r in repair:
                            if r not in code:
                                code.append(r)
                        improved = True
                        continue
                
                # Couldn't improve, put it back
                code.insert(i, removed)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
    
    # Phase 3: Additional local search iterations on the best code
    if best_code is not None and time.time() < deadline - 10:
        code = best_code.copy()
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Random order for diversity
            indices = list(range(len(code)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline - 5:
                    break
                
                removed = code.pop(i)
                uncovered = _find_uncovered(code, neighbors, total_words)
                
                if len(uncovered) == 0:
                    improved = True
                    continue
                elif len(uncovered) <= 5:
                    repair = _find_repair(uncovered, neighbors, total_words)
                    if repair is not None and repair not in code:
                        code.append(repair)
                        improved = True
                        continue
                    elif len(uncovered) <= 10:
                        repair = _find_repair_multi(uncovered, neighbors, total_words, max_add=2)
                        if repair is not None:
                            for r in repair:
                                if r not in code:
                                    code.append(r)
                            improved = True
                            continue
                
                code.insert(i, removed)
            
            if not improved:
                break
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
    
    if best_code is None:
        # Fallback
        return _fallback_ternary_8_1()
    
    # Convert to digits
    result = []
    for w in best_code:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
    return result


def _greedy(neighbors, total_words, strategy='first'):
    """Greedy set cover."""
    covered = np.zeros(total_words, dtype=bool)
    code = []
    uncovered_count = total_words
    
    while uncovered_count > 0:
        scores = np.sum(~covered[neighbors], axis=1)
        max_score = np.max(scores)
        candidates = np.where(scores == max_score)[0]
        
        if strategy == 'first':
            best_idx = candidates[0]
        else:
            best_idx = random.choice(candidates.tolist())
        
        code.append(int(best_idx))
        nbs = neighbors[best_idx]
        newly = np.sum(~covered[nbs])
        covered[nbs] = True
        uncovered_count -= newly
    
    return code


def _find_uncovered(code, neighbors, total_words):
    """Find words not covered by any codeword in code."""
    if not code:
        return list(range(total_words))
    
    covered = np.zeros(total_words, dtype=bool)
    for c in code:
        covered[neighbors[c]] = True
    
    return np.where(~covered)[0].tolist()


def _find_repair(uncovered, neighbors, total_words):
    """Find a single codeword that covers all uncovered words."""
    if not uncovered:
        return None
    
    # For each candidate, check if it covers all uncovered words
    # A candidate c covers word w if w is in neighbors[c]
    # Equivalently, c is in neighbors[w]
    
    # Find candidates that cover the first uncovered word
    first = uncovered[0]
    candidates = neighbors[first].tolist()
    
    # For each candidate, check if it covers all uncovered
    for c in candidates:
        c_neighbors = set(neighbors[c].tolist())
        if all(w in c_neighbors for w in uncovered):
            return c
    
    return None


def _find_repair_multi(uncovered, neighbors, total_words, max_add=2):
    """Find up to max_add codewords that cover all uncovered words."""
    if not uncovered:
        return None
    
    if max_add == 1:
        return _find_repair(uncovered, neighbors, total_words)
    
    # Greedy: repeatedly pick the best covering codeword
    remaining = set(uncovered)
    added = []
    
    for _ in range(max_add):
        if not remaining:
            break
        
        # Find the codeword that covers the most remaining words
        # Candidates are neighbors of the first remaining word
        first = next(iter(remaining))
        candidates = neighbors[first].tolist()
        
        best_c = None
        best_count = -1
        for c in candidates:
            c_neighbors = set(neighbors[c].tolist())
            count = len(remaining & c_neighbors)
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c is None or best_count == 0:
            return None
        
        added.append(best_c)
        c_neighbors = set(neighbors[best_c].tolist())
        remaining -= c_neighbors
    
    if remaining:
        return None
    
    return added


def _fallback_ternary_8_1():
    """Fallback using tetracode construction."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
