import time
import random

N = 17
NUM_INPUTS = 1 << N  # 131072

def _build_initial_vectors():
    vectors = []
    for i in range(N):
        v = 0
        block = 1 << i
        pattern = ((1 << block) - 1) << block
        num_repeats = NUM_INPUTS // (2 * block)
        for r in range(num_repeats):
            v |= pattern << (r * 2 * block)
        vectors.append(v)
    return vectors

_INITIAL_VECTORS = _build_initial_vectors()

def _validate(comparators):
    state = list(_INITIAL_VECTORS)
    for (i, j) in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
    for m in range(1, N):
        if state[m] & ~state[m-1]:
            return False
    return True

def _merge(n, m, offset=0):
    if n == 0:
        return []
    if n == 1 and m == 1:
        return [(offset, offset + 1)]
    if n == 1:
        return [(offset, offset + 1)] + _merge(1, m - 1, offset + 1)
    
    comps = []
    total = n + m
    for i in range(0, total - 1, 2):
        comps.append((offset + i, offset + i + 1))
    for i in range(1, total - 1, 2):
        comps.append((offset + i, offset + i + 1))
    
    n_even = (n + 1) // 2
    m_even = (m + 1) // 2
    n_odd = n // 2
    m_odd = m // 2
    
    even_comps = _merge(n_even, m_even, 0)
    even_mapped = [(offset + 2*a, offset + 2*b) for (a, b) in even_comps]
    
    odd_comps = _merge(n_odd, m_odd, 0)
    odd_mapped = [(offset + 2*a + 1, offset + 2*b + 1) for (a, b) in odd_comps]
    
    return comps + even_mapped + odd_mapped

def _sort_network(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    h = n // 2
    left = _sort_network(h)
    right = [(i + h, j + h) for (i, j) in _sort_network(n - h)]
    merge = _merge(h, n - h, 0)
    return left + right + merge

def construct():
    start_time = time.time()
    deadline = start_time + 48
    
    # Build starting network
    comparators = _sort_network(N)
    
    # Validate
    assert _validate(comparators), f"Starting network invalid! Count: {len(comparators)}"
    
    best_comparators = list(comparators)
    best_count = len(comparators)
    
    random.seed(42)
    
    # Phase 1: Multiple random greedy removal passes
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        comps = list(best_comparators)
        # Try random removals
        for _ in range(len(comps) + 10):
            if time.time() > deadline - 5:
                break
            if not comps:
                break
            idx = random.randint(0, len(comps) - 1)
            new_comps = comps[:idx] + comps[idx+1:]
            if _validate(new_comps):
                comps = new_comps
        
        if len(comps) < best_count:
            best_count = len(comps)
            best_comparators = list(comps)
    
    # Phase 2: Final greedy pass (reverse order)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(best_comparators) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            new_comps = best_comparators[:idx] + best_comparators[idx+1:]
            if _validate(new_comps):
                best_comparators = new_comps
                changed = True
                break
    
    # Phase 3: Local search - try moving comparators
    while time.time() < deadline - 1:
        improved = False
        comps = list(best_comparators)
        
        for idx in range(len(comps)):
            if time.time() > deadline - 1:
                break
            # Try moving comparator idx to a random position
            new_idx = random.randint(0, len(comps) - 1)
            if new_idx == idx:
                continue
            moved = comps[idx]
            temp = comps[:idx] + comps[idx+1:]
            if new_idx > idx:
                new_idx -= 1
            temp = temp[:new_idx] + [moved] + temp[new_idx:]
            
            if _validate(temp):
                # Try to remove some comparators
                changed = True
                while changed and time.time() < deadline - 1:
                    changed = False
                    for j in range(len(temp) - 1, -1, -1):
                        if time.time() > deadline - 1:
                            break
                        new_comps = temp[:j] + temp[j+1:]
                        if _validate(new_comps):
                            temp = new_comps
                            changed = True
                            break
                if len(temp) < best_count:
                    best_count = len(temp)
                    best_comparators = list(temp)
                    improved = True
        
        if not improved:
            # Try a few more random moves
            for _ in range(100):
                if time.time() > deadline - 1:
                    break
                idx = random.randint(0, len(comps) - 1)
                new_idx = random.randint(0, len(comps) - 1)
                if new_idx == idx:
                    continue
                moved = comps[idx]
                temp = comps[:idx] + comps[idx+1:]
                if new_idx > idx:
                    new_idx -= 1
                temp = temp[:new_idx] + [moved] + temp[new_idx:]
                
                if _validate(temp):
                    changed = True
                    while changed and time.time() < deadline - 1:
                        changed = False
                        for j in range(len(temp) - 1, -1, -1):
                            if time.time() > deadline - 1:
                                break
                            new_comps = temp[:j] + temp[j+1:]
                            if _validate(new_comps):
                                temp = new_comps
                                changed = True
                                break
                    if len(temp) < best_count:
                        best_count = len(temp)
                        best_comparators = list(temp)
                        improved = True
            
            if not improved:
                break
    
    return best_comparators
