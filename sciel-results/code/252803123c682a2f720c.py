import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not A or not B:
        return False
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

def _greedy_build_pair(n=256, max_size=64, rng=None, strategy='asc'):
    """Build a pair using greedy with incremental checking."""
    if rng is None:
        rng = random.Random()
    
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    set_A = set()
    set_B = set()
    
    def can_add_to_A(x):
        if x in set_A:
            return False
        for a in set_A:
            if a + x in sums_A:
                return False
        if 2 * x in sums_A:
            return False
        for a in set_A:
            d1 = x - a
            d2 = a - x
            if d1 in diffs_B or d2 in diffs_B:
                return False
            if d1 in diffs_A or d2 in diffs_A:
                return False
        return True
    
    def can_add_to_B(x):
        if x in set_B:
            return False
        for b in set_B:
            if b + x in sums_B:
                return False
        if 2 * x in sums_B:
                return False
        for b in set_B:
            d1 = x - b
            d2 = b - x
            if d1 in diffs_A or d2 in diffs_A:
                return False
            if d1 in diffs_B or d2 in diffs_B:
                return False
        return True
    
    def add_to_A(x):
        for a in set_A:
            sums_A.add(a + x)
        sums_A.add(2 * x)
        for a in set_A:
            diffs_A.add(x - a)
            diffs_A.add(a - x)
        set_A.add(x)
        A.append(x)
    
    def add_to_B(x):
        for b in set_B:
            sums_B.add(b + x)
        sums_B.add(2 * x)
        for b in set_B:
            diffs_B.add(x - b)
            diffs_B.add(b - x)
        set_B.add(x)
        B.append(x)
    
    candidates = list(range(1, n + 1))
    if strategy == 'desc':
        candidates.sort(reverse=True)
    elif strategy == 'rand':
        rng.shuffle(candidates)
    else:
        candidates.sort()
    
    # Phase 1: Build A
    for x in candidates:
        if len(A) >= max_size:
            break
        if can_add_to_A(x):
            add_to_A(x)
    
    # Phase 2: Build B
    candidates_b = list(range(1, n + 1))
    if strategy == 'desc':
        candidates_b.sort(reverse=True)
    elif strategy == 'rand':
        rng.shuffle(candidates_b)
    else:
        candidates_b.sort()
    
    for x in candidates_b:
        if len(B) >= max_size:
            break
        if can_add_to_B(x):
            add_to_B(x)
    
    # Phase 3: Extend A
    candidates_a2 = list(range(1, n + 1))
    if strategy == 'desc':
        candidates_a2.sort(reverse=True)
    elif strategy == 'rand':
        rng.shuffle(candidates_a2)
    else:
        candidates_a2.sort()
    
    for x in candidates_a2:
        if len(A) >= max_size:
            break
        if can_add_to_A(x):
            add_to_A(x)
    
    # Phase 4: Extend B
    candidates_b2 = list(range(1, n + 1))
    if strategy == 'desc':
        candidates_b2.sort(reverse=True)
    elif strategy == 'rand':
        rng.shuffle(candidates_b2)
    else:
        candidates_b2.sort()
    
    for x in candidates_b2:
        if len(B) >= max_size:
            break
        if can_add_to_B(x):
            add_to_B(x)
    
    return A, B

def _local_search_fast(A, B, n=256, max_size=64, rng=None, max_time=5.0):
    """Fast local search using incremental validity checking."""
    if rng is None:
        rng = random.Random()
    
    start = time.time()
    
    A_list = list(A)
    B_list = list(B)
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    set_A = set(A_list)
    set_B = set(B_list)
    
    for a in A_list:
        for b in A_list:
            if a <= b:
                sums_A.add(a + b)
    for b in B_list:
        for c in B_list:
            if b <= c:
                sums_B.add(b + c)
    for i, a in enumerate(A_list):
        for j, b in enumerate(A_list):
            if i != j:
                diffs_A.add(b - a)
    for i, b in enumerate(B_list):
        for j, c in enumerate(B_list):
            if i != j:
                diffs_B.add(c - b)
    
    def can_add_to_A(x):
        if x in set_A:
            return False
        for a in set_A:
            if a + x in sums_A:
                return False
        if 2 * x in sums_A:
            return False
        for a in set_A:
            d1 = x - a
            d2 = a - x
            if d1 in diffs_B or d2 in diffs_B:
                return False
            if d1 in diffs_A or d2 in diffs_A:
                return False
        return True
    
    def can_add_to_B(x):
        if x in set_B:
            return False
        for b in set_B:
            if b + x in sums_B:
                return False
        if 2 * x in sums_B:
            return False
        for b in set_B:
            d1 = x - b
            d2 = b - x
            if d1 in diffs_A or d2 in diffs_A:
                return False
            if d1 in diffs_B or d2 in diffs_B:
                return False
        return True
    
    def add_to_A(x):
        for a in set_A:
            sums_A.add(a + x)
        sums_A.add(2 * x)
        for a in set_A:
            diffs_A.add(x - a)
            diffs_A.add(a - x)
        set_A.add(x)
        A_list.append(x)
    
    def add_to_B(x):
        for b in set_B:
            sums_B.add(b + x)
        sums_B.add(2 * x)
        for b in set_B:
            diffs_B.add(x - b)
            diffs_B.add(b - x)
        set_B.add(x)
        B_list.append(x)
    
    best_score = len(A_list) + len(B_list)
    
    while time.time() - start < max_time:
        improved = False
        
        # Try to add to A
        if len(A_list) < max_size:
            for x in rng.sample(range(1, n + 1), min(50, n)):
                if can_add_to_A(x):
                    add_to_A(x)
                    best_score = len(A_list) + len(B_list)
                    improved = True
                    break
        
        # Try to add to B
        if len(B_list) < max_size:
            for x in rng.sample(range(1, n + 1), min(50, n)):
                if can_add_to_B(x):
                    add_to_B(x)
                    best_score = len(A_list) + len(B_list)
                    improved = True
                    break
        
        if not improved:
            # Try swap: remove one, add another
            if len(A_list) > 1:
                idx = rng.randint(0, len(A_list) - 1)
                old_x = A_list[idx]
                # Remove old_x
                set_A.discard(old_x)
                A_list.remove(old_x)
                # Rebuild sums and diffs for A
                sums_A = set()
                diffs_A = set()
                for i, a in enumerate(A_list):
                    for j, b in enumerate(A_list):
                        if i <= j:
                            sums_A.add(a + b)
                        if i != j:
                            diffs_A.add(b - a)
                
                found = False
                for new_x in rng.sample(range(1, n + 1), min(100, n)):
                    if can_add_to_A(new_x):
                        add_to_A(new_x)
                        best_score = len(A_list) + len(B_list)
                        found = True
                        break
                
                if not found:
                    # Restore old_x
                    add_to_A(old_x)
            
            if len(B_list) > 1:
                idx = rng.randint(0, len(B_list) - 1)
                old_x = B_list[idx]
                set_B.discard(old_x)
                B_list.remove(old_x)
                sums_B = set()
                diffs_B = set()
                for i, b in enumerate(B_list):
                    for j, c in enumerate(B_list):
                        if i <= j:
                            sums_B.add(b + c)
                        if i != j:
                            diffs_B.add(c - b)
                
                found = False
                for new_x in rng.sample(range(1, n + 1), min(100, n)):
                    if can_add_to_B(new_x):
                        add_to_B(new_x)
                        best_score = len(A_list) + len(B_list)
                        found = True
                        break
                
                if not found:
                    add_to_B(old_x)
            
            if not improved:
                break
    
    return A_list, B_list

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random(42)
    
    strategies = ['asc', 'desc', 'rand', 'rand', 'rand', 'rand']
    
    for restart in range(100000):
        if time.time() > deadline:
            break
        
        strategy = strategies[restart % len(strategies)]
        A, B = _greedy_build_pair(n, max_size, rng, strategy)
        
        if not A: A = [1]
        if not B: B = [2]
        
        score = len(A) + len(B)
        
        # Quick validity check
        if _is_valid_pair(A, B):
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        # Local search with time budget
        remaining = deadline - time.time()
        if remaining > 2:
            ls_time = min(remaining - 1, 3.0)
            A_ls, B_ls = _local_search_fast(A, B, n, max_size, rng, max_time=ls_time)
            
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
        
        # Early exit if we've hit the theoretical max
        if best_score >= 2 * max_size:
            break
    
    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]
