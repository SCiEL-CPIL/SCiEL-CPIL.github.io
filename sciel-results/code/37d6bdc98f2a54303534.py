import time
import random
import itertools
from math import log, floor

def score_set(U):
    """
    Compute the score for a given set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def generate_geometric_set(base, k):
    """Generate {0, floor(base^1), ..., floor(base^k)}"""
    U = [0]
    val = 1
    for i in range(1, k + 1):
        val = floor(val * base)
        if val <= 0:
            val = 1
        U.append(val)
    return sorted(set(U))

def local_search(U, deadline, max_iterations=2000):
    """
    Local search to improve the score of set U.
    Uses a mix of element replacement and addition/removal.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    current_score = score_set(U)
    best_score = current_score
    best_set = U[:]
    
    # If initial is invalid, try to fix it or return best known
    if current_score < 0:
        return -1, [0, 1]

    no_improvement_count = 0
    max_no_improvement = 100
    
    # Pre-compute some properties for faster neighbor generation? 
    # For n~10-20, O(n^2) is fine.
    
    while time.time() < deadline - 5 and no_improvement_count < max_no_improvement:
        improved = False
        move_type = random.random()
        
        # Generate candidate values based on current set structure
        # We prefer values that are "structurally compatible" (e.g., near powers of 2, 3)
        max_val = max(U)
        candidates = set()
        
        # 1. Geometric neighbors
        for base in [1.5, 2, 2.5, 3]:
            for k in range(1, 10):
                val = floor(max_val * (base ** k))
                if val > 0:
                    candidates.add(val)
                val2 = floor(max_val / (base ** k))
                if val2 > 0:
                    candidates.add(val2)
        
        # 2. Sums and differences of existing elements (to maintain structure)
        U_list = U
        for i in range(len(U_list)):
            for j in range(len(U_list)):
                s = U_list[i] + U_list[j]
                if s > 0 and s <= max_val * 4:
                    candidates.add(s)
                d = abs(U_list[i] - U_list[j])
                if d > 0 and d <= max_val * 2:
                    candidates.add(d)
        
        # 3. Random small perturbations around existing elements
        for u in U:
            if u == 0: continue
            for delta in [-3, -2, -1, 1, 2, 3]:
                v = u + delta
                if v > 0:
                    candidates.add(v)
        
        # 4. Random values in the range
        for _ in range(20):
            candidates.add(random.randint(1, max_val * 3))
        
        candidates = [c for c in candidates if c not in U]
        
        if move_type < 0.6:
            # Replace an element
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1) # Don't replace 0
                old_val = U[idx]
                # Try a few random candidates
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U): continue # Ensure distinctness
                    new_score = score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        elif move_type < 0.8:
            # Add an element
            if len(U) < 15 and candidates:
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    new_U = U + [new_val]
                    new_U = sorted(set(new_U))
                    new_score = score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        else:
            # Remove an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True

        if improved:
            no_improvement_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
        else:
            no_improvement_count += 1

    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    best_score = -1
    best_set = [0, 1, 3]
    
    # Phase 1: Evaluate strong structured candidates
    # Binary sets are known to be strong
    candidates = []
    # Powers of 2
    for k in range(3, 15):
        candidates.append(generate_geometric_set(2, k))
    # Powers of 3
    for k in range(3, 12):
        candidates.append(generate_geometric_set(3, k))
    # Mixed small sets
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    candidates.append([0, 1, 3, 9, 27, 81, 243, 729])
    candidates.append([0, 1, 4, 16, 64, 256, 1024])
    
    for U in candidates:
        if time.time() > deadline - 20:
            break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # Phase 2: Local Search from best candidate
    # Run multiple restarts with slight perturbations to escape local minima
    # We have ~120s left. Each local search run should take < 10s.
    
    # We'll run local search on the best candidate found
    if time.time() < deadline - 10:
        s2, set2 = local_search(best_set, deadline - 5)
        if s2 > best_score:
            best_score = s2
            best_set = set2

    # Phase 3: Try a few random restarts with local search
    # This helps if the structured candidates are trapped in a local optimum
    # that a random start might escape or find a better one.
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Run ~3-5 random restarts
        num_restarts = min(5, int(remaining_time / 20))
        for _ in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Generate a random structured-like set
            base = random.choice([2, 3, 4])
            k = random.randint(4, 10)
            U_rand = generate_geometric_set(base, k)
            
            # Perturb it
            for i in range(len(U_rand)):
                if i == 0: continue
                if random.random() < 0.3:
                    U_rand[i] += random.randint(-5, 5)
                    if U_rand[i] <= 0:
                        U_rand[i] = 1
            
            U_rand = sorted(set(U_rand))
            if 0 not in U_rand:
                U_rand = [0] + U_rand
            
            s_r, set_r = local_search(U_rand, deadline - 5)
            if s_r > best_score:
                best_score = s_r
                best_set = set_r

    return best_set

# EVOLVE-BLOCK-END
