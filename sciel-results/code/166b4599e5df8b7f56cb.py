"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _count_3aps(S):
    """Count distinct 3-term APs (positive difference, counted once) in set S."""
    S_set = set(S)
    count = 0
    for a in S:
        for b in S:
            if a < b:
                c = 2*b - a
                if c > b and c in S_set:
                    count += 1
    return count

def _has_4ap_with(S, new_elem):
    """Check if adding new_elem to S creates a 4-term AP."""
    S_set = set(S) | {new_elem}
    for d in range(1, 257):
        for a in S_set:
            if a + 3*d <= 256:
                if all(a + j*d in S_set for j in range(4)):
                    # Check if this 4-AP includes new_elem
                    if any(a + j*d == new_elem for j in range(4)):
                        return True
    return False

def _has_4ap(S):
    """Check if S contains any 4-term AP."""
    S_set = set(S)
    for d in range(1, 65):  # max d for 4-AP in [1,256] is (256-1)//3 = 85, but be safe
        for a in range(1, 257 - 3*d + 1):
            if all(a + j*d in S_set for j in range(4)):
                return True
    return False

def _base4_construction():
    """Generate a 4-AP-free set using base-4 digit construction (no digits 0 or 3)."""
    result = []
    for x in range(1, 257):
        # Convert to base 4
        digits = []
        n = x
        while n > 0:
            digits.append(n % 4)
            n //= 4
        # Check if all digits are in {1, 2}
        if all(d in (1, 2) for d in digits):
            result.append(x)
    return result

def _greedy_initial():
    """Greedy construction that tries to maximize 3-APs."""
    selected = set()
    # Start with base-4 construction as seed
    base = _base4_construction()
    selected = set(base)
    
    # Try adding elements that create 3-APs without creating 4-APs
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    for x in candidates:
        if x in selected:
            continue
        # Check if adding x creates a 4-AP
        if _has_4ap_with(selected, x):
            continue
        # Check if adding x creates any 3-AP
        new_3aps = 0
        for y in selected:
            if x < y:
                # Check if x, y, z forms 3-AP: z = 2y - x
                z = 2*y - x
                if z > y and z in selected:
                    new_3aps += 1
            elif y < x:
                # Check if y, x, z forms 3-AP: z = 2x - y
                z = 2*x - y
                if z > x and z in selected:
                    new_3aps += 1
                # Check if w, y, x forms 3-AP: w = 2y - x
                w = 2*y - x
                if w < y and w in selected:
                    new_3aps += 1
        if new_3aps > 0:
            selected.add(x)
    
    return selected

def _sa_polish(S, time_limit=45.0):
    """Simulated annealing to maximize 3-APs while maintaining no 4-APs."""
    S = set(S)
    best_S = set(S)
    best_score = _count_3aps(best_S)
    
    start_time = time.time()
    T = 10.0  # initial temperature
    cooling = 0.995
    
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < 0.01:
            break
        
        # Random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            if len(S) >= 256:
                continue
            candidates = [x for x in range(1, 257) if x not in S]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_S = S | {x}
            
            # Check 4-AP constraint
            if _has_4ap_with(S, x):
                continue
            
            new_score = _count_3aps(new_S)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
        
        elif move_type == 'remove':
            if len(S) <= 1:
                continue
            x = random.choice(list(S))
            new_S = S - {x}
            new_score = _count_3aps(new_S)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
        
        elif move_type == 'swap':
            if len(S) < 2:
                continue
            in_S = list(S)
            not_in_S = [x for x in range(1, 257) if x not in S]
            if not in_S or not not_in_S:
                continue
            x_out = random.choice(in_S)
            x_in = random.choice(not_in_S)
            new_S = (S - {x_out}) | {x_in}
            
            # Check 4-AP constraint for the new element
            if _has_4ap_with(S - {x_out}, x_in):
                continue
            
            new_score = _count_3aps(new_S)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
    
    return best_S

def construct():
    """Construct a subset of [1,256] with no 4-term AP, maximizing 3-term APs."""
    deadline = time.time() + 85.0  # Safety margin
    
    # Phase 1: Get a good initial solution
    S = _greedy_initial()
    best_S = set(S)
    best_score = _count_3aps(best_S)
    
    # Phase 2: Simulated annealing polish
    if time.time() < deadline:
        polished = _sa_polish(S, time_limit=min(45.0, deadline - time.time()))
        score = _count_3aps(polished)
        if score > best_score:
            best_S = polished
            best_score = score
    
    # Phase 3: Try a few more restarts if time permits
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 3:
        S2 = _greedy_initial()
        if time.time() < deadline - 5.0:
            polished2 = _sa_polish(S2, time_limit=min(20.0, deadline - time.time() - 5.0))
            score2 = _count_3aps(polished2)
            if score2 > best_score:
                best_S = polished2
                best_score = score2
        restarts += 1
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
