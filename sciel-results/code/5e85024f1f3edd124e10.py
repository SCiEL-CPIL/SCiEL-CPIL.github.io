import time
import random

N = 256
DEADLINE = 78.0  # safety margin before 90s kill

def count_3aps(A):
    """Count distinct 3-term APs (positive diff, counted once)."""
    A_set = set(A)
    count = 0
    for i, a in enumerate(A):
        for b in A[i+1:]:
            c = 2*b - a
            if c in A_set:
                count += 1
    return count

def count_4ap_violations(A):
    """Count 4-term AP violations."""
    A_set = set(A)
    count = 0
    for i, a in enumerate(A):
        for b in A[i+1:]:
            for c in A[i+2:]:
                d = 2*c - b
                if d in A_set and d > c:
                    count += 1
    return count

def has_4ap(A):
    """Check if A contains a nonconstant 4-term AP."""
    A_set = set(A)
    for i, a in enumerate(A):
        for b in A[i+1:]:
            d = b - a
            if a + 3*d <= N and a + 2*d in A_set and a + 3*d in A_set:
                return True
    return False

def base_salem_spencer():
    """
    Construct a base set using 3-adic digit representation.
    Numbers with digits in {0,1} in base 3 (up to 3^5=243) form a set
    with no 3-term AP, hence no 4-term AP. We extend to [1,256].
    """
    # Numbers whose base-3 representation uses only digits 0 and 1
    # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243
    # So we can form numbers up to 1+3+9+27+81+243 = 364, but we only go to 256
    base_set = set()
    for mask in range(1, 1 << 6):  # 6 digits (0 through 3^5)
        val = 0
        for bit in range(6):
            if mask & (1 << bit):
                val += 3 ** bit
        if 1 <= val <= N:
            base_set.add(val)
    return sorted(base_set)

def construct():
    start_time = time.time()
    
    # Start with the Salem-Spencer base set (guaranteed no 3-AP, hence no 4-AP)
    # This gives us a valid starting point with 0 3-APs.
    # Better: use a modified construction that allows some 3-APs but no 4-APs.
    
    # Alternative: start with all numbers and iteratively remove 4-AP violators,
    # then use local search to maximize 3-APs.
    
    # Let's use a smarter initial construction:
    # Numbers with base-3 digits in {0,1,2} but avoiding patterns that create 4-APs.
    # Actually, let's use a different approach: start with a larger set and prune.
    
    # Strategy: Start with all numbers, greedily remove to eliminate 4-APs,
    # then do simulated annealing to maximize 3-APs.
    
    # Better initial: use the fact that if we take numbers with even 2-adic valuation
    # (i.e., divisible by 2 but not 4, or divisible by 4 but not 8, etc.), we can
    # create structured sets.
    
    # Let me try: take all numbers whose base-3 representation has no digit equal to 2
    # in positions that would create 4-APs. This is complex.
    
    # Simplest robust approach: greedy construction that avoids 4-APs while
    # trying to include numbers that form 3-APs.
    
    # Greedy: process numbers 1..256, add x if it doesn't create a 4-AP.
    # But also try to add numbers that complete 3-APs.
    
    # Let's use a two-phase approach:
    # Phase 1: Build a valid set (no 4-AP) using a heuristic that favors 3-AP formation
    # Phase 2: Local search (swap moves) to maximize 3-AP count
    
    random.seed(42)
    
    # Phase 1: Greedy construction with 3-AP awareness
    A = set()
    # Precompute potential 3-AP partners
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        for d in range(1, (N - x) // 3 + 1):
            # Check x, x+d, x+2d, x+3d
            if x + d in A and x + 2*d in A and x + 3*d in A:
                creates_4ap = True
                break
            # Check x-3d, x-2d, x-d, x
            if x - 3*d >= 1 and x - 3*d in A and x - 2*d in A and x - d in A:
                creates_4ap = True
                break
            # Check x-d, x, x+d, x+2d
            if x - d >= 1 and x - d in A and x + d in A and x + 2*d in A:
                creates_4ap = True
                break
            # Check x-2d, x-d, x, x+d
            if x - 2*d >= 1 and x - 2*d in A and x - d in A and x + d in A:
                creates_4ap = True
                break
        if not creates_4ap:
            A.add(x)
    
    current_3ap = count_3aps(sorted(A))
    
    # Phase 2: Simulated annealing to maximize 3-APs
    # Moves: add a number, remove a number, swap a number
    best_A = set(A)
    best_3ap = current_3ap
    T = 50.0  # initial temperature
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        if T < 0.1:
            break
        
        # Choose a random move
        move_type = random.random()
        new_A = set(A)
        
        if move_type < 0.4:
            # Add a random number not in A
            candidates = [x for x in range(1, N + 1) if x not in A]
            if candidates:
                x = random.choice(candidates)
                new_A.add(x)
        elif move_type < 0.7:
            # Remove a random number from A
            if A:
                x = random.choice(list(A))
                new_A.discard(x)
        else:
            # Swap: remove one, add another
            if A:
                x_remove = random.choice(list(A))
                candidates = [x for x in range(1, N + 1) if x not in A]
                if candidates:
                    x_add = random.choice(candidates)
                    new_A.discard(x_remove)
                    new_A.add(x_add)
        
        # Check validity (no 4-AP)
        if has_4ap(sorted(new_A)):
            continue
        
        new_3ap = count_3aps(sorted(new_A))
        delta = new_3ap - current_3ap
        
        if delta > 0 or random.random() < (10 ** (delta / T)):
            A = new_A
            current_3ap = new_3ap
            if current_3ap > best_3ap:
                best_3ap = current_3ap
                best_A = set(A)
        
        T *= cooling
    
    # Return the best valid set found
    return sorted(best_A)
