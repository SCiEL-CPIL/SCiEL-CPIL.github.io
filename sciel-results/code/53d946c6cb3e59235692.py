"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _compute_max_mult(A):
    """Compute the maximum representation multiplicity for targets [128,256]."""
    A_set = set(A)
    counts = Counter()
    for a in A_set:
        for b in A_set:
            if a <= b:
                s = a + b
                if 128 <= s <= 256:
                    counts[s] += 1
    # Check coverage
    for t in range(128, 257):
        if counts[t] == 0:
            return float('inf')
    return max(counts.values())

def _is_valid(A):
    """Check if A covers all targets [128,256]."""
    A_set = set(A)
    for t in range(128, 257):
        found = False
        for a in A_set:
            if a <= t and (t - a) in A_set and (t - a) >= a:
                found = True
                break
        if not found:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Initial structured set: use a mix to ensure good coverage
    # Try several different initial structures
    random.seed(42)
    
    best_A = None
    best_score = float('inf')
    
    # Initial candidate: dense set in middle range
    # [0, 128] range, need sums in [128, 256]
    # If A = [0, 128], every sum from 0 to 256 is covered, but multiplicities are high
    # Better: use a set that's spread out
    
    # Strategy 1: Arithmetic progression
    # A = {0, k, 2k, ..., mk} where mk <= 128
    # For k=4: {0,4,8,...,128} = 33 elements, sums range 0-256
    
    # Strategy 2: Two interleaved progressions
    # Strategy 3: Random with density
    
    initial_sets = []
    
    # Dense middle: [32, 96] gives sums [64, 192] - not enough for 256
    # Need to reach 256, so need elements up to ~128
    # [0, 128] full range works but high multiplicity
    
    # Try: {0, 1, ..., 32} U {64, 65, ..., 96} - two blocks
    # Sums: [0,64] from first block, [64,128] cross, [128,192] second block
    # Doesn't reach 256
    
    # Better: {0, 32, 64, 96, 128} plus fillers
    # Or: use a set with good additive properties
    
    # Let's try a structured approach:
    # A = {i * d for i in range(k)} union some offsets
    
    # Simple good start: every 4th element from 0 to 128
    initial_sets.append(list(range(0, 129, 4)))  # 33 elements
    
    # Every 3rd: 0,3,6,...,126 - 43 elements
    initial_sets.append(list(range(0, 129, 3)))
    
    # Mixed: even numbers 0-64, odd numbers 64-128
    initial_sets.append(list(range(0, 65, 2)) + list(range(65, 129, 2)))
    
    # Two blocks: [0,32] and [96,128]
    initial_sets.append(list(range(0, 33)) + list(range(96, 129)))
    
    # Random dense set
    for _ in range(5):
        size = random.randint(35, 55)
        A_rand = sorted(random.sample(range(129), size))
        initial_sets.append(A_rand)
    
    # Evaluate initial sets
    for A in initial_sets:
        if time.time() > deadline:
            break
        score = _compute_max_mult(A)
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    # Local search: try adding/removing/swapping elements
    A = list(best_A)
    current_score = best_score
    
    # Precompute for efficiency
    def _get_counts(A_set):
        counts = Counter()
        for a in A_set:
            for b in A_set:
                if a <= b:
                    s = a + b
                    if 128 <= s <= 256:
                        counts[s] += 1
        return counts
    
    A_set = set(A)
    counts = _get_counts(A_set)
    
    # Local search iterations
    max_iters = 20000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Try removing an element
            if len(A) <= 20:
                continue
            idx = random.randint(0, len(A) - 1)
            elem = A[idx]
            new_A = A[:idx] + A[idx+1:]
            new_A_set = set(new_A)
            
            # Quick check: does removal break coverage?
            # Only check if elem is critical
            new_counts = _get_counts(new_A_set)
            valid = True
            for t in range(128, 257):
                if new_counts[t] == 0:
                    valid = False
                    break
            if not valid:
                continue
            new_score = max(new_counts.values())
            if new_score < current_score:
                A = new_A
                A_set = new_A_set
                counts = new_counts
                current_score = new_score
                best_score = min(best_score, current_score)
                best_A = list(A)
                
        elif move_type < 0.7:
            # Try adding an element
            candidates = [x for x in range(129) if x not in A_set]
            if not candidates:
                continue
            elem = random.choice(candidates)
            new_A_set = A_set | {elem}
            new_counts = _get_counts(new_A_set)
            new_score = max(new_counts.values())
            if new_score < current_score:
                A.append(elem)
                A_set = new_A_set
                counts = new_counts
                current_score = new_score
                best_score = min(best_score, current_score)
                best_A = list(A)
                
        else:
            # Try swapping: remove one, add another
            if len(A) < 20:
                continue
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            candidates = [x for x in range(129) if x not in A_set]
            if not candidates:
                continue
            new_elem = random.choice(candidates)
            new_A_set = A_set - {old_elem} | {new_elem}
            new_counts = _get_counts(new_A_set)
            valid = True
            for t in range(128, 257):
                if new_counts[t] == 0:
                    valid = False
                    break
            if not valid:
                continue
            new_score = max(new_counts.values())
            if new_score < current_score:
                A[idx] = new_elem
                A_set = new_A_set
                counts = new_counts
                current_score = new_score
                best_score = min(best_score, current_score)
                best_A = list(A)
    
    # Final verification
    if best_A is None or not _is_valid(best_A):
        return list(range(64, 129))
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
