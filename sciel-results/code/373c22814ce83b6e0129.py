"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the score for a set U of nonnegative integers."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    n = len(U)
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0 or diff_size <= 0:
        return 0.0
    
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / denom
    return score


def _greedy_construct(base, max_elems=200):
    """
    Greedily construct a set U starting with {0} by adding elements that
    maximize the increase in |U-U| while minimizing the increase in |U+U|.
    
    We use a base-B representation approach: candidates are numbers with
    specific binary patterns to control carry behavior in addition.
    """
    U = [0]
    U_set = {0}
    sumset = {0}
    diffset = {0}
    
    # Generate candidate elements based on base-B binary patterns
    # Focus on numbers that are sums of distinct powers of base
    candidates = []
    max_val = base ** 8  # Limit the range
    
    # Generate all numbers up to max_val that have sparse binary/base representations
    for val in range(1, min(max_val, 100000)):
        # Count number of 1-bits (or non-zero digits in base)
        bits = bin(val).count('1')
        if bits <= 4:  # Sparse representation
            candidates.append(val)
    
    random.shuffle(candidates)
    
    for candidate in candidates:
        if len(U) >= max_elems:
            break
        
        # Compute new sumset and diffset sizes
        new_sumset_size = 0
        new_diffset_size = 0
        
        # Efficiently estimate by counting new elements
        added_sum = 0
        added_diff = 0
        
        for a in U:
            s = a + candidate
            if s not in sumset:
                added_sum += 1
            d1 = a - candidate
            d2 = candidate - a
            if d1 not in diffset:
                added_diff += 1
            if d2 not in diffset and d2 != d1:
                added_diff += 1
        
        # Also need to account for sum of candidate with itself
        s2 = candidate + candidate
        if s2 not in sumset and s2 != candidate:  # avoid double count
            added_sum += 1
        # diff of candidate with itself is 0, already in diffset
        
        new_sum_size = len(sumset) + added_sum
        new_diff_size = len(diffset) + added_diff
        
        if new_sum_size == 0:
            continue
        
        # Objective: maximize log(diff_size) - log(sum_size) approximately
        # Or maximize diff_size / sum_size
        ratio_new = new_diff_size / new_sum_size
        ratio_old = len(diffset) / len(sumset) if len(sumset) > 0 else 0
        
        # We want to maximize the ratio
        if ratio_new > ratio_old * 0.95:  # Allow slight decrease for exploration
            U.append(candidate)
            U_set.add(candidate)
            # Update sumset
            for a in U:
                sumset.add(a + candidate)
            # Update diffset
            for a in U:
                diffset.add(a - candidate)
                diffset.add(candidate - a)
    
    return sorted(U)


def _construct_by_base_powers(base, num_powers=6):
    """
    Construct U as a set of numbers that are sums of distinct powers of base,
    chosen to maximize the difference/sum set ratio.
    
    This creates a structure where additions don't cause carries (if we use
    base 2 and only one bit per position), making |U+U| grow slowly.
    """
    # Start with {0}
    U = [0]
    
    # For base 2: use numbers with single bits
    # For general base: use numbers that are single powers of base
    
    powers = [base ** i for i in range(num_powers)]
    
    # Greedily add powers that improve the ratio
    U_set = {0}
    sumset = {0}
    diffset = {0}
    
    for p in powers:
        # Check if adding p improves the ratio
        added_sum = 0
        added_diff = 0
        
        for a in U_set:
            s = a + p
            if s not in sumset:
                added_sum += 1
            d1 = a - p
            d2 = p - a
            if d1 not in diffset:
                added_diff += 1
            if d2 not in diffset and d2 != d1:
                added_diff += 1
        
        s2 = p + p
        if s2 not in sumset and p != 0:
            added_sum += 1
        
        new_sum_size = len(sumset) + added_sum
        new_diff_size = len(diffset) + added_diff
        
        if new_sum_size > 0 and new_diff_size > 0:
            ratio_new = new_diff_size / new_sum_size
            ratio_old = len(diffset) / len(sumset) if len(sumset) > 0 else 0
            
            if ratio_new >= ratio_old:
                U_set.add(p)
                for a in U_set:
                    sumset.add(a + p)
                for a in U_set:
                    diffset.add(a - p)
                    diffset.add(p - a)
    
    return sorted(U_set)


def _random_search(max_iterations=500, max_elems=100, base=2):
    """
    Random search: generate random sparse sets and evaluate them.
    """
    best_U = [0]
    best_score = _score_set(best_U)
    
    for _ in range(max_iterations):
        # Generate a random sparse set
        U = {0}
        for _ in range(random.randint(5, max_elems)):
            # Random number with sparse binary representation
            num_bits = random.randint(1, 8)
            positions = random.sample(range(1, 12), num_bits)
            val = sum(1 << p for p in positions)
            U.add(val)
        
        U = sorted(U)
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    return best_U


def construct():
    """
    Main construction function. Uses a combination of strategies:
    1. Greedy construction with sparse binary representations
    2. Base-power construction
    3. Random search
    Returns the best set found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_U = [0]
    best_score = _score_set(best_U)
    
    # Strategy 1: Greedy construction with different bases
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        U = _greedy_construct(base, max_elems=50)
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Strategy 2: Base-power construction
    for base in [2, 3, 4]:
        if time.time() > deadline:
            break
        for num_powers in range(4, 10):
            U = _construct_by_base_powers(base, num_powers)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Strategy 3: Random search
    if time.time() < deadline - 10:
        remaining_iters = int((deadline - time.time()) / 0.01)  # estimate
        remaining_iters = min(remaining_iters, 1000)
        U = _random_search(max_iterations=remaining_iters, max_elems=50, base=2)
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U = [0] + best_U
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
