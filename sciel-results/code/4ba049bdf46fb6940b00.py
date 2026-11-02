"""Rewritten program: optimize finite set U to maximize sumset/difference-set exponent lower bound."""
import time
import math
import random
import itertools

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock limit; use ~145s for safety

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _score(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    # Also include negatives of differences (U-U already includes them if we do a-b for all pairs)
    # Actually U-U = {a-b : a,b in U}, which is symmetric around 0.
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _greedy_superincreasing(k):
    """Build a superincreasing sequence of length k starting with 0."""
    U = [0]
    current = 1
    for i in range(1, k):
        # Next element must be > sum of all previous to ensure no sum collisions across blocks
        s = sum(U)
        U.append(s + 1)
    return U

def _greedy_staircase(n_blocks, block_size):
    """
    Build a staircase: blocks of size block_size, with gaps growing exponentially.
    Block i starts at offset_i, and within each block we place block_size elements.
    The gap between block i and i+1 is large enough to prevent cross-block sum collisions.
    """
    U = [0]
    offset = 1
    for i in range(n_blocks):
        # Place block_size elements in this block
        for j in range(1, block_size):
            U.append(offset + j)
        # Next block offset: ensure gap is large enough
        # Max sum within block i is roughly 2*(offset + block_size)
        # To avoid cross-block collisions, next offset should be > max possible sum from previous blocks
        # Simplified: make offset grow fast
        offset += block_size * (block_size + 1) + 1
    return sorted(set(U))

def _local_search(U, max_iter=5000):
    """
    Local search: try adding/removing/shifting elements to improve score.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    best_score = _score(U)
    best_U = list(U)
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        if random.random() < 0.3:
            # Try adding a new element
            max_u = max(U)
            candidates = list(range(1, max_u + 2))
            # Bias towards larger values to increase max_u (denominator) but also potentially increase |U-U|
            # Actually we want to balance. Let's try random positions.
            new_val = random.choice(candidates)
            if new_val not in U:
                new_U = sorted(U + [new_val])
                s = _score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
        elif random.random() < 0.5:
            # Try removing an element (not 0)
            removable = [x for x in U if x != 0]
            if removable:
                remove_val = random.choice(removable)
                new_U = sorted([x for x in U if x != remove_val])
                s = _score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
        else:
            # Try shifting an element
            shiftable = [x for x in U if x != 0]
            if shiftable:
                shift_val = random.choice(shiftable)
                delta = random.choice([-2, -1, 1, 2, 3, 5, 10])
                new_val = shift_val + delta
                if new_val >= 0 and new_val not in U:
                    new_U = sorted([new_val if x == shift_val else x for x in U])
                    s = _score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        U = new_U
    return best_U, best_score

def _prime_step_gaps(n_elements):
    """
    Build a sequence using prime-step gaps.
    Start with 0, then add gaps that are primes.
    This can create non-superincreasing sequences with interesting sum/diff structure.
    """
    # Generate first n_elements primes
    def gen_primes(count):
        primes = []
        candidate = 2
        while len(primes) < count:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    primes = gen_primes(n_elements)
    U = [0]
    for p in primes[1:]:
        U.append(U[-1] + p)
    return U

def _random_construct(n_elements):
    """Random construction with some structure."""
    if n_elements <= 1:
        return [0]
    # Start with 0, then pick random increasing values
    U = [0]
    max_val = n_elements * n_elements
    for _ in range(n_elements - 1):
        val = random.randint(1, max_val)
        if val not in U:
            U.append(val)
    return sorted(U)

def construct():
    """
    Main entry point. Returns a list of nonnegative integers containing 0.
    Score is 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    """
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Strategy 1: Superincreasing sequences of various sizes
    for k in range(2, 20):
        if _remaining() < 5.0:
            break
        U = _greedy_superincreasing(k)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Strategy 2: Prime-step gaps
    for k in range(2, 20):
        if _remaining() < 5.0:
            break
        U = _prime_step_gaps(k)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        # Also try local search on prime-step
        if _remaining() > 10.0:
            U_ls, s_ls = _local_search(U, max_iter=2000)
            if s_ls > best_score:
                best_score = s_ls
                best_U = list(U_ls)
    
    # Strategy 3: Staircase constructions
    for n_blocks in range(2, 8):
        if _remaining() < 5.0:
            break
        for block_size in range(2, 8):
            if _remaining() < 5.0:
                break
            U = _greedy_staircase(n_blocks, block_size)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
            # Local search
            if _remaining() > 10.0:
                U_ls, s_ls = _local_search(U, max_iter=2000)
                if s_ls > best_score:
                    best_score = s_ls
                    best_U = list(U_ls)
    
    # Strategy 4: Random constructions with local search
    for trial in range(50):
        if _remaining() < 10.0:
            break
        n = random.randint(3, 15)
        U = _random_construct(n)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        if _remaining() > 15.0:
            U_ls, s_ls = _local_search(U, max_iter=3000)
            if s_ls > best_score:
                best_score = s_ls
                best_U = list(U_ls)
    
    # Strategy 5: Local search from best found so far
    if _remaining() > 20.0:
        U_ls, s_ls = _local_search(best_U, max_iter=10000)
        if s_ls > best_score:
            best_score = s_ls
            best_U = list(U_ls)
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
        best_U = sorted(best_U)
    
    return best_U
