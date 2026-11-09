import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Generate initial candidates
    candidates = []

    # Random starts
    for _ in range(30):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        candidates.append(lst)

    # Structured starts
    candidates.append(list(range(1, 1 + 31 * 16, 16)))  # AP with d=16
    candidates.append(list(range(1, 33)))  # Dense at start
    candidates.append(list(range(481, 513)))  # Dense at end

    # Greedy construction: add elements that maximize new AP differences
    def greedy_construct():
        """Build a set greedily by adding elements that create new AP differences."""
        current = set()
        mask = 0
        # Start with a random seed element
        current.add(random.randint(1, MAX_VAL))
        mask |= (1 << current.pop())
        
        # Try multiple random starts and keep the best
        best_greedy = None
        best_greedy_score = -1
        
        for _ in range(5):
            cur_set = set()
            cur_mask = 0
            cur_set.add(random.randint(1, MAX_VAL))
            cur_mask |= (1 << next(iter(cur_set)))
            
            for _ in range(N - 1):
                best_elem = None
                best_new_score = -1
                # Sample candidate elements
                for _ in range(50):
                    elem = random.randint(1, MAX_VAL)
                    if elem in cur_set:
                        continue
                    new_mask = cur_mask | (1 << elem)
                    s = score_mask(new_mask)
                    if s > best_new_score:
                        best_new_score = s
                        best_elem = elem
                if best_elem is not None:
                    cur_set.add(best_elem)
                    cur_mask |= (1 << best_elem)
            
            if score_mask(cur_mask) > best_greedy_score:
                best_greedy_score = score_mask(cur_mask)
                best_greedy = list(cur_set)
        
        return best_greedy

    if time.time() < deadline - 30:
        candidates.append(greedy_construct())

    # Evaluate initial candidates
    best_score = -1
    best_list = None
    best_mask = 0

    for lst in candidates:
        if lst is None:
            continue
        m = list_to_mask(lst)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    # Simulated Annealing with restart on stagnation
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask
    current_set = set(current_list)

    T = 10.0
    cooling_rate = 0.9998
    min_T = 0.1
    stagnation_limit = 2000
    stagnation_count = 0

    # Precompute all possible d values for faster checking
    all_d = list(range(1, 256))

    while time.time() < deadline:
        # Generate a neighbor
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]

        # Pick a new value
        while True:
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val or new_val in current_set:
                continue
            break

        new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
        new_score = score_mask(new_mask)
        delta = new_score - current_score

        if delta > 0:
            accept = True
            stagnation_count = 0
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = random.random() < math.exp(delta / T)

        if accept:
            current_list[idx] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            current_score = new_score
            current_mask = new_mask

            if current_score > best_score:
                best_score = current_score
                best_list = current_list[:]
                best_mask = current_mask
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1

        T *= cooling_rate
        if T < min_T:
            T = min_T

        # Restart if stuck
        if stagnation_count > stagnation_limit:
            # Try a new random start
            if time.time() < deadline - 10:
                new_lst = random.sample(range(1, MAX_VAL + 1), N)
                new_m = list_to_mask(new_lst)
                new_s = score_mask(new_m)
                if new_s >= best_score - 2:
                    current_list = new_lst[:]
                    current_score = new_s
                    current_mask = new_m
                    current_set = set(current_list)
                    T = 5.0
                    stagnation_count = 0

    return best_list
# EVOLVE-BLOCK-END
