import time
import random

START_TIME = time.time()
TIME_LIMIT = 55.0

def compute_structures(S_list):
    """Compute triple sums and pair sums indexed by position."""
    n = len(S_list)
    triple_sums = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple_sums.add(S_list[i] + S_list[j] + S_list[k])
    pair_sums_by_idx = {}
    for i in range(n):
        pairs = set()
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                pairs.add(S_list[j] + S_list[k])
        pair_sums_by_idx[i] = pairs
    return triple_sums, pair_sums_by_idx

def can_add(S_list, x, triple_sums, pair_sums_by_idx):
    """Check if x can be added to S_list without violating constraints."""
    # Check: x should not divide any existing triple sum
    for t in triple_sums:
        if t % x == 0:
            return False
    # Check: for each existing element a, x + (pair sum of two others) should not be divisible by a
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for p in pair_sums_by_idx[i]:
            if (x + p) % a == 0:
                return False
    return True

def add_element(S_list, x, triple_sums, pair_sums_by_idx):
    """Add element x to S_list and update structures."""
    S_list.append(x)
    n = len(S_list) - 1
    # Update triple sums: new triples include x and two existing elements
    for i in range(n):
        for j in range(i+1, n):
            triple_sums.add(x + S_list[i] + S_list[j])
    # Update pair sums for existing elements
    for i in range(n):
        for j in range(n):
            if j != i:
                pair_sums_by_idx[i].add(x + S_list[j])
    # Create pair sums for new element
    pf = set()
    for i in range(n):
        for j in range(i+1, n):
            pf.add(S_list[i] + S_list[j])
    pair_sums_by_idx[n] = pf

def construct():
    global START_TIME
    START_TIME = time.time()
    best_size = 0
    best_list = []
    numbers = list(range(1, 193))
    
    # Try multiple different ordering strategies
    orderings = [
        'large_first', 'small_first', 'random', 'alternating'
    ]
    
    while time.time() - START_TIME < TIME_LIMIT:
        current_list = []
        triple_sums = set()
        pair_sums_by_idx = {}
        
        # Choose an ordering strategy
        strategy = random.choice(orderings)
        
        if strategy == 'large_first':
            candidates = sorted(numbers, reverse=True)
        elif strategy == 'small_first':
            candidates = sorted(numbers)
        elif strategy == 'alternating':
            candidates = []
            large = [x for x in numbers if x > 96]
            small = [x for x in numbers if x <= 96]
            random.shuffle(large)
            random.shuffle(small)
            for i in range(max(len(large), len(small))):
                if i < len(large):
                    candidates.append(large[i])
                if i < len(small):
                    candidates.append(small[i])
        else:  # random
            candidates = list(numbers)
            random.shuffle(candidates)
        
        # Greedy phase
        for x in candidates:
            if time.time() - START_TIME >= TIME_LIMIT:
                break
            if can_add(current_list, x, triple_sums, pair_sums_by_idx):
                add_element(current_list, x, triple_sums, pair_sums_by_idx)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_list = list(current_list)
            if best_size == 48:
                break
        
        # Local search phase: try removing one element and adding others
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            for i in range(len(current_list)):
                if time.time() - START_TIME >= TIME_LIMIT:
                    break
                
                removed = current_list[i]
                temp_list = [x for x in current_list if x != removed]
                temp_set = set(temp_list)
                
                # Recompute structures for temp_list
                temp_triple, temp_pairs = compute_structures(temp_list)
                
                remaining = [x for x in numbers if x not in temp_set]
                random.shuffle(remaining)
                
                added = 0
                for x in remaining:
                    if can_add(temp_list, x, temp_triple, temp_pairs):
                        add_element(temp_list, x, temp_triple, temp_pairs)
                        temp_set.add(x)
                        added += 1
                        if added >= 2:
                            break
                
                if added >= 2:
                    current_list = temp_list
                    triple_sums = temp_triple
                    pair_sums_by_idx = temp_pairs
                    improved = True
                    break
            
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_list = list(current_list)
                if best_size == 48:
                    break
        
        if best_size == 48:
            break
    
    return best_list
