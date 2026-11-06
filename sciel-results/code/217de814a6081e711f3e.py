"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_construct():
    """Deterministic greedy: iterate 2..128, add if no product conflict."""
    values, products = [], set()
    for x in range(2, 129):
        tails = []
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        tails.append(x)  # singleton
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _get_all_products(seq):
    """Return set of all nonempty consecutive block products."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products


def _is_valid(seq):
    """Check if seq is strictly increasing in [2,128] with distinct block products."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if any(x < 2 or x > 128 for x in seq):
        return False
    products = _get_all_products(seq)
    return len(products) == len(seq) * (len(seq) + 1) // 2


def _try_insert(seq, x):
    """Try to insert x into seq maintaining increasing order and distinct products.
    Returns (new_seq, True) if successful, (None, False) otherwise."""
    # Find position to insert x
    pos = -1
    for i in range(len(seq)):
        if seq[i] > x:
            pos = i
            break
    if pos == -1:
        pos = len(seq)
    # Check x not already in seq
    if pos < len(seq) and seq[pos] == x:
        return None, False
    if pos > 0 and seq[pos-1] == x:
        return None, False
    
    new_seq = seq[:pos] + [x] + seq[pos:]
    if not _is_valid(new_seq):
        return None, False
    return new_seq, True


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Get initial solution via greedy
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    
    rng = random.Random(42)
    
    all_numbers = set(range(2, 129))
    
    while time.time() < deadline:
        # Start from best_seq
        seq = best_seq[:]
        current_len = len(seq)
        
        # Randomly remove 1-3 elements to allow rearrangement
        if current_len > 3:
            num_removals = rng.randint(1, 3)
            for _ in range(num_removals):
                if len(seq) > 2:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
        
        # Try to add back removed or new elements
        available = all_numbers - set(seq)
        # Shuffle available
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        improved = False
        for x in avail_list:
            new_seq, ok = _try_insert(seq, x)
            if ok:
                seq = new_seq
                improved = True
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_len = len(seq)
        
        # Also try local search: swap adjacent elements and see if we can add more
        # Try removing one element and adding a different one
        for _ in range(5):  # limited attempts
            if time.time() >= deadline:
                break
            if len(seq) < 2:
                break
            idx = rng.randint(0, len(seq) - 1)
            removed = seq.pop(idx)
            available = all_numbers - set(seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            added = False
            for x in avail_list[:20]:  # try first 20 shuffled
                new_seq, ok = _try_insert(seq, x)
                if ok and len(new_seq) > len(seq):
                    seq = new_seq
                    added = True
                    break
                elif ok:
                    # Same length but different element, might help later
                    if rng.random() < 0.3:
                        seq = new_seq
                        added = True
                        break
            if not added:
                # Put back the removed element
                pos = 0
                for i in range(len(seq)):
                    if seq[i] > removed:
                        pos = i
                        break
                seq.insert(pos, removed)
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
    
    return best_seq

# EVOLVE-BLOCK-END
