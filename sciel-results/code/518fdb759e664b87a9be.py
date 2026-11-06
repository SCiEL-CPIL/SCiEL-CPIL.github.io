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


def _try_insert_fast(seq, x, existing_products):
    """
    Try to insert x into sorted seq at its unique position.
    Only checks products that involve x for conflicts with existing_products.
    Returns (new_seq, True) if valid, (None, False) otherwise.
    """
    if not seq:
        return [x], True
    
    # Find insertion position
    pos = 0
    for i in range(len(seq)):
        if x < seq[i]:
            pos = i
            break
        else:
            pos = i + 1
    
    # Check if x is already in seq
    if pos < len(seq) and seq[pos] == x:
        return None, False
    if pos > 0 and seq[pos-1] == x:
        return None, False
    
    n = len(seq)
    
    # Compute all new products that involve x
    # New sequence: seq[0..pos-1], x, seq[pos..n-1]
    # Blocks involving x: start s in [0, pos], end e in [pos, n-1] (new indices)
    
    # Precompute left suffix products: prod(seq[i..pos-1]) for i = 0..pos-1
    # left_prod[i] = prod(seq[i..pos-1])
    left_prods = {}
    if pos > 0:
        lp = 1
        for i in range(pos - 1, -1, -1):
            lp *= seq[i]
            left_prods[i] = lp
    
    # Precompute right prefix products: prod(seq[pos..j]) for j = pos..n-1
    # right_prod[j] = prod(seq[pos..j])
    right_prods = {}
    if pos < n:
        rp = 1
        for j in range(pos, n):
            rp *= seq[j]
            right_prods[j] = rp
    
    # Now check all blocks involving x
    new_products = set()
    
    # Singleton: just x
    new_products.add(x)
    if x in existing_products:
        return None, False
    
    # Blocks starting at or before pos (in new index) and ending at pos
    # i.e., left part + x
    for s in range(pos):
        # product = prod(seq[s..pos-1]) * x
        p = left_prods[s] * x
        if p in existing_products or p in new_products:
            return None, False
        new_products.add(p)
    
    # Blocks starting at pos and ending after pos
    # i.e., x + right part
    for e in range(pos, n):
        # product = x * prod(seq[pos..e])
        p = x * right_prods[e]
        if p in existing_products or p in new_products:
            return None, False
        new_products.add(p)
    
    # Blocks starting before pos and ending after pos
    # i.e., left part + x + right part
    for s in range(pos):
        for e in range(pos, n):
            p = left_prods[s] * x * right_prods[e]
            if p in existing_products or p in new_products:
                return None, False
            new_products.add(p)
    
    new_seq = seq[:pos] + [x] + seq[pos:]
    return new_seq, True


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Get initial solution via greedy
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    best_products = _get_all_products(best_seq)
    
    rng = random.Random(42)
    
    all_numbers = set(range(2, 129))
    
    while time.time() < deadline:
        # Start from best_seq
        seq = best_seq[:]
        current_len = len(seq)
        current_products = best_products.copy()
        
        # Randomly remove 1-3 elements to allow rearrangement
        if current_len > 3:
            num_removals = rng.randint(1, 3)
            for _ in range(num_removals):
                if len(seq) > 2:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
            # Recompute products after removal
            current_products = _get_all_products(seq)
            current_len = len(seq)
        
        # Try to add back removed or new elements
        available = all_numbers - set(seq)
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        improved = False
        for x in avail_list:
            if time.time() >= deadline:
                break
            new_seq, ok = _try_insert_fast(seq, x, current_products)
            if ok:
                seq = new_seq
                current_products = _get_all_products(seq)
                improved = True
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_len = len(seq)
                    best_products = current_products.copy()
        
        # Also try local search: swap adjacent elements and see if we can add more
        # Try removing one element and adding a different one
        for _ in range(5):
            if time.time() >= deadline:
                break
            if len(seq) < 2:
                break
            idx = rng.randint(0, len(seq) - 1)
            removed = seq.pop(idx)
            current_products = _get_all_products(seq)
            
            available = all_numbers - set(seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            added = False
            for x in avail_list[:20]:
                new_seq, ok = _try_insert_fast(seq, x, current_products)
                if ok and len(new_seq) > len(seq):
                    seq = new_seq
                    current_products = _get_all_products(seq)
                    added = True
                    if len(seq) > best_len:
                        best_seq = seq[:]
                        best_len = len(seq)
                        best_products = current_products.copy()
                    break
                elif ok:
                    # Same length but different element, might help later
                    if rng.random() < 0.3:
                        seq = new_seq
                        current_products = _get_all_products(seq)
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
                current_products = _get_all_products(seq)
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
            best_products = current_products.copy()
    
    return best_seq

# EVOLVE-BLOCK-END
