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


def _check_insertion(seq, pos, x, existing_products):
    """
    Check if inserting x at position pos in seq creates any duplicate products.
    Only computes the new products that involve x (O(n) instead of O(n^2)).
    Returns True if valid, False otherwise.
    """
    n = len(seq)
    # New sequence: seq[0..pos-1], x, seq[pos..n-1]
    # A block in the new seq that includes x: starts at some s in [0, pos] and ends at some e in [pos, n-1]
    # where the block in the new sequence is new_seq[s..e] and includes position pos (which is x).
    # So s can be 0..pos, e can be pos..n-1 (in new indexing).
    # But we need to compute the product of new_seq[s..e].
    
    # Precompute: left products ending at pos-1 (in old seq), i.e., prod(seq[i..pos-1]) for i=0..pos-1
    # And right products starting at pos (in old seq), i.e., prod(seq[pos..j]) for j=pos..n-1
    # And x itself.
    
    # A block including x in the new sequence:
    # new_seq = [seq[0], ..., seq[pos-1], x, seq[pos], ..., seq[n-1]]
    # Block from new index s to new index e where s <= pos <= e.
    # If s == pos and e == pos: product = x
    # If s < pos and e == pos: product = prod(seq[s..pos-1]) * x
    # If s == pos and e > pos: product = x * prod(seq[pos..e])
    # If s < pos and e > pos: product = prod(seq[s..pos-1]) * x * prod(seq[pos..e])
    
    # Let's compute:
    # left_prods[i] = prod(seq[i..pos-1]) for i = 0, 1, ..., pos-1 (and left_prods[pos] = 1 for the case s=pos)
    # right_prods[j] = prod(seq[pos..j]) for j = pos, pos+1, ..., n-1 (and right_prods[pos-1] = 1 for the case e=pos-1... wait)
    
    # Actually, let me re-index. In the new sequence:
    # Index 0..pos-1: seq[0]..seq[pos-1]
    # Index pos: x
    # Index pos+1..n: seq[pos]..seq[n-1]
    
    # A block [s..e] in new indices includes x iff s <= pos <= e.
    # Product = prod(new_seq[s..e])
    
    # Let L[i] = prod(new_seq[i..pos-1]) for i = 0..pos-1, and L[pos] = 1
    # Let R[j] = prod(new_seq[pos+1..j]) for j = pos+1..n, and R[pos] = 1
    # Then for s <= pos <= e:
    #   product = L[s] * x * R[e] if s < pos and e > pos
    #   product = L[s] * x if s < pos and e == pos
    #   product = x * R[e] if s == pos and e > pos
    #   product = x if s == pos and e == pos
    
    # Let me just compute all products involving x directly.
    
    # Products ending at x (s from 0 to pos, e = pos):
    # new_seq[s..pos] = seq[s..pos-1] + [x] (if s < pos), or just [x] (if s == pos)
    left_prod = 1
    for s in range(pos - 1, -1, -1):
        left_prod *= seq[s]
        p = left_prod * x
        if p in existing_products:
            return False
    
    # Products starting at x (s = pos, e from pos to n-1 in new indices, i.e., pos+1 to n in old):
    right_prod = 1
    for e in range(pos, n):
        right_prod *= seq[e]
        p = x * right_prod
        if p in existing_products:
            return False
    
    # Products that span across x (s < pos and e > pos in new indices):
    # new_seq[s..e] = seq[s..pos-1] + [x] + seq[pos..e-1] (in old indices, e-1 because new index e corresponds to old index e-1 when e > pos)
    # Wait, let me be more careful.
    
    # New index mapping:
    # new_idx 0..pos-1 -> old_idx 0..pos-1
    # new_idx pos -> x
    # new_idx pos+1..n -> old_idx pos..n-1
    
    # So for s in [0, pos] and e in [pos, n] (new indices), with s < pos and e > pos:
    # product = prod(seq[s..pos-1]) * x * prod(seq[pos..e-1])
    
    # Let me precompute left products: L[s] = prod(seq[s..pos-1]) for s = 0..pos-1
    # And right products: R[e] = prod(seq[pos..e]) for e = pos..n-1
    
    left_prods = {}
    lp = 1
    for s in range(pos - 1, -1, -1):
        lp *= seq[s]
        left_prods[s] = lp
    
    right_prods = {}
    rp = 1
    for e in range(pos, n):
        rp *= seq[e]
        right_prods[e] = rp
    
    # Now check all spanning products
    for s in range(0, pos):
        for e in range(pos, n):
            p = left_prods[s] * x * right_prods[e]
            if p in existing_products:
                return False
    
    # Also need to check that x itself is not in existing_products
    if x in existing_products:
        return False
    
    return True


def _try_insert_fast(seq, x, existing_products):
    """
    Try to insert x into the sorted sequence seq.
    Uses fast O(n) check for products involving x.
    Returns (new_seq, new_products, True) if successful, (None, None, False) otherwise.
    """
    if not seq:
        return [x], {x}, True
    
    # Find insertion point to keep seq sorted
    pos = 0
    for i in range(len(seq)):
        if x < seq[i]:
            pos = i
            break
        else:
            pos = i + 1
    
    # Check if x is already in seq
    if pos < len(seq) and seq[pos] == x:
        return None, None, False
    if pos > 0 and seq[pos-1] == x:
        return None, None, False
    
    # Check validity using fast method
    if not _check_insertion(seq, pos, x, existing_products):
        return None, None, False
    
    # Construct new sequence and update products
    new_seq = seq[:pos] + [x] + seq[pos:]
    
    # Compute new products involving x
    n = len(seq)
    new_prods = set()
    
    # Products ending at x
    left_prod = 1
    for s in range(pos - 1, -1, -1):
        left_prod *= seq[s]
        new_prods.add(left_prod * x)
    new_prods.add(x)
    
    # Products starting at x
    right_prod = 1
    for e in range(pos, n):
        right_prod *= seq[e]
        new_prods.add(x * right_prod)
    
    # Products spanning across x
    left_prods = {}
    lp = 1
    for s in range(pos - 1, -1, -1):
        lp *= seq[s]
        left_prods[s] = lp
    
    right_prods = {}
    rp = 1
    for e in range(pos, n):
        rp *= seq[e]
        right_prods[e] = rp
    
    for s in range(0, pos):
        for e in range(pos, n):
            new_prods.add(left_prods[s] * x * right_prods[e])
    
    updated_products = existing_products.union(new_prods)
    
    return new_seq, updated_products, True


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
        products = best_products.copy()
        current_len = len(seq)
        
        # Randomly remove 1-3 elements to allow rearrangement
        if current_len > 3:
            num_removals = rng.randint(1, 3)
            for _ in range(num_removals):
                if len(seq) > 2:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
            # Recompute products after removal
            products = _get_all_products(seq)
        
        # Try to add back removed or new elements
        available = all_numbers - set(seq)
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        improved = False
        for x in avail_list:
            new_seq, new_prods, ok = _try_insert_fast(seq, x, products)
            if ok:
                seq = new_seq
                products = new_prods
                improved = True
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_products = products
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
            products = _get_all_products(seq)
            available = all_numbers - set(seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            added = False
            for x in avail_list[:20]:  # try first 20 shuffled
                new_seq, new_prods, ok = _try_insert_fast(seq, x, products)
                if ok and len(new_seq) > len(seq):
                    seq = new_seq
                    products = new_prods
                    added = True
                    break
                elif ok:
                    # Same length but different element, might help later
                    if rng.random() < 0.3:
                        seq = new_seq
                        products = new_prods
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
                products = _get_all_products(seq)
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_products = products
            best_len = len(seq)
    
    return best_seq

# EVOLVE-BLOCK-END
