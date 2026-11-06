import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Greedy Construction
    seq, prods = _greedy_construct()
    best_seq = seq[:]
    best_len = len(seq)
    
    rng = random.Random(42)
    
    # 2. Local Search
    # We iterate, trying to improve the sequence.
    # Each iteration:
    # - Randomly remove 1-2 elements.
    # - Try to insert missing elements to maximize length.
    
    while time.time() < deadline - 10.0: # Leave 10s for backtracking
        current_seq = best_seq[:]
        current_prods = _get_products(current_seq)
        
        # Perturbation
        if len(current_seq) > 2:
            num_removals = rng.randint(1, 2)
            for _ in range(num_removals):
                if len(current_seq) > 1:
                    idx = rng.randint(0, len(current_seq) - 1)
                    current_seq.pop(idx)
            # Recompute products after removal
            current_prods = _get_products(current_seq)
            
        # Improvement Phase: Try to insert any missing element
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            candidates = [x for x in range(2, 129) if x not in current_seq]
            rng.shuffle(candidates)
            
            for x in candidates:
                new_seq, new_prods = _try_insert(current_seq, current_prods, x)
                if new_seq is not None:
                    current_seq = new_seq
                    current_prods = new_prods
                    improved = True
                    break # Restart candidate loop
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
    # 3. Backtracking Search
    # Try to find a sequence longer than best_len
    # We run a DFS with a time limit.
    if time.time() < deadline - 1.0:
        bt_seq = _backtracking_search(best_len, deadline - 1.0)
        if bt_seq and len(bt_seq) > best_len:
            best_seq = bt_seq
            best_len = len(bt_seq)
            
    return best_seq

def _greedy_construct():
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values, products

def _get_products(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def _try_insert(seq, prods, x):
    """
    Try to insert x into the sorted sequence seq.
    Returns (new_seq, new_prods) if valid, else (None, None).
    """
    if not seq:
        return [x], {x}
        
    # Find insertion point
    pos = 0
    for i in range(len(seq)):
        if x < seq[i]:
            pos = i
            break
        else:
            pos = i + 1
            
    new_seq = seq[:pos] + [x] + seq[pos:]
    
    # Check validity:
    # New subarrays are those that include x.
    # These are:
    # 1. Subarrays ending at x: [seq[pos-1], x], [seq[pos-2], seq[pos-1], x], ... [seq[0], ..., x]
    # 2. Subarrays starting at x: [x], [x, seq[pos]], [x, seq[pos], seq[pos+1]], ...
    # Note: [x] is covered by both (or just one).
    
    # Let's compute all new products involving x.
    new_prods = set()
    
    # Left side (ending at x)
    prod = x
    if prod in prods:
        return None, None
    new_prods.add(prod)
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        if prod in prods:
            return None, None
        new_prods.add(prod)
        
    # Right side (starting at x)
    # We already added [x]. Now add [x, seq[pos]], etc.
    # Note: The product [x] is already in new_prods.
    # We need to check products [x, seq[pos]], [x, seq[pos], seq[pos+1]], ...
    # These are distinct from the left side products unless the left side was empty (pos=0).
    # If pos=0, left side was just [x]. Right side adds [x, seq[0]], etc.
    # If pos>0, left side added [x], [seq[pos-1], x], ...
    # Right side adds [x, seq[pos]], [x, seq[pos], seq[pos+1]], ...
    # Are there overlaps between left and right new products?
    # Left: x * (product of suffix of seq[:pos])
    # Right: x * (product of prefix of seq[pos:])
    # They are distinct unless the suffix product equals the prefix product.
    # But we already checked against `prods` (old products).
    # We also need to ensure new products don't collide with *each other*.
    # If P_left = P_right, then x * S_left = x * S_right => S_left = S_right.
    # S_left is a product of a suffix of the left part.
    # S_right is a product of a prefix of the right part.
    # If S_left == S_right, then this product would have been in `prods`?
    # No, S_left and S_right are subarrays of the *old* sequence?
    # S_left is a suffix of seq[:pos]. This is a subarray of the old sequence.
    # S_right is a prefix of seq[pos:]. This is a subarray of the old sequence.
    # If S_left == S_right, then that value is in `prods`.
    # But we checked `prod` (which is x * S_left) against `prods`.
    # Wait. `prods` contains products of the *old* sequence.
    # `x * S_left` is a NEW product.
    # If `x * S_left == x * S_right`, then `S_left == S_right`.
    # This doesn't mean `x * S_left` is in `prods`.
    # It means two NEW products are equal.
    # So we must check for collisions among new products.
    
    # Let's just collect all new products and check for duplicates within them, and against old prods.
    # Actually, the check `if prod in prods` handles collision with old.
    # We need to handle collision between new products.
    
    # Let's re-evaluate.
    # New products are:
    # L_i = x * prod(seq[pos-i-1:pos]) for i=0..pos-1 (where L_0=x)
    # R_j = x * prod(seq[pos:pos+j+1]) for j=0..len(seq)-pos-1 (where R_0=x)
    # Note L_0 = R_0 = x.
    # We need all L_i and R_j to be distinct from each other and from `prods`.
    
    # We already checked L_i against `prods`.
    # We need to check R_j against `prods`.
    # And we need to check if any L_i == R_j (for i>0, j>0).
    
    # Let's redo the check properly.
    
    # 1. Check all new products against old `prods`.
    # 2. Check if any two new products are equal.
    
    # Let's generate the list of new products.
    new_products_list = []
    
    # Left side
    p = x
    new_products_list.append(p)
    for i in range(pos - 1, -1, -1):
        p *= seq[i]
        new_products_list.append(p)
        
    # Right side (excluding x which is already in list)
    p = x
    for j in range(pos, len(seq)):
        p *= seq[j]
        new_products_list.append(p)
        
    # Check for duplicates in new_products_list
    if len(set(new_products_list)) != len(new_products_list):
        return None, None
        
    # Check against old prods
    for p in new_products_list:
        if p in prods:
            return None, None
            
    # If all checks pass, update prods
    new_prods = prods.union(new_products_list)
    return new_seq, new_prods

def _backtracking_search(current_best_len, deadline):
    """
    Attempt to find a sequence longer than current_best_len using DFS.
    """
    best_found = []
    
    def dfs(seq, prods, last_val):
        nonlocal best_found
        if time.time() > deadline:
            return
        
        if len(seq) > len(best_found):
            best_found = seq[:]
            
        # Pruning: if even adding all remaining numbers doesn't beat best, stop
        # Max possible length is 128 - last_val + len(seq)
        if len(seq) + (128 - last_val) <= len(best_found):
            return
            
        # Try next values
        # Heuristic: Try larger values first? Or smaller?
        # Let's try larger values first as they are less likely to cause collisions.
        for x in range(128, last_val, -1):
            new_seq, new_prods = _try_append(seq, prods, x)
            if new_seq is not None:
                dfs(new_seq, new_prods, x)
                
    dfs([], set(), 1)
    return best_found

def _try_append(seq, prods, x):
    """
    Try to append x to the end of seq.
    """
    if not seq:
        return [x], {x}
        
    if x <= seq[-1]:
        return None, None
        
    # New products are suffixes of the new sequence ending at x.
    # x, x*seq[-1], x*seq[-1]*seq[-2], ...
    new_products = []
    prod = x
    new_products.append(prod)
    if prod in prods:
        return None, None
        
    for i in range(len(seq) - 1, -1, -1):
        prod *= seq[i]
        new_products.append(prod)
        if prod in prods:
            return None, None
            
    # Check for duplicates among new products?
    # Since seq is strictly increasing and > 1, the products are strictly increasing.
    # x < x*seq[-1] < ...
    # So no duplicates among new products.
    
    new_prods = prods.union(new_products)
    return seq + [x], new_prods

