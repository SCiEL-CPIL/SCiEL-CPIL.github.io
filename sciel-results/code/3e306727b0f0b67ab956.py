import time
import random

# Primes in [2, 128]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
_P_COUNT = len(_PRIMES)

# Pre-compute exponent vectors for each integer in [2, 128]
# _EXP[i] is a tuple of length _P_COUNT
_EXP = [None] * 129
for i in range(2, 129):
    n = i
    vec = [0] * _P_COUNT
    for idx, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[idx] += 1
            n //= p
    _EXP[i] = tuple(vec)

def _is_valid(seq):
    """Check if the sequence has unique block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        # Start with zero vector
        prod_vec = [0] * _P_COUNT
        for j in range(i, n):
            # Add _EXP[seq[j]] to prod_vec
            ej = _EXP[seq[j]]
            for k in range(_P_COUNT):
                prod_vec[k] += ej[k]
            key = tuple(prod_vec)
            if key in products:
                return False
            products.add(key)
    return True

def _greedy_construct():
    """Greedy baseline using exponent vectors."""
    values = []
    products = set()
    for x in range(2, 129):
        # Calculate new suffix vectors
        # If we append x, the new vectors are:
        # v_x, v_x + v_last, v_x + v_last + v_prev, ...
        new_vecs = []
        # Start with v_x
        cur_vec = list(_EXP[x])
        new_vecs.append(tuple(cur_vec))
        
        valid = True
        # Add previous elements in reverse
        for a in reversed(values):
            ea = _EXP[a]
            for k in range(_P_COUNT):
                cur_vec[k] += ea[k]
            t = tuple(cur_vec)
            if t in products:
                valid = False
                break
            new_vecs.append(t)
        
        if not valid:
            continue
            
        # Check if any new_vec is in products
        # Since we checked in the loop, if valid is True, no collision with existing products?
        # Wait. The loop checks `if t in products`.
        # But `new_vecs[0]` is `v_x`. We didn't check `v_x` against `products` in the loop start?
        # No, `t` is added to `new_vecs` and checked in the loop.
        # However, `v_x` is the first element.
        # In the loop:
        # 1. cur_vec = _EXP[x]. t = tuple(cur_vec). Check t in products?
        #    My code above appends t to new_vecs, but doesn't check it in the loop if the loop body is for the *next* element?
        #    Let's re-verify the logic.
        #    
        #    cur_vec = _EXP[x]
        #    new_vecs.append(tuple(cur_vec))  # This is v_x
        #    
        #    for a in reversed(values):
        #        cur_vec += _EXP[a]
        #        t = tuple(cur_vec)
        #        if t in products: valid = False; break
        #        new_vecs.append(t)
        #        
        #    This checks v_x * v_last, v_x * v_last * v_prev, etc.
        #    It does NOT check v_x itself against `products`.
        #    But `products` contains all block products of `values`.
        #    `v_x` is a new number. Is it possible `v_x` is in `products`?
        #    `products` contains products of consecutive blocks from `values`.
        #    Since `values` are distinct and < x, and we are in strictly increasing order,
        #    the smallest product in `values` is at least 2.
        #    Can a product of a block from `values` equal `x`?
        #    Yes. E.g. values=[2,3], products={2,3,6}. Next x=5. 5 not in products.
        #    But if values=[2,3], products={2,3,6}. Next x=6.
        #    v_6 is (1,1,0...). Is (1,1,0...) in products?
        #    Products are 2(1,0), 3(0,1), 6(1,1).
        #    So v_6 corresponds to 6, which IS in products.
        #    So we MUST check v_x against products.
        
        # Let's fix the check.
        # We should check all new_vecs against products.
        
        collision = False
        for v in new_vecs:
            if v in products:
                collision = True
                break
        
        if not collision:
            values.append(x)
            products.update(new_vecs)
            
    return values

def _beam_search_impl(init_seq, init_len, K, deadline):
    """
    Beam search using exponent vectors.
    State: [seq_list, prods_set, sufs_list]
    sufs_list is a list of tuples (exponent vectors) representing the suffix products.
    Specifically, sufs_list[0] is the vector of the last element.
    sufs_list[1] is the vector of the last two elements.
    etc.
    """
    beam = [ [[], set(), []] ]
    best_seq = init_seq[:]
    best_len = init_len
    
    for x in range(2, 129):
        new_beam = []
        
        for state in beam:
            seq, prods, sufs = state
            
            # Pruning
            if len(seq) + (129 - x) <= best_len:
                continue
            
            # Option 1: Exclude x
            # We can reuse the state object? 
            # If we modify `seq` or `prods` later, it affects this reference.
            # We don't modify existing states in the loop, we create new ones.
            # So appending the reference is safe as long as we don't mutate it.
            # We don't mutate `state` in the "Include" branch either, we create `new_state`.
            # So we can append `state` directly.
            new_beam.append(state)
            
            # Option 2: Include x
            # Calculate new suffix vectors
            # new_sufs[0] = v_x
            # new_sufs[i] = new_sufs[i-1] + v_seq[-i]
            
            new_sufs = []
            # Start with v_x
            cur_vec = list(_EXP[x])
            new_sufs.append(tuple(cur_vec))
            
            valid = True
            # Check v_x against prods
            if tuple(cur_vec) in prods:
                valid = False
            
            if valid:
                # Add previous elements
                for a in reversed(seq):
                    ea = _EXP[a]
                    for k in range(_P_COUNT):
                        cur_vec[k] += ea[k]
                    t = tuple(cur_vec)
                    if t in prods:
                        valid = False
                        break
                    new_sufs.append(t)
            
            if valid:
                # No collision
                new_prods = prods.copy()
                new_prods.update(new_sufs)
                new_seq = seq + [x]
                new_state = [new_seq, new_prods, new_sufs]
                new_beam.append(new_state)
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq[:]
        
        if not new_beam:
            break
        
        if len(new_beam) > K:
            # Sort by length desc, then last element asc
            # To speed up, we can just sort.
            new_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
            beam = new_beam[:K]
        else:
            beam = new_beam
        
        if time.time() > deadline:
            break
    
    return best_seq, best_len

def _try_insert(seq, x):
    """Try to insert x into seq maintaining sorted order. Return new seq if valid, else None."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    insert_idx = lo
    
    if insert_idx < len(seq) and seq[insert_idx] == x:
        return None

    new_seq = seq[:insert_idx] + [x] + seq[insert_idx:]
    if _is_valid(new_seq):
        return new_seq
    return None

def _try_swap(seq, i, x):
    """Try to replace seq[i] with x. Maintain sorted order."""
    if i > 0 and x <= seq[i-1]:
        return None
    if i < len(seq) - 1 and x >= seq[i+1]:
        return None
    if x == seq[i]:
        return None
    
    new_seq = seq[:i] + [x] + seq[i+1:]
    if _is_valid(new_seq):
        return new_seq
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Fast Greedy Initialization
    best_seq = _greedy_construct()
    best_len = len(best_seq)

    # 2. Beam Search
    # Use a reasonable beam width. 
    # The state space grows, but pruning helps.
    K = 50
    beam_deadline = start_time + 40.0
    best_seq, best_len = _beam_search_impl(best_seq, best_len, K, beam_deadline)

    # 3. Local Search
    # Try to insert or swap to improve.
    local_deadline = deadline
    improved = True
    # Use a set for fast membership check
    seq_set = set(best_seq)
    
    while improved and time.time() < local_deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in seq_set]
        random.shuffle(candidates)
        
        # Try inserts
        for x in candidates:
            if time.time() > local_deadline:
                break
            new_seq = _try_insert(best_seq, x)
            if new_seq is not None:
                best_seq = new_seq
                best_len = len(best_seq)
                seq_set = set(best_seq)
                improved = True
                break
        
        if improved:
            continue

        # If no insert found, try swaps?
        # Swaps are more expensive to validate but might help.
        # However, swaps change the length? No, length stays same.
        # The goal is to maximize length.
        # If we can't insert, maybe a swap allows a future insert?
        # This is a hill-climbing move that doesn't change the objective (length) 
        # but might change the state to allow a better move later.
        # But since we restart the candidate loop, we can try swaps if we have time.
        
        # Let's stick to inserts first. If we have time, we can do a simple swap pass.
        # But swaps are O(N^2) to validate.
        # Let's just do inserts for now.
        
        # If we did a swap, we would need to update best_seq.
        # For now, if no insert, we break the loop to avoid infinite loop of no-ops.
        # Wait, if we don't improve, we should break.
        # The `improved` flag handles this.
        
        # However, we might want to try swaps even if length doesn't increase immediately,
        # if we believe it leads to a longer sequence later.
        # This is a deeper search.
        # Given time constraints, let's just do inserts.
        
        # If no improvement, break.
        if not improved:
            break

    return best_seq
