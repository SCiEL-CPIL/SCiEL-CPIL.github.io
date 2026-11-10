# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _prime_factorization(n):
    """Return prime factorization as a dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Precompute factorizations for all numbers in [2, 128]
_FACTS = {}
for _x in range(2, 129):
    _FACTS[_x] = _prime_factorization(_x)

def _vec_add(a, b):
    """Add two factorization dicts."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result

def _vec_sub(a, b):
    """Subtract factorization dict b from a."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) - v
        if result[k] == 0:
            del result[k]
    return result

def _compute_all_product_vecs(seq):
    """Compute all consecutive block product factorization vectors."""
    vecs = set()
    n = len(seq)
    for i in range(n):
        prod_vec = {}
        for j in range(i, n):
            prod_vec = _vec_add(prod_vec, _FACTS[seq[j]])
            vecs.add(frozenset(prod_vec.items()))
    return vecs

def _try_insert_incremental(seq, x, existing_vecs):
    """
    Try to insert x into seq at the correct position.
    Returns (new_seq, new_vecs) or (None, None).
    existing_vecs is a set of frozensets of current block products.
    """
    n = len(seq)
    # Find position where x would go (maintaining strictly increasing)
    pos = 0
    while pos < n and seq[pos] < x:
        pos += 1
    if pos < n and seq[pos] == x:
        return None, None
    
    # Check bounds (strictly increasing)
    if pos > 0 and seq[pos-1] >= x:
        return None, None
    if pos < n and seq[pos] <= x:
        return None, None
        
    # Calculate new block products involving x
    # Blocks are [i, j] in the new sequence.
    # Since x is at pos, the new blocks are those that include x.
    # These are [i, pos] for i in [0, pos] and [pos, j] for j in [pos, n-1].
    # Note: [pos, pos] is the singleton x.
    
    new_vecs_to_add = []
    
    # Blocks ending at pos (starting from i to pos)
    # In the new sequence, x is at index pos.
    # The elements before x are seq[0..pos-1].
    # Block [i, pos] corresponds to seq[i..pos-1] * x.
    # Let's compute product of seq[i..pos-1] for i from pos down to 0.
    # If i=pos, product is empty (identity), so block is just x.
    # If i=pos-1, block is seq[pos-1]*x.
    
    # Let's precompute suffix products of the left part and prefix products of the right part?
    # Actually, just iterating is fine since n is small (< 128).
    
    # Left side: blocks ending at x
    left_prod = {}
    for i in range(pos, -1, -1):
        if i < pos:
            left_prod = _vec_add(left_prod, _FACTS[seq[i]])
        # The block is seq[i..pos-1] combined with x
        block_vec = _vec_add(left_prod, _FACTS[x])
        key = frozenset(block_vec.items())
        if key in existing_vecs:
            return None, None
        new_vecs_to_add.append(key)
        
    # Right side: blocks starting at x
    # Blocks [pos, j] for j in [pos, n-1].
    # In new sequence, x is at pos. Elements after are seq[pos..n-1].
    # Block [pos, j] corresponds to x * seq[pos..j].
    # Let's compute prefix products of the right part.
    right_prod = {}
    for j in range(pos, n):
        right_prod = _vec_add(right_prod, _FACTS[seq[j]])
        # The block is x combined with seq[pos..j]
        block_vec = _vec_add(_FACTS[x], right_prod)
        key = frozenset(block_vec.items())
        if key in existing_vecs:
            return None, None
        # Avoid adding duplicate keys to new_vecs_to_add if they were added in left loop?
        # The singleton x is added in left loop (i=pos, left_prod={}, block_vec=x).
        # In right loop, j=pos, right_prod=seq[pos], block_vec=x*seq[pos].
        # Wait, the right loop starts at j=pos.
        # The block [pos, pos] in new sequence is just x.
        # My right loop logic:
        # j=pos: right_prod = _FACTS[seq[pos]]. block_vec = x * seq[pos]. This is block [pos, pos+1] in new sequence?
        # No.
        # Let's trace indices carefully.
        # New sequence: S' = S[0..pos-1] + [x] + S[pos..n-1].
        # Indices in S':
        # 0 .. pos-1 : S[0..pos-1]
        # pos : x
        # pos+1 .. n : S[pos..n-1]
        
        # Blocks involving x (index pos in S'):
        # 1. [i, pos] for i in 0..pos.
        #    Elements: S[i..pos-1] and x.
        #    My left loop: i from pos down to 0.
        #    If i=pos: left_prod = {}. block = x. Correct.
        #    If i=pos-1: left_prod = S[pos-1]. block = S[pos-1]*x. Correct.
        #    
        # 2. [pos, j] for j in pos..n.
        #    Elements: x and S[pos..j-1].
        #    My right loop: j from pos to n-1.
        #    If j=pos: right_prod = S[pos]. block = x * S[pos].
        #    This corresponds to elements at indices pos and pos+1 in S'.
        #    So this is block [pos, pos+1].
        #    What about block [pos, pos] (just x)? It was covered by left loop.
        #    So right loop should start from j=pos, representing extension to the right.
        #    But we must ensure we don't double count or miss.
        #    The set of blocks is:
        #    { [i, pos] | 0<=i<=pos } U { [pos, j] | pos<j<=n }
        #    The left loop covers [i, pos].
        #    The right loop as written covers [pos, j] where the element after x is S[j].
        #    So for j=pos, it covers x*S[pos] which is [pos, pos+1].
        #    So the right loop covers [pos, pos+1], [pos, pos+2], ..., [pos, n].
        #    This is correct.
        
        # However, we must check for collision with existing_vecs.
        # And we must add to new_vecs_to_add.
        # Note: We already added [pos, pos] (x) in left loop.
        # So we should not add it again.
        # The right loop starts with block [pos, pos+1].
        # So no duplicate with left loop's [pos, pos].
        
        if key in existing_vecs:
            return None, None
        new_vecs_to_add.append(key)
        
    # Also need to check if any new_vecs_to_add collide with EACH OTHER?
    # The problem states "Products of every nonempty consecutive block must be distinct".
    # The new blocks are distinct from each other?
    # [i1, pos] and [i2, pos] with i1 != i2 have different lengths, so different products?
    # Not necessarily, e.g., 2*3 = 6 and 6. But 2,3,6 are distinct numbers.
    # If seq is strictly increasing, can two different blocks have the same product?
    # Yes, e.g., 2, 3, 6 is not strictly increasing? 2<3<6.
    # Blocks: 2, 3, 6, 2*3=6, 3*6=18, 2*3*6=36.
    # Product 6 appears as singleton [6] and block [2,3].
    # So we MUST check if new blocks collide with each other.
    
    # Let's use a set to check uniqueness among new blocks.
    new_block_set = set(new_vecs_to_add)
    if len(new_block_set) != len(new_vecs_to_add):
        return None, None
        
    # If all checks pass, update existing_vecs
    new_existing_vecs = existing_vecs | new_block_set
    new_seq = seq[:pos] + [x] + seq[pos:]
    return new_seq, new_existing_vecs

def _local_search(seq, deadline, rng):
    """Apply local search with insertions and swaps to improve sequence length."""
    best = list(seq)
    # Precompute initial vecs
    best_vecs = _compute_all_product_vecs(best)
    improved = True
    max_passes = 50
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        n = len(best)
        available = [x for x in range(2, 129) if x not in best]
        
        # 1. Try to insert any available number
        # Sort available by number of prime factors (fewer = less collision risk)
        available.sort(key=lambda x: sum(_FACTS[x].values()))
        
        for x in available:
            if time.time() >= deadline:
                break
            new_seq, new_vecs = _try_insert_incremental(best, x, best_vecs)
            if new_seq is not None:
                best = new_seq
                best_vecs = new_vecs
                improved = True
                break
        
        if time.time() >= deadline:
            break
            
        # 2. If no insertion worked, try 1-out/1-in swaps
        if not improved:
            found_swap = False
            # Iterate over all positions to remove
            for i in range(len(best)):
                if time.time() >= deadline:
                    break
                removed = best[i]
                base = best[:i] + best[i+1:]
                # Compute vecs for base
                base_vecs = _compute_all_product_vecs(base)
                
                for x in available:
                    if time.time() >= deadline:
                        break
                    if x == removed:
                        continue
                    new_seq, new_vecs = _try_insert_incremental(base, x, base_vecs)
                    if new_seq is not None:
                        best = new_seq
                        best_vecs = new_vecs
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    rng = random.Random(42)
    
    # Greedy construction with heuristic
    values = []
    values_vecs = set()
    remaining = list(range(2, 129))
    remaining.sort(key=lambda x: sum(_FACTS[x].values()))
    
    for x in remaining:
        if time.time() >= deadline - 10.0:
            break
        new_seq, new_vecs = _try_insert_incremental(values, x, values_vecs)
        if new_seq is not None:
            values = new_seq
            values_vecs = new_vecs
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 2.0:
        improved = _local_search(values, deadline, rng)
        if len(improved) > len(values):
            values = improved
    
    # Random restarts
    if time.time() < deadline - 5.0:
        for restart in range(5):
            if time.time() >= deadline - 2.0:
                break
            rem = [x for x in range(2, 129) if x not in values]
            rng.shuffle(rem)
            rem.sort(key=lambda x: sum(_FACTS[x].values()))
            trial = list(values)
            trial_vecs = _compute_all_product_vecs(trial)
            for x in rem:
                if time.time() >= deadline - 2.0:
                    break
                new_seq, new_vecs = _try_insert_incremental(trial, x, trial_vecs)
                if new_seq is not None:
                    trial = new_seq
                    trial_vecs = new_vecs
            if len(trial) > len(values):
                values = trial
                if time.time() < deadline - 2.0:
                    improved = _local_search(values, deadline, rng)
                    if len(improved) > len(values):
                        values = improved
    
    return values
# EVOLVE-BLOCK-END
