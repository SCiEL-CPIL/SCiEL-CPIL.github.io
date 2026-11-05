"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def _compute_products(seq):
    """Compute all consecutive block products for a sequence."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def _is_valid(seq):
    """Check if sequence is strictly increasing in [2,128] with distinct block products."""
    if len(seq) == 0:
        return True
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i+1]:
            return False
    if seq[0] < 2 or seq[-1] > 128:
        return False
    return len(_compute_products(seq)) == len(seq) * (len(seq) + 1) // 2

def _greedy_with_lookahead(deadline, start_order=None):
    """Greedy construction with lookahead to avoid future collisions."""
    values = []
    products = set()
    
    # Determine candidate order
    if start_order is None:
        candidates = list(range(2, 129))
    else:
        candidates = start_order
    
    for x in candidates:
        if values and x <= values[-1]:
            continue
        if time.time() > deadline:
            break
        
        # Compute new suffix products
        new_tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            new_tails.append(prod)
        
        # Check for collisions
        if products.intersection(new_tails):
            continue
        
        # Lookahead: check if adding x makes it harder to extend
        # Simple heuristic: prefer numbers with fewer prime factors (more flexible)
        values.append(x)
        products.update(new_tails)
    
    return values

def _greedy_standard(deadline):
    """Standard greedy: try each number in order, add if no collision."""
    values = []
    products = set()
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        
        new_tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            new_tails.append(prod)
        
        if not products.intersection(new_tails):
            values.append(x)
            products.update(new_tails)
    
    return values

def _local_repair(seq, deadline):
    """Try to improve a sequence by local swaps and insertions."""
    if not seq or len(seq) == 0:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    # Try inserting missing numbers at various positions
    used = set(best)
    missing = [x for x in range(2, 129) if x not in used]
    
    # Try to insert each missing number at each position
    for x in missing:
        if time.time() > deadline:
            break
        for pos in range(len(best) + 1):
            # Check if inserting at pos maintains strictly increasing
            if pos > 0 and best[pos-1] >= x:
                continue
            if pos < len(best) and x >= best[pos]:
                continue
            
            candidate = best[:pos] + [x] + best[pos:]
            if _is_valid(candidate):
                if len(candidate) > best_len:
                    best = candidate
                    best_len = len(candidate)
    
    # Try swapping adjacent elements (if it maintains increasing order)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best) - 1):
            # Swap only if result is still increasing
            if best[i] + 1 < best[i+1]:  # Can we swap and keep increasing?
                candidate = best[:]
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                if _is_valid(candidate):
                    best = candidate
                    improved = True
    
    return best

def _beam_search(deadline, beam_width=5):
    """Beam search maintaining multiple candidate sequences."""
    # Each beam entry: (sequence, product_set)
    beams = [([], set())]
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        
        new_beams = []
        for seq, prods in beams:
            if seq and x <= seq[-1]:
                continue
            
            # Compute new suffix products
            new_tails = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                new_tails.append(prod)
            
            if not prods.intersection(new_tails):
                new_seq = seq + [x]
                new_prods = prods | set(new_tails)
                new_beams.append((new_seq, new_prods))
        
        if not new_beams:
            break
        
        # Keep top beam_width by length, then by "flexibility" (fewer products)
        new_beams.sort(key=lambda x: (-len(x[0]), len(x[1])))
        beams = new_beams[:beam_width]
    
    if not beams:
        return []
    
    # Return the longest sequence
    beams.sort(key=lambda x: -len(x[0]))
    return beams[0][0]

def _try_multiple_greedy_orders(deadline):
    """Try greedy with different candidate orderings."""
    results = []
    
    # Standard order
    results.append(_greedy_standard(deadline))
    
    # Try starting with primes
    primes = [x for x in range(2, 129) if all(x % i != 0 for i in range(2, int(x**0.5)+1))]
    non_primes = [x for x in range(2, 129) if x not in primes]
    order1 = primes + non_primes
    results.append(_greedy_with_lookahead(deadline, order1))
    
    # Try starting with numbers with fewer prime factors
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    sorted_by_pf = sorted(range(2, 129), key=lambda x: (prime_factor_count(x), x))
    results.append(_greedy_with_lookahead(deadline, sorted_by_pf))
    
    # Return the longest
    results.sort(key=lambda x: -len(x))
    return results[0]

def construct():
    """Construct the longest valid sequence."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best = []
    
    # Strategy 1: Standard greedy
    seq1 = _greedy_standard(deadline)
    if len(seq1) > len(best):
        best = seq1
    
    # Strategy 2: Multiple greedy orders
    seq2 = _try_multiple_greedy_orders(deadline)
    if len(seq2) > len(best):
        best = seq2
    
    # Strategy 3: Beam search
    if time.time() < deadline:
        seq3 = _beam_search(deadline, beam_width=3)
        if len(seq3) > len(best):
            best = seq3
    
    # Strategy 4: Local repair on the best found
    if time.time() < deadline and best:
        best = _local_repair(best, deadline)
    
    return best
# EVOLVE-BLOCK-END
