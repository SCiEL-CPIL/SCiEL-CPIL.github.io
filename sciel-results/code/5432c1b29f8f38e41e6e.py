import time
import random
import itertools

def _all_block_products(seq):
    """Return the set of all nonempty consecutive block products."""
    n = len(seq)
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check if seq is strictly increasing in [2,128] and all block products distinct."""
    if len(seq) == 0:
        return True
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i+1]:
            return False
    if seq[0] < 2 or seq[-1] > 128:
        return False
    prods = _all_block_products(seq)
    return len(prods) == len(seq) * (len(seq) + 1) // 2

def _greedy_init():
    """Original greedy: iterate 2..128, add if no tail product collision."""
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _try_insert(seq, x):
    """Try to insert x into seq at every valid position. Return best (longest) result or None."""
    n = len(seq)
    best = None
    for i in range(n + 1):
        # Check strictly increasing
        if i > 0 and seq[i-1] >= x:
            continue
        if i < n and x >= seq[i]:
            continue
        new_seq = seq[:i] + [x] + seq[i:]
        if _is_valid(new_seq):
            if best is None or len(new_seq) > len(best):
                best = new_seq
    return best

def _try_remove(seq, idx):
    """Remove element at idx. Always valid if seq was valid."""
    return seq[:idx] + seq[idx+1:]

def _try_swap(seq, idx, new_val):
    """Replace seq[idx] with new_val. Check validity."""
    if new_val < 2 or new_val > 128:
        return None
    if idx > 0 and seq[idx-1] >= new_val:
        return None
    if idx < len(seq) - 1 and new_val >= seq[idx+1]:
        return None
    new_seq = seq[:idx] + [new_val] + seq[idx+1:]
    if _is_valid(new_seq):
        return new_seq
    return None

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Initialize with greedy
    best_seq = _greedy_init()
    best_len = len(best_seq)
    
    random.seed(42)
    
    # Local search: try to improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best_seq)
        current_set = set(best_seq)
        
        # Try insertions
        for x in range(2, 129):
            if x in current_set:
                continue
            result = _try_insert(best_seq, x)
            if result is not None and len(result) > best_len:
                best_seq = result
                best_len = len(result)
                improved = True
                break
        if improved:
            continue
        
        # Try swaps
        for idx in range(len(best_seq)):
            old_val = best_seq[idx]
            for new_val in range(2, 129):
                if new_val == old_val or new_val in current_set:
                    continue
                result = _try_swap(best_seq, idx, new_val)
                if result is not None:
                    # After swap, try to insert more
                    new_set = set(result)
                    for x in range(2, 129):
                        if x in new_set:
                            continue
                        ins_result = _try_insert(result, x)
                        if ins_result is not None and len(ins_result) > best_len:
                            best_seq = ins_result
                            best_len = len(ins_result)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        if improved:
            continue
        
        # Try removals followed by insertions (to escape local optima)
        for idx in range(len(best_seq)):
            removed = _try_remove(best_seq, idx)
            removed_set = set(removed)
            for x in range(2, 129):
                if x in removed_set:
                    continue
                ins_result = _try_insert(removed, x)
                if ins_result is not None and len(ins_result) > best_len:
                    best_seq = ins_result
                    best_len = len(ins_result)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Random restart with perturbation
        if time.time() < deadline:
            # Perturb: randomly remove 1-3 elements, then try to fill back in
            perturbed = list(best_seq)
            k = random.randint(1, min(3, len(perturbed)))
            for _ in range(k):
                if perturbed:
                    perturbed.pop(random.randint(0, len(perturbed)-1))
            # Try to greedily fill back
            current_set = set(perturbed)
            for x in range(2, 129):
                if x in current_set:
                    continue
                result = _try_insert(perturbed, x)
                if result is not None:
                    perturbed = result
                    current_set = set(perturbed)
            if len(perturbed) > best_len:
                best_seq = perturbed
                best_len = len(perturbed)
                improved = True
    
    return best_seq
