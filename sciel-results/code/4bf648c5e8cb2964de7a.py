"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(seq):
    """Check if seq has all distinct block products and is strictly increasing in [2,128]."""
    if not seq:
        return True
    if any(x < 2 or x > 128 for x in seq):
        return False
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] >= seq[j]:
                return False
    products = set()
    for i in range(len(seq)):
        p = 1
        for j in range(i, len(seq)):
            p *= seq[j]
            if p in products:
                return False
            products.add(p)
    return True

def _greedy_construct():
    """Original greedy: scan 2..128, append if no tail product collision."""
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

def _local_search(seq, deadline, rng):
    """Improve seq via 1-out/2-in moves (remove one element, insert two others) and 1-in moves."""
    best = list(seq)
    best_len = len(best)
    candidates = list(range(2, 129))
    # Remove elements already in best from candidates
    in_best = set(best)
    candidates = [c for c in candidates if c not in in_best]
    
    def try_insert(seq, pos, val):
        """Check if inserting val at pos keeps all block products distinct."""
        n = len(seq)
        # We need to check:
        # 1. New blocks ending at pos (val * suffix of seq[pos:])
        # 2. New blocks starting at pos (prefix of seq[:pos] * val)
        # 3. New blocks spanning pos (prefix of seq[:pos] * val * suffix of seq[pos:])
        # Actually, let's just recompute all products for the new sequence.
        # For efficiency, we can be smarter, but for small n it's fine.
        new_seq = seq[:pos] + [val] + seq[pos:]
        products = set()
        for i in range(len(new_seq)):
            p = 1
            for j in range(i, len(new_seq)):
                p *= new_seq[j]
                if p in products:
                    return None
                products.add(p)
        return new_seq

    def try_remove_insert(seq, rem_idx, ins_pos, val):
        """Remove seq[rem_idx], insert val at ins_pos (in the resulting list)."""
        # Build the list after removal
        rem_val = seq[rem_idx]
        base = seq[:rem_idx] + seq[rem_idx+1:]
        # Now insert val at ins_pos in base
        # ins_pos is index in the final list; we need to map
        # Actually, let's just build the candidate and check
        new_seq = base[:ins_pos] + [val] + base[ins_pos:]
        # Check strictly increasing
        for i in range(len(new_seq)-1):
            if new_seq[i] >= new_seq[i+1]:
                return None
        # Check distinct block products
        products = set()
        for i in range(len(new_seq)):
            p = 1
            for j in range(i, len(new_seq)):
                p *= new_seq[j]
                if p in products:
                    return None
                products.add(p)
        return new_seq

    while time.time() < deadline:
        improved = False
        # Try 1-in moves: insert a candidate somewhere
        for c in candidates:
            if time.time() >= deadline:
                break
            for pos in range(len(best) + 1):
                # Check strictly increasing
                if pos > 0 and best[pos-1] >= c:
                    continue
                if pos < len(best) and c >= best[pos]:
                    continue
                new_seq = try_insert(best, pos, c)
                if new_seq is not None and len(new_seq) > best_len:
                    best = new_seq
                    best_len = len(best)
                    improved = True
                    # Update candidates
                    in_best = set(best)
                    candidates = [c2 for c2 in candidates if c2 not in in_best]
                    break
            if improved:
                break
        if improved:
            continue
        # Try 1-out/2-in moves: remove one, insert two
        for rem_idx in range(len(best)):
            if time.time() >= deadline:
                break
            rem_val = best[rem_idx]
            base = best[:rem_idx] + best[rem_idx+1:]
            # Now try inserting two candidates
            # To limit search, try pairs of candidates
            # This is expensive, so we limit
            for i_c in range(len(candidates)):
                if time.time() >= deadline:
                    break
                c1 = candidates[i_c]
                for j_c in range(i_c + 1, len(candidates)):
                    c2 = candidates[j_c]
                    # Try inserting c1 and c2 into base at various positions
                    # This is O(n^2 * n^2) which is too much. Let's be smarter.
                    # Try: insert c1 at pos1, then c2 at pos2 in the resulting list
                    # Limit positions
                    for pos1 in range(len(base) + 1):
                        if pos1 > 0 and base[pos1-1] >= c1:
                            continue
                        if pos1 < len(base) and c1 >= base[pos1]:
                            continue
                        mid = base[:pos1] + [c1] + base[pos1:]
                        for pos2 in range(len(mid) + 1):
                            if pos2 > 0 and mid[pos2-1] >= c2:
                                continue
                            if pos2 < len(mid) and c2 >= mid[pos2]:
                                continue
                            new_seq = mid[:pos2] + [c2] + mid[pos2:]
                            # Check distinct block products
                            products = set()
                            valid = True
                            for i in range(len(new_seq)):
                                p = 1
                                for j in range(i, len(new_seq)):
                                    p *= new_seq[j]
                                    if p in products:
                                        valid = False
                                        break
                                    products.add(p)
                                if not valid:
                                    break
                            if valid and len(new_seq) > best_len:
                                best = new_seq
                                best_len = len(best)
                                improved = True
                                in_best = set(best)
                                candidates = [c3 for c3 in candidates if c3 not in in_best]
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # No improvement found, break to avoid infinite loop
            # But maybe shuffle candidates and try again?
            # For now, break
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    # Start with greedy
    best = _greedy_construct()
    best_len = len(best)
    
    # Try local search to improve
    if time.time() < deadline - 5:
        rng = random.Random(42)
        improved = _local_search(best, deadline - 2, rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # If we still have time, try more aggressive search
    if time.time() < deadline - 10:
        # Try different greedy orders
        for perm_seed in range(1, 50):
            if time.time() >= deadline - 5:
                break
            rng2 = random.Random(perm_seed)
            # Create a permutation of 2..128
            nums = list(range(2, 129))
            rng2.shuffle(nums)
            # Greedy with this order
            values, products = [], set()
            for x in nums:
                if time.time() >= deadline - 2:
                    break
                tails, prod = [x], x
                for a in reversed(values):
                    prod *= a
                    tails.append(prod)
                if not products.intersection(tails):
                    values.append(x)
                    products.update(tails)
            if len(values) > best_len:
                best = values
                best_len = len(values)
    
    # Final local search if time permits
    if time.time() < deadline - 5:
        rng3 = random.Random(123)
        improved = _local_search(best, deadline - 1, rng3)
        if len(improved) > best_len:
            best = improved
    
    return best
# EVOLVE-BLOCK-END
