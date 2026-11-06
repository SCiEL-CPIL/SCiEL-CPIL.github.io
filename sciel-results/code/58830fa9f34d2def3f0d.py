"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _validate(seq):
    """Check strictly increasing, in [1,256], and all consecutive block sums distinct."""
    if not seq:
        return False
    if len(set(seq)) != len(seq):
        return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
        return False
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(start_from=1):
    """
    Greedy: try to build a sequence by picking the smallest possible next value
    that doesn't create duplicate consecutive-block sums.
    """
    values = []
    sums = set()
    for x in range(start_from, 257):
        if not values:
            values.append(x)
            sums.add(x)
            continue
        # Compute new tail sums if we append x
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _local_search(seq, max_iter=50000, deadline=None):
    """
    Local search: try swapping, inserting, or removing elements to improve length
    while maintaining validity. Returns the best valid sequence found.
    """
    best = list(seq)
    if not _validate(best):
        return best
    
    n = len(best)
    # Precompute all consecutive block sums
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            t = 0
            for j in range(i, len(s)):
                t += s[j]
                sums.add(t)
        return sums
    
    best_sums = get_sums(best)
    best_len = n
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Try inserting a new element at a random position
        # or removing and re-inserting
        if random.random() < 0.3:
            # Try inserting a new value
            pos = random.randint(0, len(best))
            # Try to find a value that can be inserted
            for _attempt in range(20):
                if pos == 0:
                    lo, hi = 1, best[0] - 1 if best else 256
                elif pos == len(best):
                    lo, hi = best[-1] + 1, 256
                else:
                    lo, hi = best[pos-1] + 1, best[pos] - 1
                if lo > hi:
                    continue
                x = random.randint(lo, hi)
                new_seq = best[:pos] + [x] + best[pos:]
                if not _validate(new_seq):
                    continue
                if len(new_seq) > best_len:
                    best = new_seq
                    best_len = len(best)
                    best_sums = get_sums(best)
                    break
        else:
            # Try swapping two elements (keeping order)
            if len(best) >= 2:
                i, j = sorted(random.sample(range(len(best)), 2))
                new_seq = list(best)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                # Check if still strictly increasing
                if all(new_seq[k] < new_seq[k+1] for k in range(len(new_seq)-1)):
                    if _validate(new_seq):
                        # Same length, but might enable future improvements
                        # For now, just accept if valid
                        pass
            # Try removing one element and see if we can add a better one
            if len(best) > 1:
                rem = random.randint(0, len(best)-1)
                reduced = best[:rem] + best[rem+1:]
                if _validate(reduced):
                    # Try to add back a different element
                    for _attempt in range(10):
                        pos = random.randint(0, len(reduced))
                        if pos == 0:
                            lo, hi = 1, reduced[0] - 1 if reduced else 256
                        elif pos == len(reduced):
                            lo, hi = reduced[-1] + 1, 256
                        else:
                            lo, hi = reduced[pos-1] + 1, reduced[pos] - 1
                        if lo > hi:
                            continue
                        x = random.randint(lo, hi)
                        new_seq = reduced[:pos] + [x] + reduced[pos:]
                        if _validate(new_seq) and len(new_seq) > best_len:
                            best = new_seq
                            best_len = len(best)
                            best_sums = get_sums(best)
                            break
    
    return best

def construct():
    deadline = time.time() + 85  # 5 second safety margin
    
    # Start with greedy
    best = _greedy_build()
    best_len = len(best)
    
    # Try multiple random seeds for local search
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Perturb the greedy solution
        if random.random() < 0.5:
            # Start from greedy
            seq = list(best)
        else:
            # Start from a random valid sequence
            # Build a random increasing sequence
            seq = []
            used = set()
            for x in range(1, 257):
                if random.random() < 0.3 and x not in used:
                    # Check if we can add x
                    new_seq = seq + [x]
                    if _validate(new_seq):
                        seq = new_seq
                        used.add(x)
        
        improved = _local_search(seq, max_iter=2000, deadline=deadline)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Final polish: try to extend the best sequence
    if time.time() < deadline - 2:
        # Try adding elements from the end
        for x in range(best[-1] + 1, 257):
            new_seq = best + [x]
            if _validate(new_seq):
                best = new_seq
                break
        
        # Try inserting elements at various positions
        for _ in range(100):
            if time.time() > deadline:
                break
            pos = random.randint(0, len(best))
            if pos == 0:
                lo, hi = 1, best[0] - 1 if best else 256
            elif pos == len(best):
                lo, hi = best[-1] + 1, 256
            else:
                lo, hi = best[pos-1] + 1, best[pos] - 1
            if lo > hi:
                continue
            x = random.randint(lo, hi)
            new_seq = best[:pos] + [x] + best[pos:]
            if _validate(new_seq):
                best = new_seq
    
    return best
# EVOLVE-BLOCK-END
