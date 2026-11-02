"""Combined approach: sparse constructions + prime-based + local search optimization."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set_fast(U):
    """
    Optimized scoring. Uses pure Python sets for small N to avoid numpy overhead.
    For this problem, N is likely < 50, so pure Python is faster than np.unique.
    """
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # For small sets, direct set construction is faster than numpy broadcasting + unique
    # O(N^2) operations
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _generate_primes(max_val):
    """Generate all primes up to max_val."""
    if max_val < 2:
        return []
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    return [i for i in range(2, max_val + 1) if sieve[i]]

def _construct_golomb(n):
    """Construct a Golomb ruler of order n (all pairwise differences distinct)."""
    if n <= 1:
        return [0]
    
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96, 108],
    }
    if n in known:
        return known[n]
    
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _construct_sidon(n, max_val=None):
    """Construct a Sidon (B_2) sequence of length n."""
    if max_val is None:
        max_val = n * n * 6
    
    seq = [0]
    used_sums = {0}
    
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            for a in seq:
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                all_vals = seq + [pos]
                sums = set()
                for x in range(len(all_vals)):
                    for y in range(x, len(all_vals)):
                        s = all_vals[x] + all_vals[y]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    seq.append(pos)
                    break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _prime_gap_construction(n, primes=None, max_val=1000):
    """Construct U = {0, p1, p1+p2, p1+p2+p3, ...} using distinct primes."""
    if n <= 1:
        return [0]
    
    if primes is None:
        primes = _generate_primes(max_val)
    
    U = [0]
    current = 0
    for i in range(1, n):
        if i - 1 < len(primes):
            current += primes[i - 1]
            U.append(current)
        else:
            break
    
    return U

def _sparse_arithmetic(n, base, step, deadline=None):
    """Sparse arithmetic progression with prime-like steps."""
    U = [0]
    current = 0
    primes = _generate_primes(step * n * 2)
    
    for i in range(1, n):
        if i - 1 < len(primes):
            current += primes[i - 1] * base
            U.append(current)
        else:
            break
    
    return U

def _greedy_prime_build(n, max_val=500, deadline=None):
    """Greedy selection preferring primes to minimize sumset collisions."""
    U = [0]
    primes = _generate_primes(max_val)
    U_set = {0}
    
    for _ in range(n - 1):
        if deadline and time.time() > deadline:
            break
        
        best_score = _score_set_fast(U)
        best_candidate = None
        best_s = best_score
        
        # Try primes first
        for p in primes:
            if p in U_set:
                continue
            new_U = U + [p]
            s = _score_set_fast(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_candidate = p
            # Early exit if time is critical
            if deadline and time.time() > deadline - 0.5:
                break
        
        if best_candidate is None:
            # Try non-primes (limited range to save time)
            for v in range(1, min(max_val, 100) + 1):
                if v in U_set:
                    continue
                new_U = U + [v]
                s = _score_set_fast(new_U)
                if s > best_s + 1e-12:
                    best_s = s
                    best_candidate = v
                if deadline and time.time() > deadline - 0.5:
                    break
        
        if best_candidate is not None:
            U.append(best_candidate)
            U_set.add(best_candidate)
            U = sorted(U)
        else:
            break
    
    return U

def _simulated_annealing(U, deadline, max_size=20):
    """
    Simulated Annealing to optimize the set.
    """
    U = sorted(set(U))
    current_score = _score_set_fast(U)
    best_U = U[:]
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.999
    
    max_u = max(U) if U else 0
    upper_bound = max_u + 50
    
    while time.time() < deadline:
        if not U or 0 not in U:
            break
            
        # Generate a neighbor
        neighbor = U[:]
        op = random.choice(['modify', 'add', 'remove', 'swap'])
        
        if op == 'modify' and len(neighbor) > 1:
            idx = random.randint(1, len(neighbor) - 1)
            old_val = neighbor[idx]
            # Try to shift value
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val >= 0 and new_val != old_val and new_val not in neighbor:
                neighbor[idx] = new_val
                neighbor = sorted(neighbor)
                # Ensure 0 is still in set (it is, we didn't touch index 0 if we sorted? 
                # Wait, if we modify index 1, and it becomes 0, we have duplicate 0.
                # So we must check for duplicates.
                if len(set(neighbor)) != len(neighbor):
                    neighbor = U[:] # Revert
                else:
                    if neighbor[0] != 0: # If 0 was removed? No, we only modified idx >= 1.
                        pass
                    # If new_val < 0, revert.
                    if new_val < 0:
                        neighbor = U[:]
        
        elif op == 'add' and len(neighbor) < max_size:
            new_val = random.randint(1, upper_bound)
            if new_val not in neighbor:
                neighbor.append(new_val)
                neighbor = sorted(neighbor)
        
        elif op == 'remove' and len(neighbor) > 2:
            idx = random.randint(1, len(neighbor) - 1)
            neighbor.pop(idx)
        
        elif op == 'swap' and len(neighbor) >= 2:
            # Swap two non-zero elements
            idx1, idx2 = random.sample(range(1, len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            neighbor = sorted(neighbor)
        
        # Score neighbor
        if len(neighbor) != len(U):
             # Size changed, might be valid
             pass
             
        if len(set(neighbor)) != len(neighbor):
            continue
            
        if 0 not in neighbor:
            continue
            
        neighbor_score = _score_set_fast(neighbor)
        
        # Accept or reject
        if neighbor_score >= current_score:
            current_score = neighbor_score
            U = neighbor
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        else:
            # Boltzmann acceptance
            delta_s = current_score - neighbor_score
            if delta_s < 1e-9:
                accept = True
            else:
                accept_prob = math.exp(-delta_s / T)
                accept = random.random() < accept_prob
            
            if accept:
                current_score = neighbor_score
                U = neighbor
        
        T *= cooling
        
        # Occasionally expand search space
        if random.random() < 0.01:
            upper_bound += 20
            
    return best_U, best_score

def _local_search(U, max_iters=5000, deadline=None, max_size=40):
    """Local search with add/remove/modify operations."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 80
    
    scorer = _score_set_fast
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            if deadline and time.time() > deadline:
                break
            old_val = U_work[i]
            candidates = []
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            for _ in range(3):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = scorer(new_U_sorted)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                # Try primes first (limit count to save time)
                primes = _generate_primes(upper_bound)
                for new_val in primes[:50]: # Limit to 50 primes
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = scorer(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
                    if deadline and time.time() > deadline:
                        break
                
                if not improved and deadline and time.time() < deadline:
                    for new_val in range(1, upper_bound + 1):
                        if new_val in U:
                            continue
                        new_U = U + [new_val]
                        s = scorer(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_U = sorted(new_U)
                            improved = True
                            break
                        if deadline and time.time() > deadline:
                            break
        
        if not improved:
            # Try removing an element
            if len(U) > 2:
                for i in range(len(U)):
                    if deadline and time.time() > deadline:
                        break
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = scorer(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Random mutation
            if random.random() < 0.15:
                mutated = U[:]
                for _ in range(random.randint(1, 2)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                removable = [x for x in mutated if x != 0]
                for _ in range(random.randint(0, min(1, len(removable)))):
                    if removable:
                        idx = random.randint(0, len(removable) - 1)
                        val = removable.pop(idx)
                        mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = scorer(mutated)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 50:
            upper_bound += 30
    
    return sorted(set(best_U)), best_score

def _polish(U, deadline, max_iters=2000):
    """Final high-precision polish with tighter tolerance."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        for i in range(len(U_work)):
            if deadline and time.time() > deadline:
                break
            old_val = U_work[i]
            # Fine-grained search around current value
            for delta in range(-3, 4):
                nv = old_val + delta
                if nv < 0 or nv == old_val:
                    continue
                new_U = U_work[:]
                new_U[i] = nv
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = _score_set_fast(new_U_sorted)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try add/remove
            max_u = max(best_U) if best_U else 0
            if len(best_U) < 35:
                for new_val in range(1, max_u + 20):
                    if deadline and time.time() > deadline:
                        break
                    if new_val in best_U:
                        continue
                    new_U = best_U + [new_val]
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
            
            if not improved and len(best_U) > 3:
                for i in range(len(best_U)):
                    if deadline and time.time() > deadline:
                        break
                    if best_U[i] == 0:
                        continue
                    new_U = best_U[:i] + best_U[i+1:]
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved and it > 20:
            break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (evaluator kills at 150s, but we want to be safe)
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Generate diverse candidates from multiple construction strategies
    candidates = []
    
    # Golomb rulers
    for n in range(4, 18):
        if time.time() > deadline - 30:
            break
        gr = _construct_golomb(n)
        candidates.append(gr)
    
    # Sidon sequences
    for n in range(4, 15):
        if time.time() > deadline - 30:
            break
        ss = _construct_sidon(n, max_val=n * n * 8)
        if len(ss) >= 4:
            candidates.append(ss)
    
    # Prime gap constructions
    primes_500 = _generate_primes(500)
    for n in range(4, 15):
        if time.time() > deadline - 30:
            break
        pg = _prime_gap_construction(n, primes=primes_500)
        candidates.append(pg)
    
    # Sparse arithmetic with prime steps
    for base in [1, 2, 3]:
        for n in range(4, 12):
            if time.time() > deadline - 30:
                break
            sa = _sparse_arithmetic(n, base, 1)
            candidates.append(sa)
    
    # Greedy prime builds
    for n in [5, 8, 10]:
        if time.time() > deadline - 30:
            break
        gp = _greedy_prime_build(n, max_val=200, deadline=deadline - 30)
        if len(gp) >= 4:
            candidates.append(gp)
    
    # Random sets
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        n = random.randint(4, 15)
        max_val = random.randint(30, 200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))
    
    # Score all candidates and keep the best
    for U in candidates:
        if time.time() > deadline - 20:
            break
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 2: Simulated Annealing on the best candidate
    if time.time() < deadline - 10:
        improved_U, improved_score = _simulated_annealing(best_U, deadline=deadline - 5, max_size=25)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Local Search Polish
    if time.time() < deadline - 5:
        polished_U, polished_score = _local_search(best_U, max_iters=1000, deadline=deadline - 2, max_size=25)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U
    
    # Phase 4: Try Simulated Annealing on a few other good candidates
    if time.time() < deadline - 5:
        scored = [(s, U) for U in candidates for s in [_score_set_fast(U)]]
        scored.sort(key=lambda x: -x[0])
        
        for s, U in scored[:3]:
            if time.time() > deadline - 2:
                break
            if s < best_score - 0.05:
                continue
            improved_U, improved_score = _simulated_annealing(U, deadline=deadline - 1, max_size=25)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
