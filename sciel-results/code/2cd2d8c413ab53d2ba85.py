import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True

def _local_search(A, max_iter=3000, deadline=None):
    """Local search with structured perturbations."""
    current_A = list(A)
    current_score = _score(current_A)
    best_A = list(current_A)
    best_score = current_score
    rng = random.Random(99)
    n = len(current_A)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        k = rng.randrange(n)
        old_val = current_A[k]
        
        candidates = []
        for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
            nv = old_val + delta
            if 1 <= nv <= 10000:
                candidates.append(nv)
        if old_val > 2:
            candidates.append(max(1, old_val // 2))
            candidates.append(min(10000, old_val * 2))
        candidates.append(rng.randint(1, 10000))
        candidates.append(rng.randint(1, max(2, old_val + 10)))
        
        rng.shuffle(candidates)
        
        for new_val in candidates:
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A and new_val != old_val:
                continue
            
            current_A[k] = new_val
            new_score = _score(current_A)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(current_A)
                break
            else:
                current_A[k] = old_val
        
        if not improved:
            k = rng.randrange(n)
            old_val = current_A[k]
            new_val = rng.randint(1, 10000)
            if new_val in current_A and new_val != old_val:
                continue
            current_A[k] = new_val
            new_score = _score(current_A)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(current_A)
            else:
                current_A[k] = old_val
    
    return best_A, best_score

def _sa_search(A, start_temp=10.0, end_temp=0.1, steps=50000, seed=None, deadline=None):
    """Simulated Annealing search with deadline guard."""
    if seed is not None:
        random.seed(seed)
    
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    
    n = len(current_A)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        T = start_temp * (end_temp / start_temp) ** (step / steps)
        
        k = random.randint(0, n - 1)
        old_val = current_A[k]
        
        if random.random() < 0.7:
            delta = random.randint(-20, 20)
            new_val = old_val + delta
        elif random.random() < 0.5:
            new_val = random.randint(1, max(1, old_val // 2)) if old_val > 10 else random.randint(1, 100)
        else:
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        if new_val in current_A and new_val != old_val:
            continue
            
        current_A[k] = new_val
        new_score = _score(current_A)
        
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A[:]
        else:
            current_A[k] = old_val
            
    return best_A, best_score

def _generate_two_gp_candidates(deadline):
    """
    Generate candidates as unions of two disjoint geometric progressions.
    This maximizes multiplicative collisions via unique prime factorization.
    """
    candidates = []
    rng = random.Random(42)
    
    # Common small bases for geometric progressions
    bases = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Try pairs of bases
    for b1 in bases:
        for b2 in bases:
            if b1 >= b2:
                continue  # Only consider b1 < b2 to avoid duplicates
            if time.time() > deadline:
                return candidates
            
            # Try different sizes for each GP
            for n1 in range(3, 18):
                n2 = 20 - n1
                if n2 < 3:
                    continue
                
                # Try different starting offsets
                for start1 in [1, 2, 3, 5, 10]:
                    for start2 in [1, 2, 3, 5, 10, 50, 100]:
                        if time.time() > deadline:
                            return candidates
                        
                        A = set()
                        
                        # First GP: start1 * b1^i for i in range(n1)
                        val = float(start1)
                        for _ in range(n1):
                            iv = int(round(val))
                            if iv < 1:
                                iv = 1
                            if iv > 10000:
                                break
                            A.add(iv)
                            val *= b1
                        
                        # Second GP: start2 * b2^j for j in range(n2)
                        val = float(start2)
                        for _ in range(n2):
                            iv = int(round(val))
                            if iv < 1:
                                iv = 1
                            if iv > 10000:
                                break
                            A.add(iv)
                            val *= b2
                        
                        if len(A) == 20:
                            A_list = sorted(list(A))
                            if _valid(A_list):
                                candidates.append(A_list)
    
    # Also try mixed: one GP + one AP
    for b in [2, 3, 4, 5]:
        for n_gp in range(5, 16):
            n_ap = 20 - n_gp
            if n_ap < 5:
                continue
            
            for start_gp in [1, 2, 3, 5, 10]:
                for start_ap in [1, 2, 3, 5, 10]:
                    for d in [1, 2, 3, 5, 10]:
                        if time.time() > deadline:
                            return candidates
                        
                        A = set()
                        
                        # GP
                        val = float(start_gp)
                        for _ in range(n_gp):
                            iv = int(round(val))
                            if iv < 1:
                                iv = 1
                            if iv > 10000:
                                break
                            A.add(iv)
                            val *= b
                        
                        # AP
                        for i in range(n_ap):
                            iv = start_ap + i * d
                            if iv > 10000:
                                break
                            A.add(iv)
                        
                        if len(A) == 20:
                            A_list = sorted(list(A))
                            if _valid(A_list):
                                candidates.append(A_list)
    
    # Random structured: pick from log-spaced grid with two clusters
    for _ in range(30):
        if time.time() > deadline:
            break
        # Two clusters: small and large
        n_small = rng.randint(8, 12)
        n_large = 20 - n_small
        
        small = sorted(rng.sample(range(1, 50), n_small))
        large = sorted(rng.sample(range(500, 10001), n_large))
        A = sorted(small + large)
        if len(set(A)) == 20 and _valid(A):
            candidates.append(A)
    
    return candidates

def _generate_standard_candidates(deadline):
    """Generate standard structured candidates."""
    candidates = []
    rng = random.Random(123)
    
    # Consecutive small integers
    candidates.append(list(range(1, 21)))
    
    # Geometric progressions
    for base in [2, 3, 4, 5]:
        A = [base ** i for i in range(20)]
        if max(A) <= 10000 and len(set(A)) == 20:
            candidates.append(A)
    
    # Log-spaced
    for base_low in [1.2, 1.5, 2.0, 2.5]:
        A = []
        for i in range(20):
            t = i / 19.0
            val = int(base_low ** (t * 19))
            if val < 1:
                val = 1
            if val > 10000:
                val = 10000
            if val not in A:
                A.append(val)
            else:
                val += 1
                if val <= 10000 and val not in A:
                    A.append(val)
        if len(A) == 20 and _valid(A):
            candidates.append(A)
    
    # Random samples
    for _ in range(20):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 10001), 20))
        candidates.append(A)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    # 1. Generate two-GP structured candidates (primary strategy)
    candidates = _generate_two_gp_candidates(deadline)
    
    # 2. Generate standard candidates
    std_candidates = _generate_standard_candidates(deadline)
    candidates.extend(std_candidates)
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _score(best_A)
    
    # Local search on best
    if time.time() < deadline - 5:
        A_ls, s_ls = _local_search(best_A, max_iter=2000, deadline=deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_A = A_ls
    
    # Try local search on top diverse candidates
    scored_cands = []
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _score(cand)
        scored_cands.append((s, cand))
    scored_cands.sort(key=lambda x: x[0])
    
    # Deduplicate
    seen = set()
    unique_scored = []
    for s_c, cand in scored_cands:
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_scored.append((s_c, cand))
    
    # Run local search on top 10 unique candidates
    for s_c, cand in unique_scored[:10]:
        if time.time() > deadline - 15:
            break
        A_ls, s_ls = _local_search(cand, max_iter=1000, deadline=deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_A = A_ls
    
    # Simulated annealing from best
    if time.time() < deadline - 10:
        A_sa, s_sa = _sa_search(best_A, start_temp=5.0, end_temp=0.01, steps=30000, seed=0, deadline=deadline)
        if s_sa < best_score:
            best_score = s_sa
            best_A = A_sa
    
    # Final local polish
    if time.time() < deadline - 2:
        A_polish, s_polish = _local_search(best_A, max_iter=3000, deadline=deadline)
        if s_polish < best_score:
            best_score = s_polish
            best_A = A_polish
    
    return best_A
