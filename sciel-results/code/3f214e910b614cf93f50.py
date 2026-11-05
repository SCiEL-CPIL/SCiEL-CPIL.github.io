import numpy as np
import time
import random
from math import gcd

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def _uncovered_count(classes, base=BASE):
    """Compute number of uncovered residues."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def _compute_refcounts(classes):
    """Compute reference count for each residue."""
    refcount = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        refcount[r::m] += 1
    return refcount

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    divs = _divisors_of(BASE)
    n_classes = len(divs)
    
    # Helper to calculate delta in uncovered count for a move
    def calc_delta(refcount, idx, old_r, new_r):
        m = divs[idx]
        if old_r == new_r:
            return 0
        lost = np.sum(refcount[old_r::m] == 1)
        gained = np.sum(refcount[new_r::m] == 0)
        return lost - gained

    def apply_move(refcount, idx, old_r, new_r):
        m = divs[idx]
        refcount[old_r::m] -= 1
        refcount[new_r::m] += 1

    def revert_move(refcount, idx, old_r, new_r):
        m = divs[idx]
        refcount[new_r::m] -= 1
        refcount[old_r::m] += 1

    def greedy_init(order='desc'):
        """Fast greedy initialization with better residue selection."""
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        
        if order == 'desc':
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i], reverse=True)
        elif order == 'asc':
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i])
        elif order == 'rand':
            sorted_mods = list(range(n_classes))
            random.shuffle(sorted_mods)
        elif order == 'small_first':
            # Small moduli first - they cover fewer residues, so pick good ones early
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i])
        else:
            sorted_mods = list(range(n_classes))
            random.shuffle(sorted_mods)
        
        for i in sorted_mods:
            m = divs[i]
            best_r = 0
            best_count = -1
            
            # For small m, try all residues. For large m, use smarter sampling.
            if m <= 100:
                candidates = range(m)
            else:
                # Sample residues, biased towards those that cover more uncovered
                # Use stratified sampling across the modulus
                n_samples = min(m, 200)
                if n_samples == m:
                    candidates = range(m)
                else:
                    # Stratified sampling
                    step = m // n_samples
                    candidates = list(range(0, m, step))
                    # Add some random samples
                    extra = random.sample(range(m), min(50, m - len(candidates)))
                    candidates = list(set(candidates + extra))
            
            for r in candidates:
                cnt = np.sum(covered[r::m] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes

    def sa_search(initial_classes, max_time=30):
        """Simulated annealing with adaptive parameters."""
        current_classes = initial_classes[:]
        current_refcount = _compute_refcounts(current_classes)
        current_uncovered = int(np.sum(current_refcount == 0))
        
        best_classes = current_classes[:]
        best_uncovered = current_uncovered
        
        sa_start = time.time()
        sa_deadline = sa_start + max_time
        
        # Adaptive SA: start with higher temperature, cool down
        T_start = 5.0
        T_end = 0.05
        steps_per_temp = 200
        
        T = T_start
        step = 0
        while T > T_end and time.time() < sa_deadline:
            for _ in range(steps_per_temp):
                if time.time() >= sa_deadline:
                    break
                
                idx = random.randint(0, n_classes - 1)
                m = current_classes[idx][0]
                old_r = current_classes[idx][1]
                
                # Smart new residue selection: sample from promising regions
                if m <= 100:
                    new_r = random.randint(0, m - 1)
                else:
                    # Bias towards residues that currently cover uncovered areas
                    # Sample a few candidates and pick the best
                    candidates = [random.randint(0, m - 1) for _ in range(5)]
                    best_candidate = old_r
                    best_delta = float('inf')
                    for cr in candidates:
                        if cr == old_r:
                            continue
                        d = calc_delta(current_refcount, idx, old_r, cr)
                        if d < best_delta:
                            best_delta = d
                            best_candidate = cr
                    new_r = best_candidate
                
                if old_r == new_r:
                    continue
                    
                delta = calc_delta(current_refcount, idx, old_r, new_r)
                
                # Accept if better, or with probability exp(-delta/T)
                if delta < 0 or (delta < 20 and random.random() < np.exp(-delta / T)):
                    apply_move(current_refcount, idx, old_r, new_r)
                    current_classes[idx][1] = new_r
                    current_uncovered += delta
                    
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = current_classes[:]
                        if best_uncovered == 0:
                            return best_classes, 0
        return best_classes, best_uncovered

    def local_polish(classes, max_rounds=10, max_time=20):
        """Deterministic 1-opt local search for final polish."""
        current_classes = classes[:]
        current_refcount = _compute_refcounts(current_classes)
        current_uncovered = int(np.sum(current_refcount == 0))
        
        best_classes = current_classes[:]
        best_uncovered = current_uncovered
        
        polish_start = time.time()
        polish_deadline = polish_start + max_time
        
        improved = True
        rounds = 0
        while improved and time.time() < polish_deadline and rounds < max_rounds:
            improved = False
            rounds += 1
            current_uncovered = int(np.sum(current_refcount == 0))
            
            # Process moduli in order of increasing size (smaller ones have more options to improve)
            indices = sorted(range(n_classes), key=lambda i: divs[i])
            
            for idx in indices:
                if time.time() >= polish_deadline:
                    break
                m, r = current_classes[idx]
                
                best_r = r
                best_uncovered_local = current_uncovered
                
                # Check all residues for small m, sample for large m
                if m <= 80:
                    candidates = range(m)
                else:
                    # Smart sampling: try residues near current best and random ones
                    n_samples = min(m, 300)
                    if n_samples == m:
                        candidates = range(m)
                    else:
                        step = m // n_samples
                        candidates = list(range(0, m, step))
                        # Add neighbors of current best
                        for offset in [-2, -1, 1, 2]:
                            nr = (r + offset) % m
                            if nr not in candidates:
                                candidates.append(nr)
                        # Add random samples
                        extra = random.sample(range(m), min(50, m - len(candidates)))
                        candidates = list(set(candidates + extra))

                for r_new in candidates:
                    if r_new == r:
                        continue
                    delta = calc_delta(current_refcount, idx, r, r_new)
                    if delta < 0:
                        new_uncovered = current_uncovered + delta
                        if new_uncovered < best_uncovered_local:
                            best_uncovered_local = new_uncovered
                            best_r = r_new

                if best_r != r:
                    apply_move(current_refcount, idx, r, best_r)
                    current_classes[idx][1] = best_r
                    current_uncovered = best_uncovered_local
                    improved = True
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = current_classes[:]

        return best_classes, best_uncovered

    best_classes = None
    best_score = BASE + 1
    
    # Phase 1: Multiple greedy initializations
    orders = ['desc', 'asc', 'small_first', 'rand', 'rand', 'rand']
    for order in orders:
        if time.time() >= deadline:
            break
        classes = greedy_init(order=order)
        score = _uncovered_count(classes)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
            
    if best_classes is None:
        best_classes = greedy_init('desc')
        best_score = _uncovered_count(best_classes)

    # Phase 2: SA from best greedy solution
    if time.time() < deadline:
        sa_time = min(40, deadline - time.time() - 10)  # Reserve 10s for polish
        if sa_time > 5:
            sa_classes, sa_score = sa_search(best_classes, max_time=sa_time)
            if sa_score < best_score:
                best_score = sa_score
                best_classes = sa_classes[:]

    # Phase 3: Final polish
    if time.time() < deadline:
        polish_time = min(20, deadline - time.time() - 5)
        if polish_time > 3:
            polished_classes, polished_score = local_polish(best_classes, max_rounds=15, max_time=polish_time)
            if polished_score < best_score:
                best_score = polished_score
                best_classes = polished_classes[:]

    return best_classes
