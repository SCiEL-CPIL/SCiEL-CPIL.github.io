import time
from fractions import Fraction
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _compute_L_fast(classes, max_check=200000):
    """Compute largest L such that 1..L all covered. Uses coverage count array."""
    if not classes:
        return 0
    
    max_check = min(max_check, 100000)
    covered = bytearray(max_check + 1)  # 0 or 1
    
    for p, r in classes:
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n in range(first, max_check + 1, p):
            covered[n] = 1
    
    L = 0
    for n in range(1, max_check + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def construct():
    deadline = time.time() + 145
    random.seed(42)
    
    primes = _primes_upto(1000)
    prime_set = set(primes)
    
    # Precompute 1/p as Fraction for all primes
    inv_p = {p: Fraction(1, p) for p in primes}
    
    # Strategy: Gap-focused simulated annealing
    # State: set of primes used and their residues
    # Objective: maximize L (first gap position)
    # Moves: change residue, add prime, remove prime
    
    # Initial solution: use primes from 13 upward
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_budget = sum(inv_p[p] for p, r in seed_classes)
    seed_L = _compute_L_fast(seed_classes, 100000)
    
    best_L = seed_L
    best_classes = seed_classes[:]
    best_budget = seed_budget
    
    # Simulated annealing parameters
    T_initial = 50.0
    T_min = 0.01
    cooling = 0.9995
    
    current_classes = seed_classes[:]
    current_budget = seed_budget
    current_L = seed_L
    
    T = T_initial
    
    # Track which primes are in use
    in_use = set(p for p, r in current_classes)
    
    # Coverage count array
    max_check = 100000
    covered_count = [0] * (max_check + 1)
    
    def rebuild_coverage(classes):
        nonlocal covered_count
        covered_count = [0] * (max_check + 1)
        for p, r in classes:
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, max_check + 1, p):
                covered_count[n] += 1
    
    def get_L_from_coverage():
        L = 0
        for n in range(1, max_check + 1):
            if covered_count[n] > 0:
                L = n
            else:
                break
        return L
    
    def find_first_gap(start=1, limit=None):
        """Find first uncovered position starting from start."""
        if limit is None:
            limit = max_check
        for n in range(start, min(limit + 1, max_check + 1)):
            if covered_count[n] == 0:
                return n
        return None
    
    def get_gap_near_L(window=200):
        """Get gaps near current L for targeted moves."""
        gaps = []
        search_start = max(1, current_L - 50)
        search_end = min(current_L + window, max_check)
        for n in range(search_start, search_end + 1):
            if covered_count[n] == 0:
                gaps.append(n)
        return gaps
    
    rebuild_coverage(current_classes)
    current_L = get_L_from_coverage()
    
    if current_L > best_L:
        best_L = current_L
        best_classes = current_classes[:]
        best_budget = current_budget
    
    # Simulated annealing loop with gap-focused moves
    step = 0
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5 and T > T_min:
        step += 1
        
        # Choose move type based on strategy
        move_type = random.random()
        
        if move_type < 0.5:
            # Change residue of an existing prime - focus on gaps
            if not current_classes:
                continue
            
            # Targeted: pick a prime that could help cover a gap
            gaps = get_gap_near_L(300)
            
            if gaps and random.random() < 0.7:
                # Pick a gap and try to cover it
                target_gap = random.choice(gaps[:20]) if len(gaps) > 20 else gaps[0]
                
                # Find primes that could cover this gap
                candidate_indices = []
                for idx, (p, r) in enumerate(current_classes):
                    # If we change r to target_gap % p, this prime covers the gap
                    candidate_indices.append(idx)
                
                if candidate_indices:
                    idx = random.choice(candidate_indices[:10]) if len(candidate_indices) > 10 else random.choice(candidate_indices)
                    p, old_r = current_classes[idx]
                    new_r = target_gap % p
                else:
                    idx = random.randint(0, len(current_classes) - 1)
                    p, old_r = current_classes[idx]
                    new_r = random.randint(0, p - 1)
            else:
                idx = random.randint(0, len(current_classes) - 1)
                p, old_r = current_classes[idx]
                new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply move
            current_classes[idx] = [p, new_r]
            
            # Update coverage
            if old_r == 0:
                first_old = p
            else:
                first_old = old_r
            if first_old < 1:
                first_old += p
            for n in range(first_old, max_check + 1, p):
                covered_count[n] -= 1
            
            if new_r == 0:
                first_new = p
            else:
                first_new = new_r
            if first_new < 1:
                first_new += p
            for n in range(first_new, max_check + 1, p):
                covered_count[n] += 1
            
            new_L = get_L_from_coverage()
            delta_L = new_L - current_L
            
            # Accept if better, or with probability exp(delta_L / T) if worse
            if delta_L >= 0 or random.random() < math.exp(max(delta_L, -1000) / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_budget = current_budget
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Revert
                current_classes[idx] = [p, old_r]
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                if first_new < 1:
                    first_new += p
                for n in range(first_new, max_check + 1, p):
                    covered_count[n] -= 1
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                if first_old < 1:
                    first_old += p
                for n in range(first_old, max_check + 1, p):
                    covered_count[n] += 1
                no_improve_steps += 1
                
        elif move_type < 0.8:
            # Add a new prime (if budget allows)
            available_primes = [p for p in primes if p not in in_use]
            if not available_primes:
                continue
            
            # Choose a prime that's affordable
            affordable = [p for p in available_primes if current_budget + inv_p[p] <= 1]
            if not affordable:
                continue
            
            # Prefer smaller primes (more coverage per unit budget)
            # But also consider what gaps need covering
            gaps = get_gap_near_L(500)
            
            if gaps:
                # Score each affordable prime by how well it covers gaps
                scored = []
                for p in affordable[:50]:  # Limit to top 50 for speed
                    # Count how many gaps this prime could cover
                    # A prime p with residue r covers n if n % p == r
                    # For a gap g, we'd set r = g % p
                    # But we want to maximize coverage of multiple gaps
                    gap_coverage = 0
                    for g in gaps[:30]:
                        # This prime can cover g with residue g % p
                        # Check if it also covers other nearby gaps
                        r_candidate = g % p
                        for g2 in gaps[:30]:
                            if g2 % p == r_candidate:
                                gap_coverage += 1
                    scored.append((gap_coverage, -p, p))  # Higher coverage, smaller prime preferred
                
                scored.sort(key=lambda x: (-x[0], x[1]))
                p = scored[0][2]
            else:
                # No gaps, just pick a small affordable prime
                p = min(affordable)
            
            # Choose residue: try to cover the first gap
            gap = find_first_gap(1, current_L + 1000)
            
            if gap is not None:
                r = gap % p
            else:
                r = random.randint(0, p - 1)
            
            # Apply move
            current_classes.append([p, r])
            in_use.add(p)
            current_budget += inv_p[p]
            
            # Add coverage
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, max_check + 1, p):
                covered_count[n] += 1
            
            new_L = get_L_from_coverage()
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(max(delta_L, -1000) / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_budget = current_budget
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Revert
                current_classes.pop()
                in_use.discard(p)
                current_budget -= inv_p[p]
                if r == 0:
                    first = p
                else:
                    first = r
                if first < 1:
                    first += p
                for n in range(first, max_check + 1, p):
                    covered_count[n] -= 1
                no_improve_steps += 1
                
        else:
            # Remove a prime (to free up budget or simplify)
            if len(current_classes) <= 15:
                continue
            
            idx = random.randint(0, len(current_classes) - 1)
            p, r = current_classes[idx]
            
            # Only remove if it doesn't create a gap in 1..current_L
            safe = True
            for n in range(1, min(current_L + 1, max_check + 1)):
                if covered_count[n] == 1:
                    if n % p == r:
                        safe = False
                        break
            
            if not safe:
                continue
            
            # Apply removal
            current_classes.pop(idx)
            in_use.discard(p)
            current_budget -= inv_p[p]
            
            # Remove coverage
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, max_check + 1, p):
                covered_count[n] -= 1
            
            new_L = get_L_from_coverage()
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(max(delta_L, -1000) / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_budget = current_budget
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Revert
                current_classes.insert(idx, [p, r])
                in_use.add(p)
                current_budget += inv_p[p]
                if r == 0:
                    first = p
                else:
                    first = r
                if first < 1:
                    first += p
                for n in range(first, max_check + 1, p):
                    covered_count[n] += 1
                no_improve_steps += 1
        
        T *= cooling
        
        # If stuck, try a restart with different seed
        if no_improve_steps > max_no_improve and time.time() < deadline - 20:
            no_improve_steps = 0
            # Try a greedy construction from scratch
            greedy_classes = _greedy_construct(primes, inv_p, max_check, deadline)
            if greedy_classes:
                gL = _compute_L_fast(greedy_classes, max_check)
                if gL > best_L:
                    best_L = gL
                    best_classes = greedy_classes[:]
                    best_budget = sum(inv_p[p] for p, r in greedy_classes)
                    # Reset current state to this
                    current_classes = greedy_classes[:]
                    current_budget = best_budget
                    current_L = gL
                    in_use = set(p for p, r in current_classes)
                    rebuild_coverage(current_classes)
    
    # Final local search on best solution
    if time.time() < deadline - 3:
        final_classes = best_classes[:]
        final_budget = best_budget
        final_L = best_L
        in_use_final = set(p for p, r in final_classes)
        
        rebuild_coverage(final_classes)
        final_L = get_L_from_coverage()
        
        # Try to improve each residue with gap-focused candidates
        for _iter in range(5):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(len(final_classes)):
                if time.time() > deadline - 2:
                    break
                p, old_r = final_classes[i]
                best_r = old_r
                best_L_i = final_L
                
                # Get gaps near current L
                gaps = get_gap_near_L(500)
                
                # Build candidate residues
                candidates = set()
                candidates.add(old_r)
                
                # Add residues that cover gaps
                for g in gaps[:20]:
                    candidates.add(g % p)
                
                # Add some random candidates
                for _ in range(min(p, 10)):
                    candidates.add(random.randint(0, p - 1))
                
                for r2 in candidates:
                    if r2 == old_r:
                        continue
                    
                    # Temporarily change
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old < 1:
                        first_old += p
                    for n in range(first_old, max_check + 1, p):
                        covered_count[n] -= 1
                    
                    if r2 == 0:
                        first_new = p
                    else:
                        first_new = r2
                    if first_new < 1:
                        first_new += p
                    for n in range(first_new, max_check + 1, p):
                        covered_count[n] += 1
                    
                    new_L = get_L_from_coverage()
                    
                    # Revert
                    if r2 == 0:
                        first_new = p
                    else:
                        first_new = r2
                    if first_new < 1:
                        first_new += p
                    for n in range(first_new, max_check + 1, p):
                        covered_count[n] -= 1
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old < 1:
                        first_old += p
                    for n in range(first_old, max_check + 1, p):
                        covered_count[n] += 1
                    
                    if new_L > best_L_i:
                        best_L_i = new_L
                        best_r = r2
                
                if best_r != old_r:
                    # Apply the best
                    p, old_r = final_classes[i]
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old < 1:
                        first_old += p
                    for n in range(first_old, max_check + 1, p):
                        covered_count[n] -= 1
                    if best_r == 0:
                        first_new = p
                    else:
                        first_new = best_r
                    if first_new < 1:
                        first_new += p
                    for n in range(first_new, max_check + 1, p):
                        covered_count[n] += 1
                    final_classes[i] = [p, best_r]
                    final_L = best_L_i
                    improved = True
            
            if not improved:
                break
        
        if final_L > best_L:
            best_L = final_L
            best_classes = final_classes[:]
            best_budget = final_budget
    
    return best_classes

def _greedy_construct(primes, inv_p, max_check, deadline):
    """Greedy: start empty, add primes to cover gaps."""
    classes = []
    budget = Fraction(0)
    covered = bytearray(max_check + 1)
    
    for p in primes:
        if time.time() > deadline - 10:
            break
        cost = inv_p[p]
        if budget + cost > 1:
            continue
        
        # Find the first uncovered position
        gap = None
        for n in range(1, max_check + 1):
            if not covered[n]:
                gap = n
                break
        
        if gap is None:
            break
        
        # Choose residue to cover the gap
        r = gap % p
        
        classes.append([p, r])
        budget += cost
        
        # Update coverage
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n in range(first, max_check + 1, p):
            covered[n] = 1
    
    return classes if classes else None

# EVOLVE-BLOCK-END
