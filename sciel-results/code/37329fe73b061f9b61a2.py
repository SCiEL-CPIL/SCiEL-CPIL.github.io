"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid_system(triples):
    """Check that any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        ti = set(triples[i])
        for j in range(i + 1, n):
            tj = set(triples[j])
            if len(ti & tj) > 1:
                return False
    return True

def _independence_number(triples):
    """Compute the independence number: largest subset of points containing no triple."""
    points = set(range(16))
    best = 0
    # Try all subsets is too expensive (2^16 = 65536 is actually fine!)
    # 2^16 = 65536 subsets, each check is O(len(triples))
    # This is very fast.
    for mask in range(1 << 16):
        # Count bits
        if bin(mask).count('1') <= best:
            continue
        # Check if this subset contains any triple
        has_triple = False
        for t in triples:
            if (mask >> t[0] & 1) and (mask >> t[1] & 1) and (mask >> t[2] & 1):
                has_triple = True
                break
        if not has_triple:
            best = bin(mask).count('1')
    return best

def _can_add(triples, candidate):
    """Check if adding candidate triple maintains linearity (pairwise intersection <= 1)."""
    cand_set = set(candidate)
    for t in triples:
        t_set = set(t)
        if len(cand_set & t_set) > 1:
            return False
    return True

def _greedy_build():
    """Build a linear triple system using greedy randomized approach."""
    random.shuffle(list(range(16)))
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    
    system = []
    for t in all_triples:
        if _can_add(system, t):
            system.append(list(t))
    return system

def _local_search(triples, time_limit=10):
    """Improve the system by swapping triples to reduce independence number."""
    start = time.time()
    current_alpha = _independence_number(triples)
    best_alpha = current_alpha
    best_system = [list(t) for t in triples]
    
    # Compute which triples cover which points
    point_in_triples = [set() for _ in range(16)]
    for i, t in enumerate(triples):
        for p in t:
            point_in_triples[p].add(i)
    
    while time.time() - start < time_limit:
        # Find a triple to remove and a new triple to add
        improved = False
        random.shuffle(triples)
        
        for i in range(len(triples)):
            if time.time() - start > time_limit:
                return best_system, best_alpha
            
            # Try removing triple i
            remaining = triples[:i] + triples[i+1:]
            removed = triples[i]
            
            # Find points that are now "exposed" (in fewer triples)
            # Try to add a new triple that covers points in the independent set
            # For now, just try random additions
            for attempt in range(20):
                candidate = list(random.sample(range(16), 3))
                if _can_add(remaining, candidate):
                    new_system = remaining + [candidate]
                    new_alpha = _independence_number(new_system)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        triples = new_system
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [list(t) for t in triples]
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try a 2-opt move: remove two triples, add two new ones
            if len(triples) >= 2:
                i, j = random.sample(range(len(triples)), 2)
                i, j = min(i, j), max(i, j)
                remaining = triples[:i] + triples[i+1:j] + triples[j+1:]
                
                for attempt in range(30):
                    c1 = list(random.sample(range(16), 3))
                    c2 = list(random.sample(range(16), 3))
                    if _can_add(remaining, c1) and _can_add(remaining + [c1], c2):
                        new_system = remaining + [c1, c2]
                        new_alpha = _independence_number(new_system)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            triples = new_system
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [list(t) for t in triples]
                            break
    
    return best_system, best_alpha

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    random.seed(42)
    
    best_system = None
    best_alpha = 16  # worst case
    
    # Phase 1: Multiple greedy constructions
    for iteration in range(50):
        if time.time() > deadline:
            break
        system = _greedy_build()
        if len(system) < 5:
            continue
        alpha = _independence_number(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in system]
    
    # Phase 2: Local search on best found
    if best_system is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 2:
            improved_system, improved_alpha = _local_search(
                best_system, time_limit=min(remaining_time - 2, 30)
            )
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_system = improved_system
    
    # Phase 3: More greedy tries if time remains
    while time.time() < deadline - 5:
        system = _greedy_build()
        if len(system) < 5:
            continue
        alpha = _independence_number(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in system]
            # Do quick local search
            if time.time() < deadline - 10:
                improved_system, improved_alpha = _local_search(
                    best_system, time_limit=min(deadline - time.time() - 10, 15)
                )
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_system = improved_system
    
    if best_system is None:
        # Fallback: return the seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_system

# EVOLVE-BLOCK-END
