import time
import random
import itertools
from math import log
import numpy as np

# Try to import pysat for SAT-based approach
try:
    from pysat.solvers import Solver
    from pysat.formula import CNF
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False

def score_set(U):
    """Score a set U using efficient NumPy operations."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums and differences
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_set(U):
    """Faster scoring using sets instead of numpy for small U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use Python sets for small U
    sums_set = set()
    diffs_set = set()
    for a in U:
        for b in U:
            sums_set.add(a + b)
            diffs_set.add(a - b)
    
    ratio = len(diffs_set) / len(sums_set)
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def sat_based_search(deadline, max_size=8, max_val=50, target_ratio=1.5):
    """
    Use SAT solver to find sets where |U-U|/|U+U| >= target_ratio.
    Returns best set found or None.
    """
    if not HAS_PYSAT:
        return None
    
    best_set = None
    best_score = -1
    
    # We'll try to find sets of various sizes
    for n in range(3, max_size + 1):
        if time.time() > deadline - 15:
            break
            
        # SAT variables: x[i] = 1 if element i is in U, for i = 0..max_val
        # x[0] must be 1 (0 is always in U)
        
        # We need to encode:
        # 1. Exactly n elements in U (including 0)
        # 2. |U-U| / |U+U| >= target_ratio
        # 
        # This is complex to encode directly. Instead, let's use a simpler approach:
        # Generate candidate sets using randomization guided by SAT constraints.
        
        # Alternative: Use SAT to enforce structural properties that lead to good ratio
        # For now, let's use a hybrid approach: SAT for structure, scoring for refinement
        
        # Actually, let's try a different structural approach:
        # Use the fact that we want sparse sumset and dense difference set.
        # This suggests using arithmetic progressions or structured sets.
        
        pass
    
    return best_set, best_score

def structured_search(deadline, max_size=10, max_val=100):
    """
    Search for structured sets that naturally have good sum/difference ratio.
    Uses arithmetic progressions, geometric progressions, and combinations.
    """
    best_set = None
    best_score = -1
    
    # Try arithmetic progressions
    for a in range(1, 20):  # start
        for d in range(1, 30):  # difference
            for n in range(3, max_size + 1):
                if time.time() > deadline - 10:
                    break
                U = [0]
                for i in range(1, n):
                    val = a + i * d
                    if val > max_val:
                        break
                    if val not in U:
                        U.append(val)
                if len(U) >= 3:
                    U = sorted(set(U))
                    if 0 not in U:
                        U = sorted(set(U + [0]))
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
    
    # Try geometric progressions
    for a in range(1, 10):
        for r in range(2, 6):
            for n in range(3, max_size + 1):
                if time.time() > deadline - 10:
                    break
                U = [0]
                val = a
                for i in range(1, n):
                    if val > max_val:
                        break
                    if val not in U:
                        U.append(val)
                    val *= r
                if len(U) >= 3:
                    U = sorted(set(U))
                    if 0 not in U:
                        U = sorted(set(U + [0]))
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
    
    # Try combinations of small structured sets
    base_sets = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 10],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10, 21],
    ]
    
    for base in base_sets:
        if time.time() > deadline - 10:
            break
        # Extend base sets with additional elements
        for extra in range(1, 50):
            if extra in base:
                continue
            U = sorted(set(base + [extra]))
            if len(U) < 3:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Try random structured combinations
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        n = random.randint(3, 8)
        # Mix of arithmetic and random
        if random.random() < 0.5:
            # Arithmetic-like
            a = random.randint(1, 10)
            d = random.randint(2, 15)
            U = [0]
            for i in range(1, n):
                val = a + i * d + random.randint(-2, 2)
                if val > 0 and val not in U:
                    U.append(val)
        else:
            # Random with structure
            U = [0]
            for i in range(1, n):
                val = random.randint(1, 50)
                if val not in U:
                    U.append(val)
        
        if len(U) >= 3:
            U = sorted(set(U))
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def local_search(start_set, deadline, max_iter=5000):
    """Local search with simulated annealing."""
    current = sorted(set(start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    T = 0.05
    cooling = 0.99
    
    while time.time() < deadline - 2 and max_iter > 0:
        max_iter -= 1
        if len(current) <= 1:
            break
            
        op = random.random()
        
        if op < 0.7:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            if random.random() < 0.5:
                delta = random.randint(-3, 3)
                new_val = old_val + delta
            else:
                new_val = random.randint(1, current[-1] + 10)
                
            if new_val <= 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != len(current):
                continue
                
        elif op < 0.85:
            if len(current) >= 15:
                continue
            new_val = random.randint(1, current[-1] + 20)
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            if len(current) <= 4:
                continue
            i = random.randint(1, len(current) - 1)
            new_set = current[:i] + current[i+1:]
        
        new_score = score_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set[:]
            
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current = new_set
                current_score = new_score
                
        T *= cooling
        
    return best_set, best_score

def local_search(start_set, deadline, max_iter=5000):
    current = sorted(set(start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Simple simulated annealing / hill climbing
    T = 0.05
    cooling = 0.99
    
    while time.time() < deadline - 2 and max_iter > 0:
        max_iter -= 1
        if len(current) <= 1:
            break
            
        # Randomly choose an operation: move, add, remove
        op = random.random()
        
        if op < 0.7:
            # Move an element
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            if random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-3, 3)
                new_val = old_val + delta
            else:
                # Jump
                new_val = random.randint(1, current[-1] + 10)
                
            if new_val <= 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != len(current):
                # Collision, skip or handle? Usually skip if we want fixed size
                continue
                
        elif op < 0.85:
            # Add an element
            if len(current) >= 15: # Cap size to keep search fast
                continue
            new_val = random.randint(1, current[-1] + 20)
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            # Remove an element (not 0)
            if len(current) <= 4:
                continue
            i = random.randint(1, len(current) - 1)
            new_set = current[:i] + current[i+1:]
            
        new_score = score_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set[:]
            
        # Accept if better or with probability
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current = new_set
                current_score = new_score
                
        T *= cooling
        
    return best_set, best_score

def construct():
    deadline = time.time() + 105  # Safety margin for 120s limit
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Phase 1: Structured search for good base sets
    if time.time() < deadline - 30:
        structured_set, structured_score = structured_search(deadline - 30, max_size=10, max_val=100)
        if structured_set and structured_score > best_score:
            best_score = structured_score
            best_set = structured_set[:]
    
    # Phase 2: Local search from multiple seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 4],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
    ]
    
    # Add the structured result as a seed if we found one
    if best_set and best_set != [0, 1, 4]:
        seeds.insert(0, best_set[:])
    
    # Add some random seeds
    random.seed(42)
    for _ in range(10):
        n = random.randint(4, 8)
        max_val = random.randint(20, 80)
        U = [0]
        other_elements = random.sample(range(1, max_val), n - 1)
        U.extend(other_elements)
        seeds.append(sorted(U))
        
    # Local search from each seed
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        local_best, local_score = local_search(seed, deadline, max_iter=3000)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
    
    return best_set
