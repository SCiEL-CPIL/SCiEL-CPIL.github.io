import time
import math
import random
import numpy as np
from itertools import product

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    # For very small n, numpy overhead might be comparable to loop, but n^2 is small.
    # For n > 50, n^2 is 2500, which is fine.
    # Let's always use numpy for consistency and speed on modern CPUs.
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    if n_sums == 0:
        return -1.0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def local_search(U, deadline, max_steps=1000, step_range=5):
    """
    Local search to improve the score of U.
    Tries to move each element by a small amount.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(U)
    
    current_score = fast_score(U)
    if current_score <= 0:
        return U, current_score

    n = len(U)
    
    # Fixed 0, others can move
    # We will try to adjust elements U[1:]
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle order of indices to avoid bias
        indices = list(range(1, n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = U[i]
            # Try moving in a window around old_val
            # The window size can depend on the magnitude of old_val
            # But for small sets, values are small.
            low = max(0, old_val - step_range)
            high = old_val + step_range
            
            # Ensure we don't collide with other elements
            # We check validity after constructing new_U
            
            best_new_val = old_val
            best_new_score = current_score
            
            # Sample some candidates in the range, plus exact neighbors
            candidates = set()
            candidates.add(old_val - 1)
            candidates.add(old_val + 1)
            if step_range > 1:
                # Random samples
                for _ in range(10):
                    candidates.add(random.randint(low, high))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val < 0:
                    continue
                
                new_U = U[:i] + [new_val] + U[i+1:]
                # Check for duplicates
                if len(set(new_U)) != len(new_U):
                    continue
                # Check sorted order is maintained (since we are replacing one element,
                # we need to ensure it doesn't violate ordering if we assume U is sorted.
                # Actually, we just sort it for the score function, but for local search
                # efficiency, we should keep it sorted.
                # If new_val is not in the right place, the set is still valid, just unsorted.
                # fast_score sorts it.
                
                s = fast_score(new_U)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val != old_val:
                U[i] = best_new_val
                # Re-sort U to maintain invariant for next iterations?
                # Not strictly necessary if fast_score sorts, but helps if we relied on order.
                # However, indices in the next iteration would be wrong if we swap values.
                # So we must re-sort U and then we lose the index mapping?
                # No, we are iterating over indices of the *current* U.
                # If we change U[i], the set changes.
                # Let's just sort U after an update.
                U = sorted(U)
                current_score = best_new_score
                improved = True
                # Restart the index loop? Or continue?
                # Continuing is fine, but indices might point to different values now.
                # To be safe, break and restart the outer step loop?
                # No, that's too slow.
                # Just continue. The indices are from the pre-update state.
                # This is a "steepest ascent" with stale indices.
                # It's a heuristic. It works well in practice.
        
        if not improved:
            # Try a larger step range?
            step_range += 2
            if step_range > 20:
                break
                
    return U, current_score

def generate_random_set(n, max_val, deadline):
    """Generate a random set of size n containing 0."""
    if n == 1:
        return [0]
    # Generate n-1 distinct non-zero integers
    # Use a power-law distribution to favor small numbers?
    # Or just uniform?
    # Small sets with small max are better.
    
    # Strategy: Pick 0, then pick n-1 numbers from 1..max_val
    # Ensure distinctness.
    if max_val < n:
        return list(range(n))
        
    S = {0}
    while len(S) < n:
        # Bias towards smaller numbers
        # x = int(random.random() ** 2 * max_val)
        x = random.randint(1, max_val)
        S.add(x)
    return sorted(S)

def generate_geometric_set(m, b, deadline):
    """Generate {0, 1, b, b^2, ...}"""
    U = [0, 1]
    cur = 1
    for i in range(m-1):
        cur = int(cur * b)
        if cur in U:
            # Collision, perturb
            cur += 1
            while cur in U:
                cur += 1
        U.append(cur)
    return sorted(set(U))

def construct():
    start = time.time()
    # Evaluator kills at 150s. Keep margin.
    deadline = start + 115
    
    best_set = [0, 1, 2, 4, 8]
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Explore small sets with local search
    # Sizes 3 to 10 seem promising.
    # Max value should be small to keep denominator small.
    # If n=5, max=8, log(17) ~ 2.8.
    # If n=10, max=500, log(1001) ~ 6.9.
    
    # We will run multiple restarts.
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Determine parameters for this restart
        # Mix of strategies
        strategy = random.random()
        
        if strategy < 0.4:
            # Random small set
            n = random.randint(3, 8)
            max_val = random.randint(10, 100)
            U = generate_random_set(n, max_val, deadline)
        elif strategy < 0.7:
            # Geometric-ish
            n = random.randint(3, 7)
            b = random.uniform(1.5, 3.0)
            U = generate_geometric_set(n, b, deadline)
            # Perturb
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 0: U[i] = 0
            U = sorted(set(U))
        else:
            # Specific known good structure: {0, 1, 2, 4, ...} with perturbations
            n = random.randint(4, 6)
            U = [0, 1, 2, 4]
            for i in range(4, n):
                U.append(U[-1] * 2 + random.randint(-1, 1))
            U = sorted(set(U))
            # Ensure 0 is present and unique
            if 0 not in U: U.insert(0, 0)
            U = sorted(set(U))
            if len(U) < 2:
                U = [0, 1]

        # Ensure valid set
        if len(U) < 2 or 0 not in U:
            continue
            
        # Run local search
        improved_U, score = local_search(U, deadline, max_steps=50)
        
        if score > best_score:
            best_score = score
            best_set = improved_U
            
    # Phase 2: Final polish on the best set found
    # Use a larger step range and more steps
    if time.time() < deadline - 10:
        improved_U, score = local_search(best_set, deadline, max_steps=500, step_range=10)
        if score > best_score:
            best_score = score
            best_set = improved_U
            
    # One more deep polish if time permits
    if time.time() < deadline - 2:
        improved_U, score = local_search(best_set, deadline, max_steps=100, step_range=20)
        if score > best_score:
            best_score = score
            best_set = improved_U

    return best_set

if __name__ == "__main__":
    # Quick self-test
    U = construct()
    print(f"Best Set: {U}")
    print(f"Score: {fast_score(U)}")
