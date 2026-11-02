"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    maxU = max(U)
    if maxU == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for i in range(n):
        for j in range(i, n):  # symmetric, but we need all pairs
            sumset.add(U[i] + U[j])
    # Actually need all pairs including order? No, U+U = {a+b : a,b in U}
    # Since addition is commutative, we can do i<=j but that's fine for the set
    # Wait, we need ALL pairs (a,b) not just i<=j. But since it's a set, duplicates don't matter.
    # Using i<=j gives the same set as all pairs because a+b = b+a.
    # Actually no: if i<j, we get U[i]+U[j] once. If we did all pairs, we'd get it twice. 
    # But since it's a set, the size is the same. So i<=j is fine.
    
    # Compute difference set U-U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U[i] - U[j])
    
    ratio = len(diffset) / len(sumset)
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try various initial constructions
    random.seed(42)
    
    # Strategy 1: Try small sets with various structures
    candidates = []
    
    # Arithmetic progressions
    for n in range(3, 30):
        for step in range(1, 10):
            U = list(range(0, n * step, step))
            if 0 in U:
                candidates.append(U)
    
    # Geometric-like sequences
    for base in range(2, 8):
        for n in range(3, 15):
            U = [base ** i for i in range(n)]
            if 0 in U:
                candidates.append(U)
            else:
                U.insert(0, 0)
                candidates.append(U)
    
    # Random sparse sets
    for trial in range(50):
        n = random.randint(4, 20)
        max_val = random.randint(10, 200)
        U = [0]
        remaining = list(range(1, max_val + 1))
        random.shuffle(remaining)
        for _ in range(n - 1):
            if remaining:
                U.append(remaining.pop())
        U.sort()
        candidates.append(U)
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Local search on the best candidate
    U = list(best_U)
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        n = len(U)
        maxU = max(U) if U else 0
        
        # Try different moves
        moves = []
        
        # Move 1: Add a new element
        if n < 30:
            # Try adding elements near existing ones or in gaps
            for _ in range(20):
                if random.random() < 0.5:
                    # Near an existing element
                    base = random.choice(U)
                    offset = random.randint(-5, 5)
                    new_val = base + offset
                else:
                    # Random position
                    new_val = random.randint(1, maxU + 20)
                
                if new_val > 0 and new_val not in U:
                    new_U = U + [new_val]
                    new_U.sort()
                    moves.append(new_U)
        
        # Move 2: Remove an element (if n > 2)
        if n > 2:
            for _ in range(5):
                idx = random.randint(1, n - 1)  # don't remove 0
                new_U = U[:idx] + U[idx+1:]
                moves.append(new_U)
        
        # Move 3: Replace an element
        if n >= 3:
            for _ in range(10):
                idx = random.randint(1, n - 1)  # don't replace 0
                maxU = max(U)
                new_val = random.randint(1, maxU + 10)
                if new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        # Move 4: Shift an element slightly
        if n >= 3:
            for _ in range(10):
                idx = random.randint(1, n - 1)
                delta = random.choice([-2, -1, 1, 2])
                new_val = U[idx] + delta
                if new_val > 0 and new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        if not moves:
            continue
        
        # Evaluate moves
        best_move_score = best_score
        best_move_U = U
        
        for new_U in moves:
            if time.time() > deadline:
                break
            s = compute_score(new_U)
            if s > best_move_score:
                best_move_score = s
                best_move_U = new_U
        
        if best_move_U != U:
            U = best_move_U
            best_score = best_move_score
            best_U = list(U)
    
    return best_U

# EVOLVE-BLOCK-END
