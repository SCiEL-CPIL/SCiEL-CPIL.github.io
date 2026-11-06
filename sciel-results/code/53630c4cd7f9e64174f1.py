import time
import random
from itertools import combinations
from collections import defaultdict

def count_conflicts(S):
    """Count the number of duplicate pair sums in S.
    More conflicts = smaller max Sidon subset likely."""
    sum_counts = defaultdict(int)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            sum_counts[s] += 1
    return sum(1 for c in sum_counts.values() if c > 1)

def is_sidon_fast(S):
    """Check if a list of integers is a Sidon set using bitmask."""
    mask = 0
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            bit = 1 << s
            if mask & bit:
                return False
            mask |= bit
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon_fast(subset):
                return k
    return 2 # Any 2 distinct elements form a Sidon set

def greedy_conflict_set(max_val=25, n=12, seed=None):
    """Build a set greedily to maximize pair-sum collisions."""
    if seed is not None:
        random.seed(seed)
    
    S = []
    used = set()
    
    # Start with some initial elements
    while len(S) < n:
        # Try each candidate value and pick the one that maximizes conflicts
        best_val = None
        best_conflicts = -1
        
        for v in range(0, max_val + 1):
            if v in used:
                continue
            
            # Count conflicts if we add v
            conflicts = 0
            for i in range(len(S)):
                s = S[i] + v
                # Count how many existing pairs have this sum
                for j in range(i):
                    if S[j] + S[i] == s:
                        conflicts += 1
                # Also count diagonal
                if v + v == s:
                    conflicts += 1
            
            if conflicts > best_conflicts:
                best_conflicts = conflicts
                best_val = v
        
        if best_val is not None:
            S.append(best_val)
            used.add(best_val)
        else:
            # Fallback: pick random unused
            available = [v for v in range(0, max_val + 1) if v not in used]
            if not available:
                break
            v = random.choice(available)
            S.append(v)
            used.add(v)
    
    S.sort()
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_set = None
    best_score = 13 # Max possible is 12, so 13 is safe init
    
    # Phase 1: Generate diverse initial candidates using different strategies
    candidates = []
    
    # Strategy 1: Random dense sets in small range
    for _ in range(20):
        S = random.sample(range(0, 26), 12)
        S.sort()
        candidates.append(S)
    
    # Strategy 2: Greedy conflict-maximizing sets
    for seed in range(10):
        S = greedy_conflict_set(max_val=25, n=12, seed=seed)
        candidates.append(S)
    
    # Strategy 3: Arithmetic progressions (dense in small range)
    for d in range(1, 4):
        for a in range(0, 26 - 11*d + 1):
            S = [a + i*d for i in range(12)]
            if S[-1] <= 25:
                candidates.append(S)
    
    # Strategy 4: Structured sets with forced collisions
    # e.g., pairs that sum to same value
    for base in range(0, 13):
        S = []
        # Add pairs (base, base+1), (base+2, base+1), etc. to force collisions
        for k in range(6):
            S.append(base + k)
            S.append(base + 1 + k)
        S = sorted(set(S))
        if len(S) == 12:
            candidates.append(S)
    
    # Evaluate initial candidates
    for S in candidates:
        if time.time() > deadline: break
        score = max_sidon_size(S)
        if score < best_score:
            best_score = score
            best_set = S[:]
    
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
    
    S = best_set[:]
    
    # Phase 2: Local search with conflict-aware moves
    while time.time() < deadline:
        improved = False
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try several moves for this position
            for _ in range(15):
                if time.time() > deadline:
                    break
                
                # Sample from small dense range
                new_val = random.randint(0, 28)
                if new_val in S:
                    continue
                
                new_S = S[:]
                new_S[i] = new_val
                new_S.sort()
                
                # Quick filter: check conflict count first (faster than max_sidon_size)
                new_conflicts = count_conflicts(new_S)
                current_conflicts = count_conflicts(S)
                
                # Only do expensive max_sidon_size if conflicts increased or are equal
                if new_conflicts > current_conflicts or (new_conflicts == current_conflicts and random.random() < 0.1):
                    score = max_sidon_size(new_S)
                    if score < best_score:
                        best_score = score
                        S = new_S
                        best_set = S[:]
                        improved = True
                        break
                    elif score == best_score and random.random() < 0.08:
                        # Accept equal to escape local minima
                        S = new_S
                        best_set = S[:]
            
            if improved:
                indices = list(range(12))
                random.shuffle(indices)
    
    # Phase 3: Final polish - try systematic improvements
    if time.time() < deadline:
        for i in range(12):
            if time.time() > deadline:
                break
            for new_val in range(0, 30):
                if new_val in S:
                    continue
                new_S = S[:]
                new_S[i] = new_val
                new_S.sort()
                score = max_sidon_size(new_S)
                if score < best_score:
                    best_score = score
                    S = new_S
                    best_set = S[:]
    
    return best_set
