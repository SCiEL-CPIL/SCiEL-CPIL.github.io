"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _threshold(a_trans, b_trans):
    """Compute exact shortest reset word length via BFS on subset automaton."""
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if len(curr) == 1:
            return d
        na = frozenset(a_trans[i] for i in curr)
        if na not in visited:
            visited[na] = d + 1
            q.append(na)
        nb = frozenset(b_trans[i] for i in curr)
        if nb not in visited:
            visited[nb] = d + 1
            q.append(nb)
    return -1  # not synchronizing

def _fast_sync_check(a_trans, b_trans):
    """Quick check if automaton synchronizes using BFS, returns (syncs, depth_bound)."""
    start = frozenset(range(N))
    visited = {start}
    q = deque([(start, 0)])
    while q:
        curr, d = q.popleft()
        if len(curr) == 1:
            return True, d
        na = frozenset(a_trans[i] for i in curr)
        if na not in visited:
            visited.add(na)
            q.append((na, d + 1))
        nb = frozenset(b_trans[i] for i in curr)
        if nb not in visited:
            visited.add(nb)
            q.append((nb, d + 1))
    return False, 0

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    a_trans = [(i + 1) % N for i in range(N)]  # cyclic 'a'
    
    best_b = None
    best_score = 0
    
    # Try structured candidates first: 'b' maps most states to a small set
    # Pattern: b[i] = i for i < k, b[i] = 0 for i >= k (or similar)
    structured_candidates = []
    for k in range(1, N):
        for target in range(N):
            b = list(a_trans)
            for i in range(k, N):
                b[i] = target
            structured_candidates.append(b)
    # Also try: b[i] = i for i<k, b[i] = (i-1) for i>=k
    for k in range(1, N):
        b = list(a_trans)
        for i in range(k, N):
            b[i] = i - 1
        structured_candidates.append(b)
    # b[i] = (i * c) % N for various c (permutations won't work, but near-permutations)
    for c in range(2, N):
        b = [(i * c) % N for i in range(N)]
        # Make it non-permutation by collapsing one state
        b[N-1] = b[N-2]
        structured_candidates.append(b)
    
    for b in structured_candidates:
        if time.time() > deadline:
            break
        syncs, _ = _fast_sync_check(a_trans, b)
        if not syncs:
            continue
        score = _threshold(a_trans, b)
        if score > best_score:
            best_score = score
            best_b = b[:]
    
    # Random search
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Generate random 'b' with bias toward low-rank (many states map to same target)
        # Strategy: pick a small set of target states, map most states into it
        r = random.random()
        if r < 0.3:
            # Map all states to 1-3 targets
            num_targets = random.randint(1, 3)
            targets = random.sample(range(N), num_targets)
            b = [random.choice(targets) for _ in range(N)]
        elif r < 0.6:
            # Mostly identity with a few collapses
            b = list(range(N))
            num_collapses = random.randint(1, 4)
            for _ in range(num_collapses):
                i = random.randint(0, N-1)
                j = random.randint(0, N-1)
                b[i] = b[j]
        elif r < 0.8:
            # Cyclic shift with perturbations
            shift = random.randint(1, N-1)
            b = [(i + shift) % N for i in range(N)]
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                i = random.randint(0, N-1)
                b[i] = random.randint(0, N-1)
        else:
            # Fully random
            b = [random.randint(0, N-1) for _ in range(N)]
        
        # Quick filter: must not be a permutation (permutations don't synchronize with one other letter easily)
        if len(set(b)) == N:
            continue
        
        syncs, _ = _fast_sync_check(a_trans, b)
        if not syncs:
            continue
        
        score = _threshold(a_trans, b)
        if score > best_score:
            best_score = score
            best_b = b[:]
    
    if best_b is None:
        # Fallback: simple synchronizing automaton
        best_b = [0] * N
        best_score = _threshold(a_trans, best_b)
    
    return [a_trans, best_b]
# EVOLVE-BLOCK-END
