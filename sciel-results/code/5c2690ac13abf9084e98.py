"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_conflicts():
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    
    # Use numpy for faster distance computation
    pts = np.array(points, dtype=np.int32)
    
    # Compute all pairwise squared distances
    # dist[i,j] = squared distance between points[i] and points[j]
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
    
    # conflict_pairs[i][j] is a set of indices k such that triangle i,j,k is isosceles
    conflict_pairs = [[set() for _ in range(V)] for __ in range(V)]
    
    for i in range(V):
        for j in range(i+1, V):
            d_ij = dist[i, j]
            # For each k, check if d(i,k) == d_ij or d(j,k) == d_ij or d(i,k) == d(j,k)
            d_ik = dist[i]  # array of d(i,k) for all k
            d_jk = dist[j]  # array of d(j,k) for all k
            
            # Find k where isosceles condition holds
            # d_ij == d_ik[k] or d_ij == d_jk[k] or d_ik[k] == d_jk[k]
            mask = (d_ik == d_ij) | (d_jk == d_ij) | (d_ik == d_jk)
            mask[i] = False  # exclude i
            mask[j] = False  # exclude j
            
            ks = np.where(mask)[0]
            for k in ks:
                conflict_pairs[i][j].add(k)
                conflict_pairs[j][i].add(k)
    
    return points, conflict_pairs, dist

# Global variables initialized at module load or first construct
_points, _conflict_pairs, _dist = None, None, None

def _init():
    global _points, _conflict_pairs, _dist
    if _points is None:
        _points, _conflict_pairs, _dist = _precompute_conflicts()

def _is_valid_set(S, conflict_pairs):
    """Check if S is a valid no-isosceles set."""
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a, b = S[i], S[j]
            # Check if any point in S is in conflict_pairs[a][b]
            for c in S:
                if c != a and c != b and c in conflict_pairs[a][b]:
                    return False
    return True

def _is_maximal(S, U, conflict_pairs):
    """Check if S is maximal: no point in U can be added."""
    if not U:
        return True
    forbidden = set()
    for i in range(len(S)):
        a = S[i]
        for j in range(i+1, len(S)):
            b = S[j]
            forbidden |= conflict_pairs[a][b]
    # If any point in U is not in forbidden, we can add it
    for p in U:
        if p not in forbidden:
            return False
    return True

def _local_search(S, conflict_pairs, V, max_iters=50):
    """Try to improve S by swapping points or removing/adding."""
    S_set = set(S)
    U = set(range(V)) - S_set
    
    for _ in range(max_iters):
        improved = False
        
        # Try removing a point and adding a better one
        for idx, a in enumerate(list(S)):
            S_trial = list(S)
            S_trial.pop(idx)
            S_trial_set = set(S_trial)
            U_trial = (set(range(V)) - S_trial_set)
            
            # Find best point to add
            forbidden = set()
            for i in range(len(S_trial)):
                x = S_trial[i]
                for j in range(i+1, len(S_trial)):
                    y = S_trial[j]
                    forbidden |= conflict_pairs[x][y]
            
            best_candidate = None
            best_score = -1
            for p in U_trial:
                if p in forbidden:
                    continue
                # Score: how many points does this block?
                newly_blocked = set()
                for x in S_trial:
                    newly_blocked |= conflict_pairs[p][x]
                newly_blocked &= U_trial
                score = len(newly_blocked)
                if score > best_score:
                    best_score = score
                    best_candidate = p
            
            if best_candidate is not None:
                # Check if this swap improves (same size, so we just want to keep searching)
                # Actually, swapping keeps size same, so we need a different criterion
                # For now, skip swaps and focus on other improvements
                pass
        
        # Try removing a point that allows adding two others
        # This is more complex, skip for now
        
        if not improved:
            break
    
    return S

def _greedy_construct(conflict_pairs, V, strategy='blocking', random_seed=None):
    """Greedy construction with different strategies."""
    if random_seed is not None:
        random.seed(random_seed)
    
    S = []
    U = set(range(V))
    
    while U:
        # Compute forbidden set
        forbidden = set()
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                forbidden |= conflict_pairs[a][b]
        
        # Find candidates
        candidates = [p for p in U if p not in forbidden]
        
        if not candidates:
            break
        
        best_candidate = None
        best_score = -1
        best_newly_blocked = set()
        
        for p in candidates:
            newly_blocked = set()
            for a in S:
                newly_blocked |= conflict_pairs[p][a]
            newly_blocked &= U
            
            if strategy == 'blocking':
                score = len(newly_blocked)
            elif strategy == 'random':
                score = random.random()
            else:
                score = len(newly_blocked)
            
            if score > best_score:
                best_score = score
                best_candidate = p
                best_newly_blocked = newly_blocked
            elif score == best_score and best_candidate is not None and strategy != 'random':
                if random.random() < 0.3:
                    best_candidate = p
                    best_newly_blocked = newly_blocked
        
        if best_candidate is None:
            break
        
        S.append(best_candidate)
        U.remove(best_candidate)
        U -= best_newly_blocked
    
    return S

def construct():
    _init()
    points = _points
    conflict_pairs = _conflict_pairs
    V = len(points)
    
    best_size = float('inf')
    best_set = None
    start_time = time.time()
    deadline = start_time + 285.0  # 15s buffer for safety
    
    strategies = ['blocking', 'random', 'blocking']
    strategy_idx = 0
    trial_count = 0
    
    while time.time() < deadline:
        strategy = strategies[trial_count % len(strategies)]
        seed = trial_count * 7919  # deterministic seed per trial
        
        S = _greedy_construct(conflict_pairs, V, strategy=strategy, random_seed=seed)
        
        # Try local search
        if len(S) < best_size:
            S_improved = _local_search(S, conflict_pairs, V, max_iters=20)
            if len(S_improved) < len(S):
                S = S_improved
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
        
        trial_count += 1
        
        # If we've found a very good solution, do more intensive search
        if best_size <= 8 and time.time() < deadline - 30:
            # Try more diverse strategies
            for _ in range(5):
                if time.time() >= deadline:
                    break
                S_alt = _greedy_construct(conflict_pairs, V, strategy='random', random_seed=trial_count * 13 + 1000)
                S_alt_improved = _local_search(S_alt, conflict_pairs, V, max_iters=30)
                if len(S_alt_improved) < best_size:
                    best_size = len(S_alt_improved)
                    best_set = list(S_alt_improved)
        
        if best_size <= 6:
            # Very good, can stop
            break
    
    if best_set is None:
        return []
        
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
