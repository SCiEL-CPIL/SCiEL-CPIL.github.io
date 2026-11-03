"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations
from collections import defaultdict

D = 11

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    # Strategy: Combine shell m=2 and m=4, then use a sophisticated local search
    # with simulated annealing style acceptance and kick-based exploration.
    
    shell2 = _generate_shell(2)
    shell4 = _generate_shell(4)
    
    # Concatenate
    shell = np.vstack([shell2, shell4])
    n = len(shell)
    
    # Compute norms squared
    norms_sq = np.sum(shell * shell, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Conflict: two vectors u, v conflict if ||u - v||^2 < max_norm_sq
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v < max_norm_sq
    # Since max_norm_sq = 4 (from shell4), and min norm_sq = 2 (from shell2),
    # For two shell4 vectors: 4+4-2*u.v < 4 => u.v > 2
    # For two shell2 vectors: 2+2-2*u.v < 4 => -2*u.v < 0 => u.v > 0 (but |u.v| <= 2 for m=2, so u.v can be -2,-1,0,1,2)
    # Actually for m=2: u.v can be -2, -1, 0, 1, 2. If u.v = 1, then dist^2 = 4 - 2 = 2 < 4. Conflict.
    # If u.v = 0, dist^2 = 4. OK.
    # If u.v = -1, dist^2 = 4+2=6. OK.
    # If u.v = -2, dist^2 = 8. OK.
    # So for shell2-shell2: conflict if u.v > 0, i.e., u.v = 1 or 2.
    # For shell2-shell4: 2+4-2*u.v < 4 => 6-2*u.v < 4 => 2 < 2*u.v => u.v > 1.
    # u.v for shell2 vs shell4: can be -2..2. Conflict if u.v = 2.
    # For shell4-shell2: same, u.v = 2.
    
    # Let's just compute the full conflict matrix using the general condition.
    dots = shell @ shell.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = dist_sq < max_norm_sq
    np.fill_diagonal(conflict, False)
    
    # Build adjacency lists
    adj = [set() for _ in range(n)]
    for i in range(n):
        conflicts = np.where(conflict[i])[0]
        adj[i] = set(conflicts)
    
    # Local search with kicks and greedy fill
    def greedy_fill(S_set, time_limit=deadline - 5):
        """Greedily add as many non-conflicting vectors as possible to S_set."""
        S = S_set.copy()
        # Compute which nodes are blocked by current S
        blocked = set()
        for s in S:
            blocked.update(adj[s])
        
        # Available nodes
        available = set(range(n)) - S - blocked
        
        while available and time.time() < time_limit:
            # Pick a node with minimum degree in the available subgraph
            # Approximate by picking random or min global degree
            # For speed, pick random
            idx = next(iter(available))
            S.add(idx)
            available.discard(idx)
            # Update blocked
            new_blocked = adj[idx] - S
            blocked.update(new_blocked)
            available -= new_blocked
            
        return S
    
    def kick_and_fill(S, kick_size, time_limit=deadline - 5):
        """Remove kick_size random elements from S, then greedily fill."""
        if len(S) <= kick_size:
            return S
        
        removed = set(np.random.choice(list(S), size=kick_size, replace=False))
        S_kick = S - removed
        
        # Greedily fill
        S_new = greedy_fill(S_kick, time_limit)
        
        return S_new
    
    # Main search loop
    best_S = set()
    best_size = 0
    
    # Initial solution: greedy
    current_S = greedy_fill(set())
    best_S = current_S.copy()
    best_size = len(current_S)
    
    # Local search with kicks
    num_kicks = 0
    kick_sizes = [1, 2, 3, 4, 5]
    
    while time.time() < deadline - 10:
        # Try different kick sizes
        kick_size = kick_sizes[num_kicks % len(kick_sizes)]
        num_kicks += 1
        
        S_new = kick_and_fill(current_S, kick_size)
        
        if len(S_new) > best_size:
            best_size = len(S_new)
            best_S = S_new.copy()
            current_S = S_new.copy()
        elif len(S_new) > len(current_S):
            # Accept if it's better than current (even if not best)
            current_S = S_new.copy()
        else:
            # Occasionally accept worse solutions to escape local optima
            if np.random.random() < 0.1 and len(S_new) >= len(current_S) - 2:
                current_S = S_new.copy()
        
        # Periodically restart from best
        if num_kicks % 50 == 0:
            current_S = best_S.copy()
    
    # Convert best_S to vectors
    if best_S:
        result = shell[list(best_S)]
    else:
        result = shell[:1]
        
    return result
