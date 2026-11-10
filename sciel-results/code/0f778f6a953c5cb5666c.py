import numpy as np
import random
import time
from collections import deque

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start = time.time()
    deadline = start + 580  # safety margin before 630s kill

    # Precompute forbidden differences mod 205
    forb = set()
    for x in range(1, m):
        forb.add((x * x) % m)
    forb.discard(0)

    # Build conflict bitmasks: conflict[i] is a 205-bit integer where bit j is 1
    # if (i-j) mod 205 is in forb (i.e., i and j cannot both be in the set)
    # We use Python integers as arbitrary-precision bitmasks
    conflict = [0] * m
    for i in range(m):
        mask = 0
        for d in forb:
            j = (i + d) % m
            mask |= (1 << j)
            j2 = (i - d) % m
            mask |= (1 << j2)
        mask &= ~(1 << i)  # don't conflict with self
        conflict[i] = mask

    def can_add(v, S_mask):
        """Check if v can be added to set represented by bitmask S_mask."""
        return (conflict[v] & S_mask) == 0

    def greedy(order, S_mask=0):
        """Greedy construction from given order."""
        S = set()
        for i in order:
            if not (S_mask >> i) & 1 and can_add(i, S_mask):
                S.add(i)
                S_mask |= (1 << i)
        return S, S_mask

    def local_search(S, S_mask):
        """Iterative local search: try to add elements, then do 1-out-2-in moves."""
        improved = True
        while improved:
            improved = False
            # Phase 1: try to add any element
            for v in range(m):
                if not (S_mask >> v) & 1 and can_add(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << v)
                    improved = True
                    break
            if improved:
                continue
            # Phase 2: try 1-out-2-in moves
            S_list = list(S)
            random.shuffle(S_list)
            for u in S_list:
                S.remove(u)
                S_mask &= ~(1 << u)
                # Find candidates that can be added
                candidates = []
                for v in range(m):
                    if not (S_mask >> v) & 1 and can_add(v, S_mask):
                        candidates.append(v)
                # Try to find two non-conflicting candidates
                found = False
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        if (conflict[candidates[i]] >> candidates[j]) & 1 == 0:
                            S.add(candidates[i])
                            S.add(candidates[j])
                            S_mask |= (1 << candidates[i]) | (1 << candidates[j])
                            found = True
                            break
                    if found:
                        break
                if found:
                    improved = True
                    break
                else:
                    # Undo
                    S.add(u)
                    S_mask |= (1 << u)
        return S, S_mask

    def simulated_annealing(S, S_mask, T_initial=100.0, T_final=0.01, n_iter=50000):
        """Simulated annealing on the independent set problem."""
        current = S.copy()
        current_mask = S_mask
        best = S.copy()
        best_mask = S_mask
        best_size = len(S)
        
        T = T_initial
        cool = (T_final / T_initial) ** (1.0 / n_iter)
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            T *= cool
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.3:
                # Try to add a random element
                v = random.randint(0, m - 1)
                if not (current_mask >> v) & 1 and can_add(v, current_mask):
                    new_S = current | {v}
                    new_mask = current_mask | (1 << v)
                    delta = 1
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current = new_S
                        current_mask = new_mask
                        if len(current) > best_size:
                            best = current.copy()
                            best_mask = current_mask
                            best_size = len(current)
            elif move_type < 0.5:
                # Try to remove a random element (rarely helpful but helps escape)
                if current:
                    v = random.choice(list(current))
                    new_S = current - {v}
                    new_mask = current_mask & ~(1 << v)
                    delta = -1
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current = new_S
                        current_mask = new_mask
            else:
                # Try a swap: remove u, add v
                if current:
                    u = random.choice(list(current))
                    v = random.randint(0, m - 1)
                    if v not in current:
                        # Check if removing u and adding v is valid
                        temp_mask = current_mask & ~(1 << u)
                        if can_add(v, temp_mask):
                            new_S = (current - {u}) | {v}
                            new_mask = temp_mask | (1 << v)
                            delta = 0
                            if random.random() < np.exp(delta / T):
                                current = new_S
                                current_mask = new_mask
                                if len(current) > best_size:
                                    best = current.copy()
                                    best_mask = current_mask
                                    best_size = len(current)
        
        return best, best_mask

    # Multi-start local search
    best_S = set()
    best_mask = 0
    best_size = 0
    
    # Run multiple random restarts
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        S, S_mask = greedy(order)
        S, S_mask = local_search(S, S_mask)
        if len(S) > best_size:
            best_S = S.copy()
            best_mask = S_mask
            best_size = len(S)
    
    # Run simulated annealing from the best found so far
    if time.time() < deadline - 60:
        sa_S, sa_mask = simulated_annealing(best_S, best_mask, T_initial=50.0, T_final=0.01, n_iter=30000)
        if len(sa_S) > best_size:
            best_S = sa_S.copy()
            best_mask = sa_mask
            best_size = len(sa_S)
    
    # Additional local search from SA result
    if time.time() < deadline - 30:
        S, S_mask = local_search(best_S, best_mask)
        if len(S) > best_size:
            best_S = S.copy()
            best_mask = S_mask
            best_size = len(S)
    
    # Final verification
    result = list(best_S)
    for a in result:
        for b in result:
            if a != b:
                d = (a - b) % m
                if d in forb:
                    # Should not happen, but fallback
                    result = []
                    break
        if not result:
            break
    
    if not result:
        # Fallback: simple greedy
        S = set()
        S_mask = 0
        for i in range(m):
            if can_add(i, S_mask):
                S.add(i)
                S_mask |= (1 << i)
        result = list(S)
    
    return np.array(result, dtype=np.int64)
