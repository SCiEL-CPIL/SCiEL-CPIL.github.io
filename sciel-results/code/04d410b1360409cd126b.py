import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0

def _build_conflicts():
    conflicts = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                # The 6 elements are distinct as proven above
                conflict = frozenset([a, b, c, s1, s2, s3])
                conflicts.append(conflict)
    return conflicts

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN
    
    conflicts = _build_conflicts()
    num_conflicts = len(conflicts)
    
    # Map vertex to list of conflict indices
    vert_to_conf = {v: [] for v in range(1, N + 1)}
    for i, c in enumerate(conflicts):
        for v in c:
            vert_to_conf[v].append(i)
            
    # Precompute conflict sizes (all are 6)
    conf_sizes = [len(c) for c in conflicts]

    def _count_violations(S):
        counts = [0] * num_conflicts
        for v in S:
            for ci in vert_to_conf[v]:
                counts[ci] += 1
        violations = 0
        for i in range(num_conflicts):
            if counts[i] == conf_sizes[i]:
                violations += 1
        return violations, counts

    def _delta_violations(v, S, counts):
        """
        Returns delta in violation count if v is flipped.
        Assumes v is NOT in S (so we are adding it).
        If v IS in S, we are removing it.
        """
        delta = 0
        if v in S:
            # Removing v
            for ci in vert_to_conf[v]:
                if counts[ci] == conf_sizes[ci]:
                    delta -= 1
        else:
            # Adding v
            for ci in vert_to_conf[v]:
                if counts[ci] == conf_sizes[ci] - 1:
                    delta += 1
        return delta

    def _update_counts(v, delta, counts):
        for ci in vert_to_conf[v]:
            counts[ci] += delta

    best_set = set(range(65, N + 1))
    best_size = len(best_set)
    best_violations, best_counts = _count_violations(best_set)
    
    # Simulated Annealing
    # Start with a random set to explore more space, or stick to large half?
    # Let's try multiple restarts with different initializations.
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            cur_set = set(range(65, N + 1))
        elif r % 2 == 0:
            # Bias towards larger numbers
            cur_set = set()
            for v in range(1, N + 1):
                p = 0.8 if v > 64 else 0.3
                if random.random() < p:
                    cur_set.add(v)
        else:
            # Random
            cur_set = set()
            for v in range(1, N + 1):
                if random.random() < 0.5:
                    cur_set.add(v)
            
        cur_violations, cur_counts = _count_violations(cur_set)
        cur_size = len(cur_set)
        
        # Objective: Minimize ( -size + lambda * violations )
        # We want to maximize size, so minimizing -size.
        # Penalty for violations should be high to encourage valid sets.
        # But we must be able to move through invalid states to escape local optima.
        # Let's use a score: score = size - 10 * violations
        # We want to maximize score.
        
        T_start = 10.0
        T_end = 0.01
        T_decay = 0.995
        T = T_start
        
        sa_best_set = cur_set.copy()
        sa_best_size = cur_size
        sa_best_violations = cur_violations
        
        iterations = 0
        max_iters = 200000
        
        while T > T_end and iterations < max_iters:
            if time.time() > deadline:
                break
                
            v = random.randint(1, N)
            
            if v in cur_set:
                # Remove v
                delta_v = _delta_violations(v, cur_set, cur_counts)
                delta_size = -1
                delta_score = -delta_size - 10 * delta_v
                # Wait, score = size - 10*viol.
                # New score = (size-1) - 10*(viol+delta_v) = score - 1 - 10*delta_v.
                # Delta score = -1 - 10*delta_v.
                
                if delta_score > 0 or random.random() < (1.0 / (1.0 + delta_score * T)):
                    cur_set.remove(v)
                    _update_counts(v, -1, cur_counts)
                    cur_size += delta_size
                    cur_violations += delta_v
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_set = cur_set.copy()
                        sa_best_violations = 0
            else:
                # Add v
                delta_v = _delta_violations(v, cur_set, cur_counts)
                delta_size = 1
                delta_score = -delta_size - 10 * delta_v
                # New score = (size+1) - 10*(viol+delta_v) = score + 1 - 10*delta_v.
                # Delta score = 1 - 10*delta_v.
                
                if delta_score > 0 or random.random() < (1.0 / (1.0 + delta_score * T)):
                    cur_set.add(v)
                    _update_counts(v, 1, cur_counts)
                    cur_size += delta_size
                    cur_violations += delta_v
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_set = cur_set.copy()
                        sa_best_violations = 0
            
            T *= T_decay
            iterations += 1
            
        # After SA, try to polish
        if sa_best_violations == 0 and sa_best_size > best_size:
            best_size = sa_best_size
            best_set = sa_best_set.copy()
            best_violations, best_counts = _count_violations(best_set)
            
        # Local Search on the best set found in this restart
        if sa_best_violations == 0:
            ls_set = sa_best_set.copy()
            ls_violations, ls_counts = _count_violations(ls_set)
            ls_size = len(ls_set)
            
            # Try to add any possible vertex
            improved = True
            while improved:
                improved = False
                for v in range(1, N + 1):
                    if v not in ls_set:
                        if _delta_violations(v, ls_set, ls_counts) == 0:
                            ls_set.add(v)
                            _update_counts(v, 1, ls_counts)
                            ls_size += 1
                            improved = True
                            
            if ls_size > best_size:
                best_size = ls_size
                best_set = ls_set.copy()
                best_violations, best_counts = _count_violations(best_set)

    # Final Polish
    if best_violations == 0:
        # Try to add more
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if v not in best_set:
                    if _delta_violations(v, best_set, best_counts) == 0:
                        best_set.add(v)
                        _update_counts(v, 1, best_counts)
                        best_size += 1
                        improved = True

    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
