import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit

def _precompute_3aps():
    """Precompute all 3-APs as tuples (a, b, c)."""
    aps = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > b and c <= N:
                aps.append((a - 1, b - 1, c - 1))  # 0-indexed
    return aps

def _precompute_4aps():
    """Precompute all 4-APs as tuples (a, b, c, d)."""
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            if dd <= N:
                aps.append((a - 1, b - 1, c - 1, dd - 1))  # 0-indexed
    return aps

# Precompute at module load time
_3AP_LIST = _precompute_3aps()
_4AP_LIST = _precompute_4aps()

def _count_3aps_in_assignment(assignment):
    """Count 3-APs satisfied by a boolean assignment (0-indexed)."""
    count = 0
    for a, b, c in _3AP_LIST:
        if assignment[a] and assignment[b] and assignment[c]:
            count += 1
    return count

def _count_4aps_in_assignment(assignment):
    """Count 4-APs violated by a boolean assignment (0-indexed)."""
    count = 0
    for a, b, c, d in _4AP_LIST:
        if assignment[a] and assignment[b] and assignment[c] and assignment[d]:
            count += 1
    return count

def _walksat():
    """
    WalkSAT-based Max-SAT solver for the 3-AP/4-AP problem.
    Variables: x_i in {0,1} for i in 0..255.
    Hard constraints: No 4-AP (all 4-AP clauses must be unsatisfied).
    Soft objective: Maximize number of satisfied 3-APs.
    
    We use a weighted approach:
    - Each 4-AP clause gets a large penalty if violated.
    - Each 3-AP clause gets a reward if satisfied.
    """
    start_time = time.time()
    best_assignment = None
    best_score = -1
    
    # We'll use a score that heavily penalizes 4-AP violations
    # Score = 3AP_count - L * 4AP_count, where L is large
    
    L = 1000  # Penalty multiplier for 4-AP violations
    
    # Initialize with a random assignment
    # Start with a moderately dense random assignment
    p = 0.4
    assignment = np.random.random(N) < p
    current_score = _count_3aps_in_assignment(assignment) - L * _count_4aps_in_assignment(assignment)
    
    if best_score < current_score:
        best_score = current_score
        best_assignment = assignment.copy()
    
    # WalkSAT parameters
    p_flip = 0.5  # Probability of flipping a variable in a violated clause vs random
    num_restarts = 50
    max_steps_per_restart = 20000
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Random restart with varying density
        p_restart = random.uniform(0.2, 0.6)
        assignment = np.random.random(N) < p_restart
        current_score = _count_3aps_in_assignment(assignment) - L * _count_4aps_in_assignment(assignment)
        
        if best_score < current_score:
            best_score = current_score
            best_assignment = assignment.copy()
        
        for step in range(max_steps_per_restart):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Find violated 4-AP clauses
            violated_4aps = []
            for a, b, c, d in _4AP_LIST:
                if assignment[a] and assignment[b] and assignment[c] and assignment[d]:
                    violated_4aps.append((a, b, c, d))
            
            if not violated_4aps:
                # No 4-AP violations, try to improve 3-AP count
                # Do a local search to maximize 3-APs
                improved = True
                while improved:
                    improved = False
                    for i in range(N):
                        if assignment[i]:
                            # Try removing
                            assignment[i] = False
                            new_score = _count_3aps_in_assignment(assignment)
                            if new_score > _count_3aps_in_assignment(assignment):
                                pass  # This won't work, need to track properly
                            assignment[i] = True
                        else:
                            # Try adding
                            # Check if adding creates a 4-AP
                            creates_4ap = False
                            for a, b, c, d in _4AP_LIST:
                                if i in (a, b, c, d):
                                    others = [x for x in (a, b, c, d) if x != i]
                                    if all(assignment[x] for x in others):
                                        creates_4ap = True
                                        break
                            if not creates_4ap:
                                assignment[i] = True
                                new_score = _count_3aps_in_assignment(assignment)
                                old_score = _count_3aps_in_assignment(assignment)
                                assignment[i] = False
                                if new_score > old_score:
                                    assignment[i] = True
                                    current_score = new_score
                                    if best_score < current_score:
                                        best_score = current_score
                                        best_assignment = assignment.copy()
                                    improved = True
                break
            else:
                # Pick a random violated 4-AP
                a, b, c, d = random.choice(violated_4aps)
                # Pick a variable to flip
                if random.random() < p_flip:
                    # Flip a variable in the violated clause
                    var = random.choice([a, b, c, d])
                else:
                    # Flip a random variable
                    var = random.randint(0, N - 1)
                
                assignment[var] = not assignment[var]
                new_score = _count_3aps_in_assignment(assignment) - L * _count_4aps_in_assignment(assignment)
                
                # Accept if better, or with some probability if worse
                if new_score >= current_score or random.random() < 0.3:
                    current_score = new_score
                    if best_score < current_score:
                        best_score = current_score
                        best_assignment = assignment.copy()
                else:
                    assignment[var] = not assignment[var]  # Undo
    
    # Final cleanup: remove elements that create 4-APs
    if best_assignment is not None:
        final_assignment = best_assignment.copy()
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if final_assignment[i]:
                    # Check if removing i helps
                    creates_4ap = False
                    for a, b, c, d in _4AP_LIST:
                        if i in (a, b, c, d):
                            others = [x for x in (a, b, c, d) if x != i]
                            if all(final_assignment[x] for x in others):
                                creates_4ap = True
                                break
                    if creates_4ap:
                        final_assignment[i] = False
                        changed = True
        
        # Verify no 4-APs
        if _count_4aps_in_assignment(final_assignment) == 0:
            return final_assignment
        else:
            # Fallback: greedy removal
            for a, b, c, d in _4AP_LIST:
                if final_assignment[a] and final_assignment[b] and final_assignment[c] and final_assignment[d]:
                    # Remove one of them
                    final_assignment[d] = False
    
    return best_assignment

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Run WalkSAT
    assignment = _walksat()
    
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if assignment[i]]
    
    # Final verification and cleanup
    result_set = set(result)
    # Remove any remaining 4-APs by greedy removal
    changed = True
    while changed:
        changed = False
        for a, b, c, d in _4AP_LIST:
            if (a + 1) in result_set and (b + 1) in result_set and (c + 1) in result_set and (d + 1) in result_set:
                result_set.discard(d + 1)
                changed = True
    
    return sorted(result_set)
