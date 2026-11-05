import math
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    def is_compatible(m1, r1, m2, r2):
        g = gcd(m1, m2)
        return r1 % g != r2 % g
    
    def is_valid_assignment(assignment):
        """Check if a list of (m, r) is pairwise disjoint."""
        n = len(assignment)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = assignment[i]
                m2, r2 = assignment[j]
                if not is_compatible(m1, r1, m2, r2):
                    return False
        return True
    
    def score(assignment):
        return sum(1.0/m for m, r in assignment)
    
    def greedy_build(order, assignment=None):
        """Greedy: add moduli in given order, pick best valid residue."""
        if assignment is None:
            assignment = []
        for m in order:
            if any(m == sm for sm, _ in assignment):
                continue
            # Find a valid residue
            best_r = None
            for r in range(m):
                valid = True
                for sm, sr in assignment:
                    if not is_compatible(m, r, sm, sr):
                        valid = False
                        break
                if valid:
                    best_r = r
                    break
            if best_r is not None:
                assignment.append((m, best_r))
        return assignment
    
    # Strategy 1: Greedy with moduli in increasing order
    best_assignment = []
    best_score = 0.0
    
    # Try greedy with increasing moduli
    order1 = list(range(20, 201))
    a1 = greedy_build(order1)
    s1 = score(a1)
    if s1 > best_score:
        best_score = s1
        best_assignment = a1
    
    # Try greedy with decreasing moduli
    order2 = list(range(200, 19, -1))
    a2 = greedy_build(order2)
    s2 = score(a2)
    if s2 > best_score:
        best_score = s2
        best_assignment = a2
    
    # Local search: try to improve by swapping or adding
    def local_search(assignment, max_iter=5000):
        assignment = list(assignment)
        current_score = score(assignment)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 10:
            improved = False
            iters += 1
            # Try adding a new modulus
            for m in range(20, 201):
                if any(m == sm for sm, _ in assignment):
                    continue
                for r in range(m):
                    valid = True
                    for sm, sr in assignment:
                        if not is_compatible(m, r, sm, sr):
                            valid = False
                            break
                    if valid:
                        new_assignment = assignment + [(m, r)]
                        new_score = current_score + 1.0/m
                        if new_score > current_score + 1e-12:
                            assignment = new_assignment
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # Try swapping: remove one, add one
            for i in range(len(assignment)):
                for m_new in range(20, 201):
                    if any(m_new == sm for sm, _ in assignment if sm != assignment[i][0]):
                        continue
                    if m_new == assignment[i][0]:
                        continue
                    for r_new in range(m_new):
                        # Check compatibility with all except i
                        valid = True
                        for j in range(len(assignment)):
                            if j == i:
                                continue
                            sm, sr = assignment[j]
                            if not is_compatible(m_new, r_new, sm, sr):
                                valid = False
                                break
                        if valid:
                            new_assignment = [assignment[j] for j in range(len(assignment)) if j != i] + [(m_new, r_new)]
                            new_score = current_score - 1.0/assignment[i][0] + 1.0/m_new
                            if new_score > current_score + 1e-12:
                                assignment = new_assignment
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return assignment, current_score
    
    # Apply local search
    best_assignment, best_score = local_search(best_assignment, max_iter=2000)
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        # Random order with bias toward small moduli
        order = list(range(20, 201))
        random.shuffle(order)
        # Bias: put small moduli earlier with some probability
        order.sort(key=lambda m: (random.random() > 0.5, m))
        
        a = greedy_build(order)
        s = score(a)
        if s > best_score:
            best_score = s
            best_assignment = a
        
        # Local search on this
        a_ls, s_ls = local_search(a, max_iter=500)
        if s_ls > best_score:
            best_score = s_ls
            best_assignment = a_ls
    
    # Final local search
    best_assignment, best_score = local_search(best_assignment, max_iter=3000)
    
    # Convert to required format
    result = [[m, r] for m, r in best_assignment]
    return result
