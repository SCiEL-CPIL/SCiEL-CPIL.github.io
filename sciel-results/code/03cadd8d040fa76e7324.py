import time
import math
import random
import sys
from collections import defaultdict

N = 128
sys.setrecursionlimit(10000)

# Precompute conflicts: for each x, conflicts[x] is a set of (u,v) pairs
# such that {x, u, v} is a forbidden triple (all three pairwise LCMs equal).
conflicts = [set() for _ in range(N + 1)]

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

precompute_conflicts()

def can_add(x, current_set):
    """Check if x can be added to current_set without creating a forbidden triple."""
    for u, v in conflicts[x]:
        if u in current_set and v in current_set:
            return False
    return True

# Global pair conflict matrix
# pair_conflict[i][j] = True if there exists k such that {i,j,k} is a forbidden triple.
pair_conflict = [[False] * (N + 1) for _ in range(N + 1)]

def precompute_pair_conflicts():
    for a in range(1, N + 1):
        for (b, c) in conflicts[a]:
            pair_conflict[a][b] = True
            pair_conflict[a][c] = True
            pair_conflict[b][c] = True
            pair_conflict[b][a] = True
            pair_conflict[c][a] = True
            pair_conflict[c][b] = True

precompute_pair_conflicts()

def greedy_construct():
    """Greedy construction with random restarts."""
    best_s = set()
    for _ in range(300):
        order = list(range(1, N + 1))
        random.shuffle(order)
        cur = set()
        for x in order:
            if can_add(x, cur):
                cur.add(x)
        if len(cur) > len(best_s):
            best_s = cur
    return best_s

def local_search(s, deadline):
    """
    Improve set s using simulated annealing-style local search.
    """
    s = set(s)
    best_s = set(s)
    best_size = len(s)
    current_temp = 1.0
    min_temp = 0.01
    
    while time.time() < deadline:
        improved = False
        # Try to add any element
        for x in range(1, N + 1):
            if x not in s and can_add(x, s):
                s.add(x)
                improved = True
                if len(s) > best_size:
                    best_size = len(s)
                    best_s = set(s)
                break
        if improved:
            continue
        
        # If no element can be added, try removing one element to allow others to be added
        candidates_to_remove = list(s)
        random.shuffle(candidates_to_remove)
        
        found_improvement = False
        for x in candidates_to_remove:
            s_new = s - {x}
            # Try to add elements to s_new
            for y in range(1, N + 1):
                if y not in s_new and can_add(y, s_new):
                    s_new.add(y)
                    if len(s_new) > best_size:
                        best_size = len(s_new)
                        best_s = set(s_new)
                        found_improvement = True
                        break
            if found_improvement:
                s = s_new
                break
        
        if not found_improvement:
            # No improvement found, decrease temperature
            current_temp *= 0.95
            if current_temp < min_temp:
                break
    
    return best_s

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Greedy + Local Search to get a good lower bound
    initial_set = greedy_construct()
    initial_set = local_search(initial_set, deadline)
    
    best_size = len(initial_set)
    best_set = initial_set
    
    # 2. Try more local search iterations with different starting points
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            # Create a perturbed version of best_set
            perturbed = set(best_set)
            # Randomly remove 1-3 elements
            to_remove = random.sample(list(perturbed), min(random.randint(1, 3), len(perturbed)))
            for x in to_remove:
                perturbed.discard(x)
            
            # Try to add back elements
            for x in range(1, N + 1):
                if x not in perturbed and can_add(x, perturbed):
                    perturbed.add(x)
            
            # Local search from perturbed state
            improved = local_search(perturbed, deadline - 5.0)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    result = list(best_set)
    if not result:
        result = [1]
        
    return result

# Test locally
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            for k in range(j + 1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if math.lcm(a, b) == math.lcm(a, c) == math.lcm(b, c):
                    valid = False
                    print(f"Invalid triple: {a}, {b}, {c}")
    print(f"Valid: {valid}")
