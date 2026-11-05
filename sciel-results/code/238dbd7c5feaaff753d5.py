import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict adjacency: adj[i] = set of j where lcm(nums[i], nums[j]) <= 256
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                adj[i].add(j)
                adj[j].add(i)
    
    # Try MIP first for a strong initial solution
    best_set = set()
    best_val = 0.0
    
    if time.time() < DEADLINE - 30:  # Reserve time for local search
        try:
            c = np.array([-1.0 / nums[i] for i in range(n)])
            
            # Build constraint matrix
            conflict_pairs = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        conflict_pairs.append((i, j))
            
            m = len(conflict_pairs)
            if m > 0:
                A = np.zeros((m, n))
                for k, (i, j) in enumerate(conflict_pairs):
                    A[k, i] = 1.0
                    A[k, j] = 1.0
                lb = np.zeros(m)
                ub = np.ones(m)
            else:
                A = np.zeros((1, n))
                lb = np.zeros(1)
                ub = np.zeros(1)
            
            bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
            constraints = LinearConstraint(A, lb=lb, ub=ub)
            integrality = np.ones(n)
            
            result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if result is not None and result.success and result.x is not None:
                best_set = set(i for i in range(n) if result.x[i] > 0.5)
                best_val = sum(1.0 / nums[i] for i in best_set)
        except Exception:
            pass
    
    # Fallback greedy if MIP didn't give a good solution
    if not best_set or best_val < 0.1:
        order = sorted(range(n), key=lambda i: -1.0/nums[i])
        for i in order:
            if all(j not in best_set for j in adj[i]):
                best_set.add(i)
        best_val = sum(1.0 / nums[i] for i in best_set)
    
    # Local search refinement
    def is_valid_addition(s, i):
        return all(j not in s for j in adj[i])
    
    def compute_val(s):
        return sum(1.0 / nums[i] for i in s)
    
    improved = True
    max_rounds = 50
    rounds = 0
    while improved and rounds < max_rounds and time.time() < DEADLINE - 2:
        improved = False
        rounds += 1
        
        # Phase 1: Try adding elements
        for i in range(n):
            if i not in best_set and is_valid_addition(best_set, i):
                best_set.add(i)
                best_val += 1.0 / nums[i]
                improved = True
        
        # Phase 2: Try 1-swap moves (remove one, add one)
        if not improved:
            for i in list(best_set):
                for j in range(n):
                    if j in best_set:
                        continue
                    # Check if removing i and adding j is valid
                    # j must not conflict with (best_set - {i})
                    if all(k != j for k in adj[i]) or True:  # placeholder
                        # More efficient: j must not conflict with any element in best_set except i
                        valid = True
                        for k in best_set:
                            if k != i and j in adj[k]:
                                valid = False
                                break
                        if valid:
                            new_val = best_val - 1.0/nums[i] + 1.0/nums[j]
                            if new_val > best_val + 1e-12:
                                best_set.remove(i)
                                best_set.add(j)
                                best_val = new_val
                                improved = True
                                break
                if improved:
                    break
        
        # Phase 3: Try 2-swap moves (remove two, add two) - only if time permits
        if not improved and time.time() < DEADLINE - 5:
            found = False
            for i in list(best_set):
                if found:
                    break
                for i2 in list(best_set):
                    if i2 <= i or found:
                        continue
                    for j in range(n):
                        if found or j in best_set:
                            continue
                        for j2 in range(j+1, n):
                            if found or j2 in best_set:
                                continue
                            # Check validity: j and j2 must not conflict with (best_set - {i, i2})
                            # and j and j2 must not conflict with each other
                            valid = True
                            for k in best_set:
                                if k == i or k == i2:
                                    continue
                                if j in adj[k] or j2 in adj[k]:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            if j in adj[j2]:
                                continue
                            new_val = best_val - 1.0/nums[i] - 1.0/nums[i2] + 1.0/nums[j] + 1.0/nums[j2]
                            if new_val > best_val + 1e-12:
                                best_set.discard(i)
                                best_set.discard(i2)
                                best_set.add(j)
                                best_set.add(j2)
                                best_val = new_val
                                improved = True
                                found = True
                                break
    
    # Final verification
    result_list = sorted([nums[i] for i in best_set])
    for idx_a in range(len(result_list)):
        for idx_b in range(idx_a + 1, len(result_list)):
            a, b = result_list[idx_a], result_list[idx_b]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                # Invalid - fall back to greedy
                order = sorted(range(n), key=lambda i: -1.0/nums[i])
                selected = set()
                for i in order:
                    if all(j not in selected for j in adj[i]):
                        selected.add(i)
                result_list = sorted([nums[i] for i in selected])
                break
    
    if not result_list:
        result_list = [2]
    
    return result_list
