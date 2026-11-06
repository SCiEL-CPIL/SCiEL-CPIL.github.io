import time
import random
from fractions import Fraction
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

DEADLINE = 85.0
_START = None

def _start_clock():
    global _START
    _START = time.time()

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _is_reciprocal_subsum_free(S):
    if not S:
        return False
    S = list(set(S))
    n = len(S)
    recips = {v: Fraction(1, v) for v in S}
    for a in S:
        target = recips[a]
        others = [recips[b] for b in S if b != a]
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        L = {Fraction(0)}
        for v in left:
            L = L | {x + v for x in L}
        R = {Fraction(0)}
        for v in right:
            R = R | {x + v for x in R}
        R_set = set(R)
        for l in L:
            if (target - l) in R_set:
                return False
    return True

def _generate_conflicts_fast():
    N = 24
    # L = lcm(1..24)
    import math
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # W_b = L // b
    W = [0] * (N + 1)
    for b in range(1, N + 1):
        W[b] = L // b
        
    constraints = []
    indices = list(range(1, N + 1))
    
    for a in range(1, N + 1):
        if _deadline_hit():
            break
        target = W[a]
        others = [v for v in indices if v != a]
        
        # MITM for integer subset sum
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        # Generate sums and masks for left
        L_sums = {}
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += W[left[i]]
            if s <= target:
                if s not in L_sums:
                    L_sums[s] = []
                L_sums[s].append(mask)
                
        # Generate sums and masks for right
        R_sums = {}
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += W[right[i]]
            if s <= target:
                if s not in R_sums:
                    R_sums[s] = []
                R_sums[s].append(mask)
                
        # Combine
        found_Ts = []
        for s_l, masks_l in L_sums.items():
            s_r_needed = target - s_l
            if s_r_needed in R_sums:
                for m_l in masks_l:
                    for m_r in R_sums[s_r_needed]:
                        T = []
                        for i in range(len(left)):
                            if m_l & (1 << i):
                                T.append(left[i])
                        for i in range(len(right)):
                            if m_r & (1 << i):
                                T.append(right[i])
                        found_Ts.append(T)
                        
        # Filter minimal subsets
        # A subset T is minimal if no proper subset T' of T is in found_Ts
        # Since N is small, we can just check all
        # To speed up, sort by length
        found_Ts.sort(key=len)
        minimal_Ts = []
        for T in found_Ts:
            # Check if any T' in minimal_Ts is a subset of T
            is_minimal = True
            for T_min in minimal_Ts:
                if set(T_min).issubset(set(T)):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_Ts.append(T)
                
        # Add constraints for minimal T
        for T in minimal_Ts:
            if not T: continue
            row = [0] * N
            row[a-1] = 1
            for b in T:
                row[b-1] = 1
            constraints.append((row, len(T)))
            
    return constraints

def construct():
    _start_clock()
    
    # Try ILP approach with fast conflict generation
    try:
        if not _deadline_hit():
            constraints = _generate_conflicts_fast()
            if not _deadline_hit():
                c = np.zeros(24)
                for i in range(24):
                    c[i] = -1 # Maximize sum x_i -> Minimize -sum x_i
                    
                if constraints:
                    A_ub = np.array([c[0] for c in constraints])
                    b_ub = np.array([c[1] for c in constraints])
                    lin_cons = LinearConstraint(A_ub, ub=b_ub)
                else:
                    lin_cons = LinearConstraint(np.zeros((0, 24)), ub=np.zeros(0))
                    
                res = milp(
                    c=c,
                    constraints=lin_cons,
                    integrality=np.ones(24),
                    bounds=Bounds(lb=0, ub=1)
                )
                
                if res.success:
                    x = res.x
                    selected = [i+1 for i in range(24) if x[i] > 0.5]
                    if _is_reciprocal_subsum_free(selected):
                        return selected
    except Exception:
        pass

    # Fallback: Heuristic
    base_set = []
    for v in range(24, 0, -1):
        if _is_reciprocal_subsum_free([v] + base_set):
            base_set.append(v)
            
    best_set = set(base_set)
    
    rng = random.Random(42)
    all_numbers = list(range(1, 25))
    
    for i in range(50):
        if _deadline_hit():
            break
        perm = all_numbers[:]
        rng.shuffle(perm)
        
        current_set = []
        for v in perm:
            trial_set = current_set + [v]
            if _is_reciprocal_subsum_free(trial_set):
                current_set.append(v)
                
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    remaining = [v for v in all_numbers if v not in best_set]
    remaining.sort(reverse=True)
    
    final_set = list(best_set)
    for v in remaining:
        if _deadline_hit():
            break
        trial = final_set + [v]
        if _is_reciprocal_subsum_free(trial):
            final_set.append(v)
            
    best_set = set(final_set)
    
    if not best_set:
        best_set = {24}
        
    return list(best_set)

if __name__ == "__main__":
    print(construct())
