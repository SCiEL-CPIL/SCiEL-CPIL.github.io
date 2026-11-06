import time
import random
from fractions import Fraction
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE = 85.0  # Safety margin under 90s
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
    """
    Checks if S is reciprocal-subsum-free.
    Condition: For every a in S, 1/a != sum(1/b for b in T) for any subset T of S\{a}.
    """
    if not S:
        return False
    
    S = list(set(S))
    n = len(S)
    recips = {v: Fraction(1, v) for v in S}
    
    # For each element a in S, check if 1/a is a subset sum of reciprocals of other elements
    for a in S:
        target = recips[a]
        others = [recips[b] for b in S if b != a]
        
        # Meet-in-the-middle subset sum check for small sets
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        L = {Fraction(0)}
        for v in left:
            L = L | {x + v for x in L}
        
        R = {Fraction(0)}
        for v in right:
            R = R | {x + v for x in R}
            
        # Check if target - l == r for some l in L, r in R
        R_set = set(R)
        for l in L:
            if (target - l) in R_set:
                return False
    return True

def _generate_conflicts():
    """
    Generates all linear constraints of the form x_a + sum(x_b for b in T) <= |T|
    where 1/a == sum(1/b for b in T).
    Returns a list of tuples (a, T, |T|) or directly the constraint rows for milp.
    """
    N = 24
    conflicts = []
    # We iterate over all possible targets 'a'
    for a in range(1, N + 1):
        target = Fraction(1, a)
        # Candidates for T are subsets of {1..24} \ {a}
        # To be efficient, we can prune.
        # Sum of reciprocals of any subset of {2..24} is bounded.
        # 1/2 + 1/3 + ... + 1/24 < 1. 1/1 = 1.
        # So a=1 can only be formed by T such that sum(1/b)=1.
        # For a > 1, target < 1.
        
        # We need to find all subsets T of [1..24]\{a} such that sum(1/b for b in T) == 1/a.
        # This is a subset sum problem on rationals.
        # Since N=24 is small, we can use MITM to find all such T efficiently for each a?
        # Or just iterate all subsets of size k?
        # Number of subsets is 2^23 ~ 8 million. For each a, checking all is too slow.
        
        # Better: Generate all subset sums of reciprocals for the set {1..24}.
        # Map sum -> list of subsets (or just one representative? No, we need ALL conflicts).
        # Actually, we just need to know if a specific sum is achievable by a specific set T.
        # The constraint is specific to the set T.
        
        # Let's precompute all possible subset sums and the sets that achieve them.
        # There are 2^24 subsets. This is 16 million.
        # We can do this once.
        pass
        
    # Given the time limit, generating ALL conflicts exactly might be heavy if done naively.
    # However, the number of distinct reciprocal sums is limited.
    # Let's try a different approach:
    # Iterate over all subsets T of {1..24}.
    # Compute sum_recip = sum(1/b for b in T).
    # Check if sum_recip == 1/a for some a not in T.
    # If so, add constraint.
    
    # 2^24 is 16,777,216.
    # In Python, iterating this many times with Fraction arithmetic might be slow (seconds to minutes).
    # We need to be smarter or use the ILP solver's capability? No, we must build the matrix.
    
    # Optimization:
    # Many subsets will have sums > 1 or < 1/24.
    # Max sum is sum(1/1..1/24) ~ 3.7.
    # Min non-zero sum is 1/24.
    # We only care about sums that equal 1/a for a in 1..24.
    # There are only 24 possible target values: 1, 1/2, ..., 1/24.
    
    # We can use MITM to find all subsets T that sum to each target 1/a.
    # For each a in 1..24:
    #   Target = 1/a.
    #   Find all subsets T of {1..24}\{a} summing to Target.
    #   For each such T, add constraint x_a + sum(x_b for b in T) <= |T|.
    
    # How to find all subsets T summing to Target?
    # MITM: Split {1..24}\{a} into L and R.
    # Compute all subset sums of L and R.
    # For each sum_l in L_sums, check if Target - sum_l is in R_sums.
    # If yes, reconstruct T.
    # This is efficient for N=23. 2^11 = 2048.
    
    constraints = []
    indices = list(range(1, 25))
    
    for a in range(1, 25):
        if _deadline_hit():
            break
        target = Fraction(1, a)
        others = [v for v in indices if v != a]
        
        # MITM to find all subsets of 'others' summing to 'target'
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        # Generate all subset sums for left part: (sum, mask)
        L_sums = {}
        # We need to track the actual elements to form the constraint.
        # Storing masks is good.
        # But we need to combine left mask and right mask.
        
        # To keep memory and time low, we just store sums and one representative mask?
        # No, we need ALL subsets T because each T generates a different constraint.
        # However, if two subsets T1 and T2 have the same sum and same size, do they generate different constraints?
        # Constraint: x_a + sum_{b in T} x_b <= |T|.
        # If T1 != T2 but |T1| == |T2| and sum(1/b) is same, the constraints are different because the variables x_b are different.
        # So we must enumerate all T.
        
        # Is the number of such T small?
        # For 1/2: {1} (but 1 is a, so a!=1 for this target? No, a=2, target=1/2. Others don't include 2. {1} is in others? Yes. 1/1=1 != 1/2. Wait.
        # If a=2, target=1/2. Others = {1, 3..24}.
        # 1/1 = 1. Too big.
        # 1/3 + 1/6 = 1/2. So T={3,6} is a solution.
        # 1/4 + 1/4 (not distinct).
        # 1/3 + 1/12 + 1/6? 1/3+1/6=1/2.
        # The number of Egyptian fraction representations is finite and usually small for small denominators.
        
        L_dict = {}
        for mask in range(1 << len(left)):
            s = Fraction(0)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            if s <= target: # Prune
                if s not in L_dict:
                    L_dict[s] = []
                L_dict[s].append(mask)
                
        R_dict = {}
        for mask in range(1 << len(right)):
            s = Fraction(0)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if s <= target:
                if s not in R_dict:
                    R_dict[s] = []
                R_dict[s].append(mask)
                
        # Combine
        found_Ts = []
        for s_l, masks_l in L_dict.items():
            s_r_needed = target - s_l
            if s_r_needed in R_dict:
                for m_l in masks_l:
                    for m_r in R_dict[s_r_needed]:
                        # Reconstruct T
                        T = []
                        for i in range(len(left)):
                            if m_l & (1 << i):
                                T.append(left[i])
                        for i in range(len(right)):
                            if m_r & (1 << i):
                                T.append(right[i])
                        found_Ts.append(T)
                        
        # Add constraints for each T
        # Constraint: x_a + sum(x_b for b in T) <= |T|
        # Variables are indexed 0..23 corresponding to values 1..24.
        # x_a is at index a-1.
        # x_b is at index b-1.
        
        for T in found_Ts:
            if not T: continue
            row = [0] * 24
            row[a-1] = 1
            for b in T:
                row[b-1] = 1
            constraints.append((row, len(T)))
            
    return constraints

def construct():
    _start_clock()
    
    # Try ILP approach
    try:
        if not _deadline_hit():
            constraints = _generate_conflicts()
            if not _deadline_hit():
                c = np.zeros(24)
                for i in range(24):
                    c[i] = -1 # Maximize sum x_i -> Minimize -sum x_i
                    
                A_ub = np.array([c[0] for c in constraints] if constraints else [])
                b_ub = np.array([c[1] for c in constraints] if constraints else [])
                
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
                    # Verify result
                    if _is_reciprocal_subsum_free(selected):
                        return selected
                    else:
                        # Fallback if ILP gave invalid result (shouldn't happen with exact constraints, but safety)
                        pass
                else:
                    # ILP failed, fallback to heuristic
                    pass
    except Exception:
        pass

    # Fallback: Original heuristic logic
    # 1. Generate a strong baseline using greedy descending order
    base_set = []
    recips_in_set = []
    for v in range(24, 0, -1):
        if _is_reciprocal_subsum_free([v] + base_set):
            base_set.append(v)
            recips_in_set.append(Fraction(1, v))
    
    best_set = set(base_set)
    
    # 2. Local Search / Swap Optimization
    rng = random.Random(42)
    all_numbers = list(range(1, 25))
    
    # Multi-start Greedy
    for i in range(50):
        if _deadline_hit():
            break
            
        perm = all_numbers[:]
        rng.shuffle(perm)
        
        current_set = []
        for v in perm:
            trial_set = current_set + [v]
            if current_set:
                current_recips = [Fraction(1, u) for u in current_set]
                target = Fraction(1, v)
                m = len(current_recips)
                mid = m // 2
                L, R = current_recips[:mid], current_recips[mid:]
                LS = {Fraction(0)}
                for val in L:
                    LS = LS | {x + val for x in LS}
                RS = {Fraction(0)}
                for val in R:
                    RS = RS | {x + val for x in RS}
                RS_set = set(RS)
                valid_add = True
                for l in LS:
                    if (target - l) in RS_set:
                        valid_add = False
                        break
                if not valid_add:
                    continue
            
            if _is_reciprocal_subsum_free(trial_set):
                current_set.append(v)
                
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    # 3. Refinement
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
