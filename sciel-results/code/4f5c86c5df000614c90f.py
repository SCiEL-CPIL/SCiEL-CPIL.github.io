import time
import random
from fractions import Fraction
from itertools import combinations

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
    """
    Checks if S is reciprocal-subsum-free.
    Condition: For every a in S, 1/a != sum(1/b for b in T) for any subset T of S\{a}.
    """
    if not S:
        return False
    
    S = list(set(S))
    n = len(S)
    recips = {v: Fraction(1, v) for v in S}
    
    for a in S:
        target = recips[a]
        others = [recips[b] for b in S if b != a]
        
        m = len(others)
        if m == 0:
            continue
            
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

def _find_bad_cores(N=24):
    """
    Finds all minimal bad cores.
    A bad core is a set C such that if C is a subset of S, S is invalid.
    Specifically, C contains an element a and a subset T of C\{a} such that sum(1/b for b in T) == 1/a.
    """
    bad_cores = []
    all_elements = list(range(1, N + 1))
    
    for a in all_elements:
        target = Fraction(1, a)
        others = [v for v in all_elements if v != a]
        
        # MITM to find all subsets T of 'others' summing to 'target'
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        L_sums = {Fraction(0): 0} # sum -> mask
        # We need to track masks to reconstruct T
        # To save memory, we just store one mask per sum? 
        # No, we need ALL minimal T.
        # However, if T1 subset T2 and both sum to target, T1 is minimal, T2 is not.
        # So we just need minimal T.
        
        # Let's generate all subset sums for left and right
        L_dict = {}
        for mask in range(1 << len(left)):
            s = Fraction(0)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            if s <= target:
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
                
        found_Ts = set()
        for s_l, masks_l in L_dict.items():
            s_r_needed = target - s_l
            if s_r_needed in R_dict:
                for m_l in masks_l:
                    for m_r in R_dict[s_r_needed]:
                        # Reconstruct T
                        T = set()
                        for i in range(len(left)):
                            if m_l & (1 << i):
                                T.add(left[i])
                        for i in range(len(right)):
                            if m_r & (1 << i):
                                T.add(right[i])
                        
                        if T: # T cannot be empty because target > 0
                            found_Ts.add(frozenset(T))
                            
        # For each T, C = T U {a} is a bad core.
        # We need to keep only minimal bad cores.
        # If C1 subset C2, C2 is redundant.
        for T in found_Ts:
            C = frozenset(T | {a})
            # Check if C is subset of any existing bad core? 
            # Or if any existing bad core is subset of C?
            # If existing core C0 is subset of C, then C is redundant.
            # If C is subset of existing core C0, then C0 is redundant (replace C0 with C).
            # We will collect all and then minimize.
            bad_cores.append(C)
            
    # Minimize the set of bad cores
    # Sort by size
    bad_cores.sort(key=len)
    minimal_cores = []
    for C in bad_cores:
        is_redundant = False
        for MC in minimal_cores:
            if MC <= C:
                is_redundant = True
                break
        if not is_redundant:
            minimal_cores.append(C)
            
    return minimal_cores

def construct():
    _start_clock()
    
    N = 24
    all_elements = list(range(1, N + 1))
    
    # 1. Pre-compute bad cores
    bad_cores = _find_bad_cores(N)
    
    # Convert to bitmasks for fast intersection checks
    core_masks = []
    for C in bad_cores:
        mask = 0
        for v in C:
            mask |= (1 << (v - 1))
        core_masks.append(mask)
    
    # 2. Branch and Bound Search for Maximum Independent Set in Hypergraph
    # We want the largest subset S such that no core_mask is a subset of S.
    # i.e., for all c in core_masks, (S & c) != c.
    
    best_mask = 0
    best_size = 0
    
    # Heuristic ordering: 
    # Small numbers are in many cores. Including them restricts others heavily.
    # Large numbers are in few cores.
    # Let's try to include large numbers first? 
    # If we include large numbers, we leave small numbers as candidates.
    # But small numbers conflict with many things.
    # Let's sort candidates in descending order (24, 23, ..., 1).
    # This is often good for "packing" problems where large items are safer.
    
    # We use a recursive search.
    # To speed up, we can pre-compute for each vertex v, the list of core masks containing v.
    # When we add v to S, we only need to check those cores.
    
    cores_by_vertex = [[] for _ in range(N)]
    for idx, cm in enumerate(core_masks):
        for v in range(1, N + 1):
            if cm & (1 << (v - 1)):
                cores_by_vertex[v - 1].append(cm)
                
    def is_valid_extension(current_mask, new_v_idx):
        # Check if adding new_v_idx to current_mask creates a bad core
        new_mask = current_mask | (1 << new_v_idx)
        # Only check cores containing new_v_idx
        for cm in cores_by_vertex[new_v_idx]:
            if (new_mask & cm) == cm:
                return False
        return True

    def backtrack(candidates_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if _deadline_hit():
            return
            
        # Pruning: Upper bound
        # Max possible size = current_size + popcount(candidates_mask)
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Pick a vertex from candidates
        # Heuristic: Pick the vertex with the highest "degree" in the remaining hypergraph?
        # Or just the lowest index (smallest number) to prune early?
        # Let's try lowest index (smallest number) first.
        # Why? Because small numbers are in many cores. If we include a small number, we might invalidate many candidates immediately.
        # If we exclude it, we keep more candidates.
        # Usually, branching on the most constrained variable is better.
        
        v_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        v_val = v_idx + 1
        v_bit = 1 << v_idx
        
        # Option 1: Include v
        if is_valid_extension(current_mask, v_idx):
            new_candidates = candidates_mask & ~v_bit
            # Optional pruning: If adding v removes many candidates, it might be good.
            backtrack(new_candidates, current_mask | v_bit, current_size + 1)
            
        # Option 2: Exclude v
        backtrack(candidates_mask & ~v_bit, current_mask, current_size)
        
    # Start search
    initial_candidates = (1 << N) - 1
    backtrack(initial_candidates, 0, 0)
    
    # 3. Refinement (Local Search)
    # The B&B should find the global optimum. 
    # However, if the search space was pruned too aggressively or heuristic was bad, 
    # we can try to improve.
    
    current_set = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    
    # Try to swap elements to see if we can find a larger set? 
    # No, B&B maximizes size. If it found size K, no set of size K+1 exists.
    # So local search for size improvement is unnecessary if B&B is correct.
    # But we can use local search to find a *different* set of the same size if we want diversity, 
    # but the score is just the size.
    
    # Let's do a sanity check with the exact verifier
    if current_set and not _is_reciprocal_subsum_free(current_set):
        # Fallback to greedy if B&B produced an invalid set (should not happen)
        base_set = []
        for v in range(24, 0, -1):
            if _is_reciprocal_subsum_free([v] + base_set):
                base_set.append(v)
        current_set = base_set
        
    if not current_set:
        current_set = [24]
        
    return current_set

if __name__ == "__main__":
    print(construct())
