import time
import sys
import random
import numpy as np

# Set recursion limit just in case, though we are using iterative approaches
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 55.0  # Safety margin for 60s limit

def _precompute_forbidden():
    """
    Precompute for each integer d in 1..192, the list of triples (i, j, k) 
    such that (i+j+k) % d == 0.
    Actually, for efficiency in the check "does d divide sum of 3 OTHERS",
    we need to know if there exist x, y, z in S \ {d} such that (x+y+z) % d == 0.
    
    Instead of storing triples, we can store for each d, a bitset of numbers 
    that are "compatible" in a specific sense? No, the sum depends on 3 numbers.
    
    A better precomputation for fast checking:
    For a fixed candidate d and a current set S (bitset), we want to check:
    1. Does d divide sum of any 3 distinct elements in S?
    2. For each a in S, does a divide (d + x + y) for distinct x, y in S \ {a, d}?
    
    Let's optimize the check for a new candidate c against existing set S (represented as a set of indices or bitset).
    
    Check 1: c divides sum of 3 distinct in S.
    We can precompute for each c, the set of residues mod c? 
    No, S changes.
    
    Let's stick to the bitset approach for the "other" constraints which are pairwise-ish.
    Constraint 2: For a in S, a | (c + x + y) <=> (x + y) % a == (-c) % a.
    This is a pair sum check modulo a.
    
    If we use a bitset for S, checking pair sums modulo a is still $O(N^2)$ if done naively.
    However, N is small (<=48). $48^2 = 2304$. 
    If we do this for each a in S (48 times), it's $48 * 2304 \approx 110k$ ops.
    With 192 candidates, total ops ~ $192 * 110k \approx 21M$. 
    Python is slow. 21M ops might take 1-2 seconds. 
    This is fast enough for a single greedy pass.
    
    But we can do better.
    Let's precompute `pair_sums_mod[a][r]`? No, a changes.
    
    Alternative: 
    For a fixed set S, we can maintain a list of elements.
    When adding c:
    1. Check c vs S: 
       For each triple in S, check sum % c == 0. 
       Number of triples in S of size k is k choose 3. For k=48, ~18k triples.
       18k checks per candidate. 192 candidates -> 3.5M checks. Very fast.
       
    2. Check S vs c:
       For each a in S, check if exists x, y in S \ {a, c} such that (c + x + y) % a == 0.
       This is equivalent to (x + y) % a == (-c) % a.
       For a fixed a, we can iterate x in S \ {a, c} and check if (target - x) % a is in S \ {a, c, x}.
       To make this O(N), we can maintain a set of residues for each a? 
       But a is the modulus, so residues are in 0..a-1.
       
       Let's just use the O(N^2) per a check, but optimized with early exit.
       Total cost per candidate c: 
       Check 1: O(|S|^3) ~ 18k
       Check 2: O(|S|^3) ~ 18k (sum over a of |S|^2/2)
       Total ~ 36k ops per candidate.
       192 candidates * 36k = 7M ops.
       In Python, 7M simple integer ops takes ~0.5-1s.
       This is perfectly feasible for 60s. We can even do multiple restarts.
    """
    pass

def _is_valid_addition(c, S):
    """
    Check if c can be added to S.
    S is a list of distinct integers.
    c is an integer not in S.
    Returns True if valid.
    """
    n = len(S)
    if n < 3:
        return True
        
    # 1. c must not divide sum of any 3 distinct elements in S
    # Iterate triples in S
    # Optimization: If c > max_sum_possible, it can't divide. 
    # Max sum of 3 from S <= 192*3 = 576.
    # If c > 576, always true. But c <= 192.
    # If c is large, multiples are sparse.
    
    # Let's just iterate.
    for i in range(n):
        for j in range(i + 1, n):
            s_ij = S[i] + S[j]
            # We need S[k] such that (s_ij + S[k]) % c == 0
            # So S[k] % c == (-s_ij) % c
            # We can precompute residues of S mod c? 
            # c changes every time. 
            # So we just loop k.
            for k in range(j + 1, n):
                if (s_ij + S[k]) % c == 0:
                    return False
                    
    # 2. For each a in S, a must not divide (c + x + y) for distinct x, y in S \ {a}
    # (c + x + y) % a == 0  =>  (x + y) % a == (-c) % a
    
    for idx_a, a in enumerate(S):
        target = (-c) % a
        # We need x, y in S \ {a} such that (x + y) % a == target
        # And x != y.
        
        # Optimization: 
        # Create a set of residues of S \ {a} mod a?
        # No, we need to ensure x and y are distinct elements.
        # Let's iterate x in S \ {a}.
        # For each x, we need y in S \ {a, x} such that y % a == (target - x) % a.
        
        # To speed up, let's build a list of (value, residue) for S \ {a}.
        # But we can just do double loop with early break.
        # Since n is small, this is fine.
        
        found_x = False
        for i in range(n):
            if i == idx_a:
                continue
            x = S[i]
            needed_res = (target - x) % a
            
            for j in range(n):
                if j == idx_a or j == i:
                    continue
                y = S[j]
                if y % a == needed_res:
                    return False
        # If we finish loops, no violation for this a.
        
    return True

def _greedy_construct(order):
    """
    Construct a set by iterating through 'order' and adding if valid.
    """
    S = []
    for c in order:
        if _is_valid_addition(c, S):
            S.append(c)
    return S

def _local_search(S, time_limit):
    """
    Try to improve S by swapping elements.
    If we remove a and add b, is it valid?
    We can try to replace the "most constraining" element with a "less constraining" one.
    Or just random swaps.
    """
    current_set = S[:]
    best_set = current_set[:]
    start = time.time()
    
    # Precompute a list of all candidates not in current_set
    all_nums = set(range(1, 193))
    
    while time.time() - start < time_limit:
        # Try to swap one element out and one in
        # Pick a random element from current_set to remove
        if not current_set:
            break
        idx_out = random.randint(0, len(current_set) - 1)
        elem_out = current_set[idx_out]
        
        # Try to find an element to add
        # Candidates: all_nums - set(current_set)
        # To be efficient, we can just iterate through a shuffled list of candidates
        candidates = [x for x in range(192, 0, -1) if x not in current_set]
        random.shuffle(candidates)
        
        improved = False
        for elem_in in candidates:
            # Check if swapping is valid
            # New set = current_set - {elem_out} + {elem_in}
            # We need to check validity of the new set.
            # This is expensive: O(N^3).
            # But we can check incrementally?
            # 1. elem_in must be valid against current_set \ {elem_out}
            # 2. All elements in current_set \ {elem_out} must be valid against elem_in
            # 3. All pairs in current_set \ {elem_out} must be valid against each other (already valid)
            
            # Let's just construct the new set and check full validity? 
            # No, full validity check is O(N^3) ~ 18k ops. 
            # If we do this for 100 swaps, 1.8M ops. Fast.
            
            new_set = [x for i, x in enumerate(current_set) if i != idx_out]
            new_set.append(elem_in)
            
            # Check if new_set is valid
            # We can reuse _is_valid_addition logic but for the whole set?
            # Actually, we just need to check if the swap introduces a violation.
            # Violations can only involve elem_in or elem_out.
            # Since elem_out is removed, any violation involving only elem_out is gone.
            # New violations can only involve elem_in.
            # So we just need to check if elem_in is valid against new_set \ {elem_in} (which is current_set \ {elem_out}).
            # AND check if any element in current_set \ {elem_out} is now invalid because of elem_in?
            # The condition is symmetric: "a does not divide sum of 3 others".
            # If we add elem_in, we must check:
            # 1. elem_in does not divide sum of any 3 in current_set \ {elem_out}
            # 2. For each a in current_set \ {elem_out}, a does not divide (elem_in + x + y) for x, y in current_set \ {elem_out, a}
            
            if _is_valid_addition(elem_in, current_set[:idx_out] + current_set[idx_out+1:]):
                current_set = new_set
                improved = True
                break
        
        if not improved:
            # Try to add an element without removing? 
            # If len(current_set) < 48, we can try adding.
            if len(current_set) < 48:
                candidates = [x for x in range(192, 0, -1) if x not in current_set]
                random.shuffle(candidates)
                for elem_in in candidates:
                    if _is_valid_addition(elem_in, current_set):
                        current_set.append(elem_in)
                        improved = True
                        break
            
            if not improved:
                # Stuck. Try a random restart or accept?
                # If we haven't found a better set, we might be in a local minimum.
                # For simplicity, we just stop if no improvement in a while.
                # Or we can try a different idx_out.
                pass

    return current_set

def construct():
    start_time = time.time()
    
    # Strategy: 
    # 1. Try greedy with descending order (192..1). Large numbers are less likely to be divisors.
    # 2. Try greedy with ascending order (1..192). Small numbers are more restrictive, might prune early? 
    #    Actually, including small numbers makes it harder to add others. 
    #    So descending is usually better for maximizing size.
    # 3. Use local search to refine.
    
    best_set = []
    
    # Greedy 1: Descending
    s1 = _greedy_construct(range(192, 0, -1))
    if len(s1) > len(best_set):
        best_set = s1
        
    # Greedy 2: Ascending
    s2 = _greedy_construct(range(1, 193))
    if len(s2) > len(best_set):
        best_set = s2
        
    # Greedy 3: Randomized restarts
    # We have ~50s left.
    # Each greedy pass is ~0.1s.
    # Each local search step is ~0.01s.
    # We can do many restarts.
    
    num_restarts = 20
    time_per_restart = 2.0
    
    for i in range(num_restarts):
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        # Random order
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        s = _greedy_construct(candidates)
        
        # Local search
        s_improved = _local_search(s, time_per_restart)
        
        if len(s_improved) > len(best_set):
            best_set = s_improved
            
    return best_set

# EVOLVE-BLOCK-END
