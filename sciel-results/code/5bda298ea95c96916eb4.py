import time
import random
import numpy as np
from collections import deque

N = 12
FULL_SET_MASK = (1 << N) - 1

def _compute_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Uses bitmask representation for subsets.
    Returns -1 if not synchronizing.
    """
    # Precompute transitions for all possible subsets? 
    # No, 2^12 = 4096 subsets. We can precompute the transition table for the subset automaton.
    # trans_subset[subset_mask][letter] -> next_subset_mask
    
    # Actually, doing it on the fly with bit operations is fast enough, 
    # but precomputing the mapping for the specific automaton is faster if we do many searches?
    # No, the automaton changes. 
    # However, we can precompute the effect of trans_a and trans_b on any subset mask.
    
    # Let's precompute the transition for each letter for all 2^N subsets.
    # This takes 2 * 2^N operations. 2 * 4096 = 8192 ops. Very fast.
    # Then BFS is just array lookups.
    
    # trans_a_mask[i] is the bitmask of the image of state i under a? No.
    # Let's define:
    # img_a[s] = trans_a[s]
    # img_b[s] = trans_b[s]
    
    # For a subset S (mask), the image under a is:
    # OR of (1 << trans_a[s]) for all s in S.
    
    # We can precompute this for all 4096 masks for the given trans_a and trans_b.
    # table_a[mask] -> next_mask
    # table_b[mask] -> next_mask
    
    table_a = np.zeros(1 << N, dtype=np.int32)
    table_b = np.zeros(1 << N, dtype=np.int32)
    
    # Optimization: We can compute table for all masks at once?
    # For each state s, it contributes to masks that contain s.
    # This is a zeta transform-like operation but for bitwise OR.
    # Actually, simple iteration is fine for 4096 items.
    
    for mask in range(1 << N):
        m_a = 0
        m_b = 0
        # Iterate through bits set in mask
        # Faster: iterate through all states and check if bit is set
        # Or just use the property:
        # image(S) = union_{s in S} {f(s)}
        
        # Since N=12, we can just do:
        # for s in range(N):
        #    if mask & (1 << s):
        #       m_a |= 1 << trans_a[s]
        #       m_b |= 1 << trans_b[s]
        
        # Let's try a vectorized approach for the precomputation step?
        # It's only 4096 iterations, so Python loop is ~1ms.
        
        m_a = 0
        m_b = 0
        for s in range(N):
            if mask & (1 << s):
                m_a |= 1 << trans_a[s]
                m_b |= 1 << trans_b[s]
        table_a[mask] = m_a
        table_b[mask] = m_b
        
    # BFS
    # Start from FULL_SET_MASK
    # Target: any singleton mask (1 << i)
    
    visited = np.zeros(1 << N, dtype=np.bool_)
    dist = np.full(1 << N, -1, dtype=np.int32)
    
    queue = deque()
    queue.append(FULL_SET_MASK)
    visited[FULL_SET_MASK] = True
    dist[FULL_SET_MASK] = 0
    
    while queue:
        cur_mask = queue.popleft()
        cur_dist = dist[cur_mask]
        
        # Check if singleton
        # A mask is a singleton if it has exactly one bit set.
        # mask & (mask - 1) == 0 and mask != 0
        if cur_mask != 0 and (cur_mask & (cur_mask - 1)) == 0:
            return cur_dist
            
        # Expand
        next_a = table_a[cur_mask]
        if not visited[next_a]:
            visited[next_a] = True
            dist[next_a] = cur_dist + 1
            queue.append(next_a)
            
        next_b = table_b[cur_mask]
        if not visited[next_b]:
            visited[next_b] = True
            dist[next_b] = cur_dist + 1
            queue.append(next_b)
            
    return -1

def construct():
    """
    Search for a synchronizing automaton with N=12 states that maximizes
    the shortest reset word length.
    Strategy: Structured search over "Cycle + Shift-with-Defect" family.
    This family is known to produce high thresholds.
    """
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    best_score = -1
    best_trans_a = None
    best_trans_b = None
    
    rng = random.Random(42)
    
    # 1. Baseline: Standard Cerny-like construction
    # a: cycle (0->1->...->11->0)
    # b: shift down (1->0, 2->1, ..., 0->0)
    # This has threshold N-1 = 11.
    
    base_a = list(range(1, N)) + [0]
    base_b = [0] + list(range(N-1))
    
    score = _compute_threshold(base_a, base_b)
    if score > best_score:
        best_score = score
        best_trans_a = base_a
        best_trans_b = base_b
        
    # 2. Structured Search
    # We vary:
    # - The cycle length/structure of 'a' (mostly keep it a single cycle, but permute the labels)
    # - The "defect" in 'b'. 
    #   Standard b: b(i) = i-1 mod N, except b(0)=0.
    #   Variations: 
    #     - Change where the "break" is (which state maps to itself or breaks the cycle)
    #     - Change the shift amount
    #     - Introduce random "jumps" in b while keeping it "mostly" a shift
    
    # Let's systematically explore the space of "almost permutations" for b.
    # A high threshold requires that b doesn't collapse the set too quickly.
    # The "bottleneck" is usually the time needed to align states so that b can map them to a smaller set.
    
    # Parameter 1: Shift amount for b (1 to N-1)
    # Parameter 2: The "fixed point" or "defect" state in b.
    # Parameter 3: Random permutation of states for a (equivalent to renaming states).
    
    # Actually, renaming states doesn't change the threshold.
    # So we can fix a = [1, 2, ..., N-1, 0] (the standard cycle).
    # And vary b.
    
    # If we fix a, we only need to search over b.
    # b is a function from N to N.
    # To maximize threshold, b should be "close" to a permutation (to avoid immediate collapse)
    # but must have at least one "defect" to allow synchronization.
    
    # Let's iterate over all possible "defect" positions and destinations for b,
    # starting from the base shift.
    
    # Base b: b[i] = (i - 1) % N, with b[0] = 0.
    # This is a permutation except for 0.
    
    # Let's try all single modifications to the base b.
    # Modification: Change b[x] to y.
    
    # We also need to consider that 'a' might not be the standard cycle.
    # But since the threshold is invariant under state relabeling, 
    # any automaton is isomorphic to one where a is the standard cycle?
    # No, only if a is a single cycle. If a is not a single cycle, the structure is different.
    # However, known high-threshold automata usually have a as a single cycle.
    
    # So, let's assume a is a single cycle. We can fix a = [1, ..., N-1, 0].
    # Then we search for b.
    
    # Search Strategy for b:
    # 1. Start with base_b.
    # 2. Try all single changes: for x in 0..N-1, for y in 0..N-1: b[x]=y.
    # 3. Try all double changes? That's 12*12 * 12*12 = ~20k candidates.
    #    With the fast BFS, we can do many more.
    
    # Let's do a randomized search around the base structure, but with more "defects".
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate a candidate b
        # Start with a shift
        shift = rng.randint(1, N - 1)
        b = [(i - shift) % N for i in range(N)]
        
        # Introduce 1 to 3 defects
        num_defects = rng.randint(1, 3)
        for _ in range(num_defects):
            idx = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            b[idx] = new_dest
            
        # Ensure it's not a permutation? 
        # If it's a permutation, it might not synchronize (if a and b generate a group that doesn't collapse).
        # Actually, if both are permutations, the automaton is a group action and never synchronizes (unless N=1).
        # So we MUST ensure b is not a permutation.
        # The random defect usually ensures this, but let's check.
        if len(set(b)) == N:
            # It's a permutation. Force a defect.
            b[0] = b[1] # Create a collision
            
        candidate_a = list(range(1, N)) + [0]
        
        score = _compute_threshold(candidate_a, b)
        
        if score > best_score:
            best_score = score
            best_trans_a = candidate_a
            best_trans_b = b
            
            # If we find a very high score, we might want to explore its neighborhood more densely
            # But for now, random restarts with the "defect" model are good.
            
        # Additional Strategy: Try "Pavlov" type structures
        # a: cycle
        # b: maps everything to 0, except one state maps to 1?
        # This usually gives low scores.
        
        # Let's also try randomizing 'a' slightly?
        # No, keep 'a' fixed to the standard cycle to exploit the isomorphism invariance.
        # Wait, is it true that any synchronizing automaton with a single cycle 'a' is isomorphic to one with standard cycle 'a'?
        # Yes, if 'a' is a single cycle, we can relabel states so that a(i) = i+1 mod N.
        # So fixing 'a' is WLOG for the class of automata where 'a' is a single cycle.
        
        # What if 'a' is NOT a single cycle?
        # Then the automaton might have a different structure.
        # However, the known maximal thresholds for N=12 are likely in the single-cycle 'a' class.
        # (The Černý conjecture upper bound is (N-1)^2, and the best known constructions are close to this).
        
        # Let's also try 'a' being a product of two cycles?
        # Probably less efficient.
        
        # Let's add a small amount of randomization to 'a' just in case, 
        # by swapping two elements in the cycle? No, that breaks the cycle structure.
        # Let's stick to single cycle 'a'.

    if best_trans_a is None:
        return [base_a, base_b]
        
    return [best_trans_a, best_trans_b]

# EVOLVE-BLOCK-END
