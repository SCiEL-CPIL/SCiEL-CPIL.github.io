import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # 1. Generate all 3-term APs (a, b, c) with a < b < c and c <= n
    #    These are our "items" to select.
    #    A 3-AP is defined by (start, diff).
    #    a = start, b = start + d, c = start + 2d
    #    Constraint: start + 2d <= n
    all_aps = []
    ap_to_index = {}
    
    for d in range(1, (n - 1) // 2 + 1):
        max_start = n - 2 * d
        for start in range(1, max_start + 1):
            a = start
            b = start + d
            c = start + 2 * d
            if c <= n:
                idx = len(all_aps)
                all_aps.append((a, b, c))
                ap_to_index[(a, b, c)] = idx
    
    m = len(all_aps)
    
    # 2. Build Conflict Graph
    #    Two APs conflict if:
    #    a) They share at least one element.
    #    b) Their union forms a 4-term AP.
    #       (e.g., {1,2,3} and {3,4,5} -> {1,2,3,4,5} contains 1,2,3,4? No, 1,2,3,4 is AP. 
    #        Wait, the condition is "no nonconstant four-term AP".
    #        If we select AP1={a,b,c} and AP2={x,y,z}, the set S = AP1 U AP2 must not contain any 4-term AP.
    #        The most dangerous case is when they merge into a 4-term AP directly.
        #       E.g., (1,2,3) and (3,4,5) -> {1,2,3,4,5}. Contains 1,2,3,4 (AP) and 2,3,4,5 (AP). So they conflict.
        #       Actually, if they share an element, we just need to check if the union contains a 4-AP.
        #       However, a simpler sufficient condition for conflict is sharing an element? 
        #       No, we CAN select multiple 3-APs that share an element, as long as no 4-AP is formed.
        #       Example: Select {1,2,3} and {1,3,5}. Union {1,2,3,5}. 
        #       4-APs in {1,2,3,5}: 
        #       1,2,3,4 (4 not in set)
        #       1,3,5,7 (7 not in set)
        #       2,3,4,5 (4 not in set)
        #       So {1,2,3} and {1,3,5} can coexist.
        #       So "sharing element" is NOT a conflict in itself.
        #       Conflict exists ONLY if the union contains a 4-term AP.
    
    #    Let's define conflicts properly.
    #    Conflict(i, j) if (AP_i U AP_j) contains a 4-term AP.
    
    #    This is expensive to check pairwise for ~20k APs.
    #    Optimization: 
    #    A 4-term AP is {x, x+d, x+2d, x+3d}.
    #    If we select a set of 3-APs, we must ensure no such 4-tuple is fully contained in the union.
    
    #    Alternative Approach: 
    #    Instead of MWIS on APs, let's stick to the element selection but use a smarter search?
    #    The prompt asks for a structural change. MWIS on APs is a strong structural change.
    #    
    #    Let's refine the conflict condition.
    #    Two 3-APs A and B conflict if A ∪ B contains a 4-AP.
    #    Since A and B are small, we can check this fast.
    #    Precompute for each AP the set of elements.
    #    For each pair (i, j), check if A_i ∪ A_j has a 4-AP.
    #    m is approx 256*128/2 ~ 16k. Pairs ~ 128M. Too slow in Python.
    
    #    Let's reconsider the problem size.
    #    n=256.
    #    Number of 3-APs: sum_{d=1}^{127} (256 - 2d) = 127*256 - 2*(127*128/2) = 32512 - 16256 = 16256.
    #    16k nodes.
    #    Building the full adjacency matrix is 16k^2 = 256M entries. 
    #    Checking conflict for each pair takes O(1) if we precompute 4-APs?
    #    
    #    Better: 
    #    Identify all 4-term APs in [1, 256].
    #    Let F be the set of all 4-term APs.
    #    We want to select a set of 3-APs S such that no 4-AP is covered by S.
    #    Wait, the condition is "no 4-term AP is a subset of the selected elements".
    #    Let E be the set of selected elements.
    #    Constraint: For all 4-APs P, P not subset of E.
    #    Objective: Maximize count of 3-APs in E.
    
    #    This is a different formulation:
    #    Variables: x_i in {0,1} for i in 1..256 (element selected?)
    #    Constraints: For each 4-AP {a,b,c,d}, x_a + x_b + x_c + x_d <= 3.
    #    Objective: Max sum_{3-APs {u,v,w}} (x_u x_v x_w) ... this is non-linear.
    
    #    The MWIS on 3-APs formulation:
    #    Nodes: 3-APs.
    #    Edge (i,j) if selecting both 3-APs creates a 4-AP in the union.
    #    Is that sufficient?
    #    If we select a set of 3-APs that is an independent set in this graph, does it guarantee no 4-AP?
    #    Yes, because any 4-AP in the union must be formed by elements from the selected 3-APs.
    #    If a 4-AP {a,b,c,d} is in the union, it must be covered by the selected 3-APs.
    #    Since 3-APs are small, any 4-AP must involve elements from at least two 3-APs (unless the 4-AP is contained in one 3-AP, which is impossible).
    #    Actually, a 4-AP {a,b,c,d} could be formed by:
    #    - Elements from two 3-APs.
    #    - Elements from three 3-APs.
    #    - Elements from four 3-APs.
    #    
    #    If the conflict graph only checks pairs, we might miss conflicts arising from triple interactions.
    #    Example: AP1={1,2,3}, AP2={3,4,5}, AP3={5,6,7}.
    #    Pair (1,2): {1,2,3,4,5} -> contains 1,2,3,4 (AP) and 2,3,4,5 (AP). Conflict!
    #    So pairwise conflicts catch most cases.
    #    What if AP1={1,3,5}, AP2={2,4,6}, AP3={3,5,7}?
    #    Union: {1,2,3,4,5,6,7}.
    #    Contains 1,2,3,4 (AP). 
    #    Does AP1 conflict with AP2? Union {1,2,3,4,5,6}. Contains 1,2,3,4. Conflict.
    #    So pairwise seems strong.
    
    #    However, implementing MWIS for 16k nodes is hard in Python within 90s.
    #    
    #    Let's try a different structural change: **Greedy Set Cover / Max Coverage with Conflict Resolution**
    #    Or: **Simulated Annealing on the Element Set** (which is what the seed does poorly?).
    #    The seed is a simple greedy: iterate 1..256, add if no 4-AP formed.
    #    This is very myopic.
    
    #    Let's try **Local Search on Elements** with a better move set.
    #    Start with the greedy solution.
    #    Moves: Add an element, Remove an element, Swap an element.
    #    Objective: Maximize #3-APs.
    #    Constraint: No 4-APs.
    #    
    #    This is a standard approach but "materially different" from the seed's one-pass greedy?
    #    The seed is NOT a local search. It's a constructive heuristic.
    #    So implementing a proper Simulated Annealing or Hill Climbing is a structural change.
    
    #    Let's implement **Simulated Annealing**.
    
    #    1. Initial State: The seed's greedy solution.
    #    2. State: Set of selected elements.
    #    3. Objective: Count of 3-APs.
    #    4. Constraint: No 4-APs.
    #    5. Moves: 
    #       - Add a random unselected element if it doesn't create a 4-AP.
    #       - Remove a random selected element.
    #       - Swap a selected element with an unselected one if valid.
    
    #    Efficiency:
    #    We need fast delta evaluation for adding/removing elements.
    #    
    #    Let's use bitmasks for the elements? 256 bits fits in 4 uint64s or just Python integers.
    #    Python integers are arbitrary precision, so we can use a single int as a bitmask.
    
    #    Precomputation:
    #    For each element x, precompute:
    #    - 3-APs containing x.
    #    - 4-APs containing x.
    
    #    When adding x to S:
    #    - Check if any 4-AP containing x is now fully in S.
    #    - Count new 3-APs formed.
    
    #    When removing x from S:
    #    - Count 3-APs lost.
    #    - (4-AP constraint is automatically satisfied by removing).
    
    #    Let's implement this.
    
    #    Precompute 3-APs and 4-APs.
    #    3-APs: list of (a,b,c)
    #    4-APs: list of (a,b,c,d)
    
    #    For fast lookup, we can use sets of tuples or bitmasks.
    #    Since n=256, we can map element i to bit i.
    
    #    Let's build:
    #    ap3_to_elements: list of bitmasks for each 3-AP.
    #    ap4_to_elements: list of bitmasks for each 4-AP.
    
    #    Also, for each element x, list of indices of 3-APs containing x.
    #    And list of indices of 4-APs containing x.
    
    #    Current state: bitmask S.
    #    Score: count of 3-APs fully in S.
    
    #    We can maintain the score incrementally.
    
    #    Let's code the SA.
    
    #    Data Structures:
    #    elems_3ap = [[] for _ in range(n+1)] # indices of 3-APs containing element i
    #    elems_4ap = [[] for _ in range(n+1)] # indices of 4-APs containing element i
    #    ap3_masks = [] # bitmask for each 3-AP
    #    ap4_masks = [] # bitmask for each 4-AP
    
    #    Initialization:
    #    Generate all 3-APs and 4-APs.
    
    #    3-APs:
    for d in range(1, (n - 1) // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            ap3_masks.append(mask)
            idx = len(ap3_masks) - 1
            elems_3ap[a].append(idx)
            elems_3ap[b].append(idx)
            elems_3ap[c].append(idx)
            
    #    4-APs:
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_elem = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d_elem-1))
            ap4_masks.append(mask)
            idx = len(ap4_masks) - 1
            elems_4ap[a].append(idx)
            elems_4ap[b].append(idx)
            elems_4ap[c].append(idx)
            elems_4ap[d_elem].append(idx)
            
    #    Initial Solution: Greedy
    #    Iterate 1..n, add if no 4-AP formed.
    #    We need to check if adding x creates a 4-AP.
    #    A 4-AP is formed if there exists a 4-AP containing x such that the other 3 elements are already in S.
    
    current_mask = 0
    current_score = 0
    #    We need to track which 3-APs are currently "complete" in S.
    #    Instead of tracking a count, we can just recalculate or track a set of active 3-AP indices.
    #    For speed, let's track `active_3aps` as a set of indices.
    #    Or better, just keep the mask and calculate score when needed? No, SA needs fast eval.
    
    #    Let's maintain `active_3aps` as a boolean array or set.
    #    Since m_3 is ~16k, a set is fine.
    
    active_3aps = set()
    
    #    Greedy Construction
    for x in range(1, n + 1):
        #    Check if adding x creates a 4-AP
        valid = True
        for ap4_idx in elems_4ap[x]:
            #    Check if the other 3 elements are in current_mask
            #    mask_ap4 = ap4_masks[ap4_idx]
            #    other_mask = mask_ap4 & ~(1 << (x-1))
            #    if other_mask & current_mask == other_mask:
            #        valid = False
            #        break
            #    Optimize: precompute for each 4-AP and each element in it, the mask of the other 3.
            pass # We will implement this check properly below
            
    #    Let's precompute `ap4_other_masks[ap4_idx][elem]` -> mask of other 3 elements.
    #    But we don't have that structure yet. Let's build it.
    
    #    Actually, let's just use the bitmask check directly. It's fast enough for n=256.
    
    def can_add(x, mask):
        bit = 1 << (x - 1)
        for ap4_idx in elems_4ap[x]:
            ap4_mask = ap4_masks[ap4_idx]
            #    We need to check if (ap4_mask & ~bit) is a subset of mask
            other_mask = ap4_mask & ~bit
            if (other_mask & mask) == other_mask:
                return False
        return True

    def add_element(x, mask, active_set):
        bit = 1 << (x - 1)
        new_mask = mask | bit
        #    Update active 3-APs
        for ap3_idx in elems_3ap[x]:
            ap3_mask = ap3_masks[ap3_idx]
            if (ap3_mask & new_mask) == ap3_mask:
                active_set.add(ap3_idx)
        return new_mask, active_set

    def remove_element(x, mask, active_set):
        bit = 1 << (x - 1)
        new_mask = mask & ~bit
        #    Update active 3-APs
        for ap3_idx in elems_3ap[x]:
            ap3_mask = ap3_masks[ap3_idx]
            if (ap3_mask & new_mask) != ap3_mask:
                active_set.discard(ap3_idx)
        return new_mask, active_set

    #    Re-run Greedy with these functions
    current_mask = 0
    active_3aps = set()
    for x in range(1, n + 1):
        if can_add(x, current_mask):
            current_mask, active_3aps = add_element(x, current_mask, active_3aps)
            
    best_mask = current_mask
    best_score = len(active_3aps)
    best_active = set(active_3aps)
    
    #    Simulated Annealing
    #    T = 100.0
    #    Cooling = 0.99
    #    Steps = 100000
    
    #    We need to be careful with time.
    #    Let's run until deadline.
    
    import random
    random.seed(42)
    
    T = 50.0
    cooling = 0.999
    current_score = len(active_3aps)
    
    while time.time() < deadline:
        #    Generate a move
        #    1. Add a random unselected element
        #    2. Remove a random selected element
        #    3. Swap
        
        r = random.random()
        
        if r < 0.4:
            #    Add
            #    Pick random x not in mask
            #    We can pick from 1..n and check if in mask
            #    Or maintain a list of unselected? 
            #    For n=256, picking random 1..n is fine.
            x = random.randint(1, n)
            if (current_mask >> (x-1)) & 1:
                continue # Already selected, skip
            if not can_add(x, current_mask):
                continue # Invalid, skip
            
            new_mask, new_active = add_element(x, current_mask, active_3aps)
            new_score = len(new_active)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                active_3aps = new_active
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_active = set(active_3aps)
                    
        elif r < 0.6:
            #    Remove
            #    Pick random x in mask
            #    We need to pick a random set bit.
            #    Convert mask to list? Slow.
            #    Just pick random 1..n and check if in mask.
            x = random.randint(1, n)
            if not ((current_mask >> (x-1)) & 1):
                continue # Not selected, skip
            
            new_mask, new_active = remove_element(x, current_mask, active_3aps)
            new_score = len(new_active)
            delta = new_score - current_score
            
            #    Removing always decreases score or keeps it same? 
            #    Removing an element can only remove 3-APs, not add them.
            #    So delta <= 0.
            #    We accept if random < exp(delta/T).
            if random.random() < np.exp(delta / T):
                current_mask = new_mask
                active_3aps = new_active
                current_score = new_score
                
        else:
            #    Swap
            #    Pick x in mask, y not in mask.
            x = random.randint(1, n)
            y = random.randint(1, n)
            if x == y:
                continue
            if ((current_mask >> (x-1)) & 1) == 0:
                continue # x not in mask
            if ((current_mask >> (y-1)) & 1) == 1:
                continue # y in mask
                
            #    Check if adding y is valid (ignoring x for a moment? No, we remove x first)
            #    Remove x, then add y.
            #    But we need to check if the intermediate state or final state has 4-AP.
            #    Since we remove x first, the 4-AP constraint is relaxed.
            #    Then we add y. We must check if adding y to (mask - x) creates a 4-AP.
            
            temp_mask, temp_active = remove_element(x, current_mask, active_3aps)
            if not can_add(y, temp_mask):
                continue # Invalid swap
            
            new_mask, new_active = add_element(y, temp_mask, temp_active)
            new_score = len(new_active)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                active_3aps = new_active
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_active = set(active_3aps)
        
        T *= cooling
        if T < 0.1:
            T = 0.1 # Floor temperature to keep exploring slightly
    
    #    Extract result from best_mask
    result = []
    for i in range(1, n + 1):
        if (best_mask >> (i-1)) & 1:
            result.append(i)
            
    return sorted(result)

#    Wait, I defined elems_3ap and elems_4ap inside construct but used them in nested functions.
#    I need to move the precomputation outside or ensure they are in scope.
#    Also, I used np.exp, need to import numpy.
#    The code above has a logical error: elems_3ap and elems_4ap are defined in the first part of the function, 
#    but the nested functions can access them via closure. 
#    However, I redefined the loops for 3-APs and 4-APs *after* defining the functions? 
#    No, I defined the loops *before* the functions in the draft above? 
#    Let's check the order in the draft.
#    The draft has:
#    1. Generate APs (loops)
#    2. Define functions
#    3. Greedy
#    4. SA
#    
#    But in the draft code block I wrote:
#    `for d in range...` (3-APs)
#    `for d in range...` (4-APs)
#    `def can_add...`
#    This is correct.
#    
#    One issue: `elems_3ap` and `elems_4ap` are lists of lists. 
#    I need to initialize them as `[[] for _ in range(n+1)]` before the loops.
#    I did that in the thought process but need to ensure it's in the code.
    
#    Let's assemble the final code carefully.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    #    Precomputation
    elems_3ap = [[] for _ in range(n + 1)]
    elems_4ap = [[] for _ in range(n + 1)]
    ap3_masks = []
    ap4_masks = []
    
    #    3-APs
    for d in range(1, (n - 1) // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            ap3_masks.append(mask)
            idx = len(ap3_masks) - 1
            elems_3ap[a].append(idx)
            elems_3ap[b].append(idx)
            elems_3ap[c].append(idx)
            
    #    4-APs
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_elem = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d_elem-1))
            ap4_masks.append(mask)
            idx = len(ap4_masks) - 1
            elems_4ap[a].append(idx)
            elems_4ap[b].append(idx)
            elems_4ap[c].append(idx)
            elems_4ap[d_elem].append(idx)
            
    #    Helper Functions
    def can_add(x, mask):
        bit = 1 << (x - 1)
        for ap4_idx in elems_4ap[x]:
            ap4_mask = ap4_masks[ap4_idx]
            other_mask = ap4_mask & ~bit
            if (other_mask & mask) == other_mask:
                return False
        return True

    def add_element(x, mask, active_set):
        bit = 1 << (x - 1)
        new_mask = mask | bit
        for ap3_idx in elems_3ap[x]:
            ap3_mask = ap3_masks[ap3_idx]
            if (ap3_mask & new_mask) == ap3_mask:
                active_set.add(ap3_idx)
        return new_mask, active_set

    def remove_element(x, mask, active_set):
        bit = 1 << (x - 1)
        new_mask = mask & ~bit
        for ap3_idx in elems_3ap[x]:
            ap3_mask = ap3_masks[ap3_idx]
            if (ap3_mask & new_mask) != ap3_mask:
                active_set.discard(ap3_idx)
        return new_mask, active_set

    #    Initial Solution: Greedy
    current_mask = 0
    active_3aps = set()
    for x in range(1, n + 1):
        if can_add(x, current_mask):
            current_mask, active_3aps = add_element(x, current_mask, active_3aps)
            
    best_mask = current_mask
    best_score = len(active_3aps)
    best_active = set(active_3aps)
    
    #    Simulated Annealing
    import random
    random.seed(42)
    
    T = 50.0
    cooling = 0.999
    current_score = len(active_3aps)
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.4:
            x = random.randint(1, n)
            if (current_mask >> (x-1)) & 1:
                continue
            if not can_add(x, current_mask):
                continue
            
            new_mask, new_active = add_element(x, current_mask, active_3aps)
            new_score = len(new_active)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                active_3aps = new_active
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_active = set(active_3aps)
                    
        elif r < 0.6:
            x = random.randint(1, n)
            if not ((current_mask >> (x-1)) & 1):
                continue
            
            new_mask, new_active = remove_element(x, current_mask, active_3aps)
            new_score = len(new_active)
            delta = new_score - current_score
            
            if random.random() < np.exp(delta / T):
                current_mask = new_mask
                active_3aps = new_active
                current_score = new_score
                
        else:
            x = random.randint(1, n)
            y = random.randint(1, n)
            if x == y:
                continue
            if ((current_mask >> (x-1)) & 1) == 0:
                continue
            if ((current_mask >> (y-1)) & 1) == 1:
                continue
                
            temp_mask, temp_active = remove_element(x, current_mask, active_3aps)
            if not can_add(y, temp_mask):
                continue
            
            new_mask, new_active = add_element(y, temp_mask, temp_active)
            new_score = len(new_active)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                active_3aps = new_active
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_active = set(active_3aps)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    result = []
    for i in range(1, n + 1):
        if (best_mask >> (i-1)) & 1:
            result.append(i)
            
    return sorted(result)

