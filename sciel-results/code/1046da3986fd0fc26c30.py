import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency masks for each node.
    # adj[i] is a bitmask (list of 4 ints of 64 bits, or just a Python int) where bit j is 1 if j is adjacent to i.
    # Actually, since m=205, we can use a single Python integer as a bitmask.
    # adj[i] will have bit j set if (i-j) % m in forbidden or (j-i) % m in forbidden.
    # Since forbidden is symmetric (if d is a square, -d is a square? No. x^2 = d => (-x)^2 = d. So d is square implies -d is square? 
    # Wait. If d = x^2, then -d = (-x)^2 only if -1 is a square. 
    # Mod 205: 205 = 5 * 41. 
    # -1 is a square mod 5? 1^2=1, 2^2=4=-1. Yes.
    # -1 is a square mod 41? 41 = 1 mod 4. Yes.
    # So -1 is a square mod 205. Thus the set of squares is symmetric: d in F iff -d in F.
    # So the graph is undirected.
    
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            diff = (i - j) % m
            if diff in forbidden:
                mask |= (1 << j)
        adj[i] = mask
        
    def get_conflicts(state_mask, i):
        # Returns number of neighbors of i that are in state
        # state_mask is a Python int
        neighbors = adj[i]
        # Count bits in (state_mask & neighbors)
        # Python 3.8+ has int.bit_count()
        return (state_mask & neighbors).bit_count()
        
    def is_valid(state_mask):
        # Check if no edges within state
        # For each i in state, check if any neighbor is in state
        # This is O(N^2) worst case, but we can do it smarter or just trust SA to find valid ones.
        # For final verification, we can extract the list and check.
        return True # Placeholder, we check the best candidate at the end
        
    def extract_set(state_mask):
        s = []
        for i in range(m):
            if (state_mask >> i) & 1:
                s.append(i)
        return s
        
    def check_validity(lst):
        s_set = set(lst)
        for a in lst:
            for b in lst:
                if a == b: continue
                diff = (a - b) % m
                if diff in forbidden:
                    return False
        return True

    best_size = 0
    best_state = 0
    
    # Initialize with a greedy solution
    curr_state = 0
    for i in range(m):
        # Check if i can be added
        # Conflicts with current set
        if get_conflicts(curr_state, i) == 0:
            curr_state |= (1 << i)
            
    curr_size = curr_state.bit_count()
    if curr_size > best_size:
        best_size = curr_size
        best_state = curr_state
        
    # Simulated Annealing
    start_time = time.time()
    deadline = 590.0 # Safety margin
    
    # Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # We need to track size and conflicts for current state to update delta efficiently?
    # Or just recalculate? 
    # Recalculating conflicts for a single flip is O(1) using bitmasks.
    # But we need to know the change in conflicts.
    # Let curr_conflicts be the total number of edges in the induced subgraph.
    # When we add i: new_conflicts = curr_conflicts + count(neighbors of i in S).
    # When we remove i: new_conflicts = curr_conflicts - count(neighbors of i in S).
    
    # Calculate initial conflicts
    curr_conflicts = 0
    # Efficient way: sum over i in S of count(neighbors of i in S) / 2
    # Or just loop.
    s_list = extract_set(curr_state)
    for i in s_list:
        curr_conflicts += get_conflicts(curr_state, i)
    curr_conflicts //= 2
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random node to flip
        i = random.randint(0, m - 1)
        bit = 1 << i
        
        if curr_state & bit:
            # Remove i
            # Delta size = -1
            # Delta conflicts = - count(neighbors of i in S)
            # Note: get_conflicts includes i? No, adj[i] does not have bit i.
            # But the neighbors of i in S are the ones contributing to conflicts involving i.
            # When we remove i, we remove all conflicts connected to i.
            # count = get_conflicts(curr_state, i)
            c = get_conflicts(curr_state, i)
            delta_conflicts = -c
            delta_size = -1
        else:
            # Add i
            # Delta size = 1
            # Delta conflicts = + count(neighbors of i in S)
            c = get_conflicts(curr_state, i)
            delta_conflicts = c
            delta_size = 1
            
        # Energy = -Size + M * Conflicts
        M = 100
        delta_E = -delta_size + M * delta_conflicts
        
        # Acceptance
        if delta_E < 0:
            accept = True
        else:
            # Boltzmann factor
            # Avoid overflow
            if delta_E / T > 50:
                accept = False
            else:
                accept = random.random() < np.exp(-delta_E / T)
                
        if accept:
            # Update state
            if curr_state & bit:
                curr_state &= ~bit
            else:
                curr_state |= bit
            curr_size += delta_size
            curr_conflicts += delta_conflicts
            
            # Update best if valid
            if curr_conflicts == 0:
                if curr_size > best_size:
                    best_size = curr_size
                    best_state = curr_state
            
        T *= cooling
        
    # Final check and return
    best_set = extract_set(best_state)
    if not check_validity(best_set):
        # Fallback to greedy if SA failed to maintain validity (shouldn't happen if we only update best when valid)
        # But just in case
        greedy_set = []
        for i in range(m):
            ok = True
            for s in greedy_set:
                d = (i - s) % m
                if d in forbidden or (-d) % m in forbidden: # -d check redundant if symmetric
                    ok = False
                    break
            if ok:
                greedy_set.append(i)
        return np.array(greedy_set, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
