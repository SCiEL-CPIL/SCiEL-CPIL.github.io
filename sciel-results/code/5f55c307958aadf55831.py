import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    # 4-AP: (a, a+d, a+2d, a+3d)
    
    aps3 = []
    aps4 = []
    
    # Map element to list of AP indices
    # Using 0-indexed for internal logic, 1-indexed for output
    # Elements are 1..256
    
    elem_to_aps3 = {i: [] for i in range(1, n+1)}
    elem_to_aps4 = {i: [] for i in range(1, n+1)}
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                aps3.append((a, b, c))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1
                
    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps4.append((a, b, c, e))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1
                
    total_aps3 = len(aps3)
    total_aps4 = len(aps4)
    
    # Helper to count 3-APs in a set A
    def count_3aps(A):
        count = 0
        for a, b, c in aps3:
            if a in A and b in A and c in A:
                count += 1
        return count
        
    # Helper to check if adding x to A creates a 4-AP
    # Returns True if A U {x} is 4-AP free
    def is_valid_add(A, x):
        # Check if any 4-AP containing x has its other 3 elements in A
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d = aps4[ap_idx]
            # Find the other 3
            others = [y for y in (a, b, c, d) if y != x]
            if all(y in A for y in others):
                return False
        return True
        
    # Helper to count 4-APs in A
    def count_4aps(A):
        count = 0
        for a, b, c, d in aps4:
            if a in A and b in A and c in A and d in A:
                count += 1
        return count

    # Initial Greedy Construction
    # Sort by some heuristic? Random order might be better for diversity, 
    # but Greedy by value is deterministic. Let's use random order for initial diversity,
    # then refine.
    
    # Actually, let's start with the standard greedy to get a good baseline,
    # then use SA to improve.
    
    def run_greedy():
        A = set()
        # Try adding in increasing order? Or random?
        # Increasing order tends to produce structured sets.
        for x in range(1, n + 1):
            if is_valid_add(A, x):
                A.add(x)
        return A

    # Let's also try a random greedy
    def run_random_greedy(seed):
        rng = random.Random(seed)
        A = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if is_valid_add(A, x):
                A.add(x)
        return A

    # Start with a few random greedy initializations
    best_set = set()
    best_score = 0
    
    # Run a few initializations
    for i in range(10):
        if i == 0:
            A = run_greedy()
        else:
            A = run_random_greedy(i)
        
        # Ensure maximality: try adding any remaining valid element
        # (The greedy loops above do this, but just in case)
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in A and is_valid_add(A, x):
                    A.add(x)
                    changed = True
        
        score = count_3aps(A)
        if score > best_score:
            best_score = score
            best_set = set(A)
            
    # Simulated Annealing Phase
    # We will run SA starting from best_set
    A = set(best_set)
    current_score = best_score
    
    # Precompute for fast delta calculation?
    # Given n=256, count_3aps takes ~16k checks. 
    # 10k iterations * 16k = 160M ops. Might be too slow in Python.
    # We need faster delta calculation.
    
    # Let's optimize the scoring update.
    # Maintain a counter for 3-APs?
    # It's hard to maintain incrementally with swaps.
    # But we can calculate delta.
    
    def get_delta_3aps(A, u, v):
        # u in A, v not in A
        # Delta = - (# 3-APs with u and other 2 in A) + (# 3-APs with v and other 2 in A\{u})
        
        # Count 3-APs lost by removing u
        lost = 0
        for ap_idx in elem_to_aps3[u]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != u]
            if all(y in A for y in others):
                lost += 1
                
        # Count 3-APs gained by adding v
        # Note: A\{u} is the set of other elements
        gained = 0
        # We need to check if the other 2 are in A and not u.
        # Since v != u, and u is being removed, we just check if others are in A.
        # But if one of the others is u, it's not in A\{u}.
        # However, u and v are distinct.
        # If a 3-AP contains v and u, then removing u breaks it anyway.
        # So we just check if the other 2 elements are in A.
        # Wait, if the 3-AP is (u, v, w), removing u breaks it. Adding v doesn't create it because u is gone.
        # So the condition "others in A" is sufficient because if one other is u, it's not in A\{u}, 
        # but my check "all(y in A for y in others)" would fail if y==u?
        # No, u IS in A currently.
        # So if the AP is (u, v, w), others are (u, w).
        # u is in A. w is in A. So it counts as "gained".
        # But actually, the AP (u, v, w) is NOT formed in A\{u} U {v} because u is gone.
        # So we must exclude APs that contain u.
        
        # Correct logic for Gained:
        # APs containing v such that the other 2 elements are in A AND NOT u.
        
        for ap_idx in elem_to_aps3[v]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != v]
            # Check if both others are in A
            # AND neither is u (because u is removed)
            # Actually, if one is u, it's not in A\{u}.
            # So we just need to check if they are in A\{u}.
            # Which is equivalent to: y in A and y != u.
            
            valid = True
            for y in others:
                if y not in A or y == u:
                    valid = False
                    break
            if valid:
                gained += 1
                
        return gained - lost

    def get_delta_4aps(A, u, v):
        # Similar logic
        lost = 0
        for ap_idx in elem_to_aps4[u]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != u]
            if all(y in A for y in others):
                lost += 1
                
        gained = 0
        for ap_idx in elem_to_aps4[v]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != v]
            valid = True
            for y in others:
                if y not in A or y == u:
                    valid = False
                    break
            if valid:
                gained += 1
                
        return gained - lost

    # SA Loop
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    steps = 0
    
    # We need to track current 4-AP count to ensure it stays 0
    current_4aps = count_4aps(A) # Should be 0
    
    while time.time() < deadline and T > min_T:
        # Choose move
        if random.random() < 0.5:
            # Try Insert
            # Pick random v not in A
            # To make it fast, maintain a list of available elements?
            # Or just pick random x in 1..n and check if in A.
            # n=256 is small.
            v = random.randint(1, n)
            if v in A:
                continue
            # Check validity
            if is_valid_add(A, v):
                # Calculate delta 3
                # Lost: 0
                # Gained: # 3-APs with v and other 2 in A
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    a, b, c = aps3[ap_idx]
                    others = [y for y in (a, b, c) if y != v]
                    if all(y in A for y in others):
                        gained += 1
                delta_3 = gained
                delta_4 = 0 # We checked validity, so no new 4-APs
                
                # Always accept if valid and score increases?
                # Yes, for Insert, score always increases (or stays same if gained=0, but gained>0 usually).
                # If gained == 0, it's neutral. Accept to grow set?
                # Growing set might allow better swaps later.
                A.add(v)
                current_score += delta_3
                # Update best if better
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(A)
            else:
                # Invalid, skip
                pass
        else:
            # Try Swap
            # Pick random u in A, v not in A
            if not A:
                continue
            u = random.choice(list(A))
            v = random.randint(1, n)
            if v in A or v == u:
                continue
            
            delta_4 = get_delta_4aps(A, u, v)
            # If current_4aps + delta_4 > 0, invalid
            if current_4aps + delta_4 > 0:
                continue
                
            delta_3 = get_delta_3aps(A, u, v)
            
            if delta_3 > 0:
                # Accept
                A.remove(u)
                A.add(v)
                current_score += delta_3
                current_4aps += delta_4
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(A)
            else:
                # Accept with prob exp(delta_3 / T)
                if random.random() < math.exp(delta_3 / T):
                    A.remove(u)
                    A.add(v)
                    current_score += delta_3
                    current_4aps += delta_4
                    # Update best? No, score decreased.
                    # But maybe leads to better state.
                    
        T *= cooling
        steps += 1
        if steps % 10000 == 0 and time.time() > deadline - 5:
            break
            
    # Final check
    # Ensure best_set is valid
    if count_4aps(best_set) != 0:
        # Fallback to greedy if SA broke it (shouldn't happen)
        best_set = run_greedy()
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
