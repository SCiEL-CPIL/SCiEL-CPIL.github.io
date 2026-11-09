import time
import random
import numpy as np

def construct():
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 150s limit
    
    # Precompute all pairs (a, b) with a <= b, a, b in 1..1000
    # For each sum s (2..2000), store list of pairs that produce s
    sum_to_pairs = {}
    for a in range(1, max_val + 1):
        for b in range(a, max_val + 1):
            s = a + b
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((a, b))
    
    # Build conflict graph:
    # Two vertices x, y conflict if including both x and y in S would cause some sum to have > 2 representations
    # More precisely: for each sum s, if we have pairs p1, p2, p3 that all produce s,
    # then any two of the "critical" elements conflict.
    # 
    # Actually, the B2[2] constraint is: for each sum s, at most 2 pairs (a,b) with a<=b, a,b in S, a+b=s.
    # This is a 3-uniform hypergraph constraint (forbidden sets of 3 pairs).
    # 
    # Simpler approach: Use simulated annealing directly on the set S.
    # State: which elements are in S (bit vector of length 1000)
    # Objective: maximize |S| subject to B2[2] constraint
    # Moves: add an element, remove an element, or swap
    
    # Use numpy for efficiency
    in_S = np.zeros(max_val + 1, dtype=bool)
    S_list = []
    counts = np.zeros(2 * max_val + 1, dtype=np.int32)
    
    def add_element(x):
        """Add x to S, update counts. Returns True if valid (no count exceeds 2)."""
        # Check if adding x would violate constraint
        if counts[2 * x] >= 2:
            return False
        for y in S_list:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # Safe to add
        in_S[x] = True
        S_list.append(x)
        counts[2 * x] += 1
        for y in S_list:
            if y != x:
                counts[x + y] += 1
        return True
    
    def remove_element(x):
        """Remove x from S, update counts."""
        in_S[x] = False
        S_list.remove(x)
        counts[2 * x] -= 1
        for y in S_list:
            if y != x:
                counts[x + y] -= 1
    
    def is_valid_add(x):
        """Check if adding x to current S would keep B2[2] valid."""
        if counts[2 * x] >= 2:
            return False
        for y in S_list:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        return True
    
    def is_valid_swap(a, c):
        """Check if swapping a (remove) and c (add) keeps B2[2] valid."""
        # Simulate: remove a, add c
        # Affected sums: 2a, a+y (y in S\{a}), 2c, c+y (y in S\{a})
        # We need to check that no sum exceeds 2 after the swap
        
        # Build delta for affected sums
        # First, what sums does a contribute to?
        a_contrib = set()
        a_contrib.add(2 * a)
        for y in S_list:
            if y != a:
                a_contrib.add(a + y)
        
        # What sums does c contribute to (if added to S\{a})?
        c_contrib = set()
        c_contrib.add(2 * c)
        for y in S_list:
            if y != a:
                c_contrib.add(c + y)
        
        # For each sum in the union, check new count
        all_affected = a_contrib | c_contrib
        for s in all_affected:
            # Current count of s
            cur = counts[s]
            # Subtract a's contribution
            if s == 2 * a:
                cur -= 1
            for y in S_list:
                if y != a and a + y == s:
                    cur -= 1
                    break  # Only one y for given s and a (since a+y=s => y=s-a)
            # Actually, for fixed a and s, y = s-a, so at most one y
            # So: a contributes to s iff s == 2a or (s - a in S and s - a != a)
            # Let me redo this more carefully
            
            # a contributes to sum s if:
            #   2a == s, OR
            #   a + y == s for some y in S, y != a, i.e., y = s - a, and s - a in S and s - a != a
            a_contrib_count = 0
            if 2 * a == s:
                a_contrib_count += 1
            y_candidate = s - a
            if 1 <= y_candidate <= max_val and y_candidate != a and in_S[y_candidate]:
                a_contrib_count += 1
            
            # c contributes to sum s (in S\{a}) if:
            #   2c == s, OR
            #   c + y == s for some y in S\{a}, i.e., y = s - c, and s - c in S\{a}
            c_contrib_count = 0
            if 2 * c == s:
                c_contrib_count += 1
            y_candidate_c = s - c
            if 1 <= y_candidate_c <= max_val and y_candidate_c != a and in_S[y_candidate_c]:
                c_contrib_count += 1
            
            new_count = counts[s] - a_contrib_count + c_contrib_count
            if new_count > 2:
                return False
            if new_count < 0:
                return False
        
        return True
    
    # Phase 1: Greedy initialization
    for i in range(1, max_val + 1):
        add_element(i)
    
    best_size = len(S_list)
    best_S = S_list.copy()
    
    # Phase 2: Simulated annealing
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.9995
    
    T = T_start
    step = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random unused element
            x = random.randint(1, max_val)
            if in_S[x]:
                continue
            if is_valid_add(x):
                add_element(x)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S = S_list.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        elif move_type < 0.5:
            # Try to remove a random element (to escape local optima)
            if S_list:
                a = random.choice(S_list)
                # Only remove if it doesn't make us worse (or with some probability)
                remove_element(a)
                no_improve_count += 1
        else:
            # Swap: remove a, add c
            if S_list:
                a = random.choice(S_list)
                c = random.randint(1, max_val)
                if in_S[c] or c == a:
                    continue
                if is_valid_swap(a, c):
                    remove_element(a)
                    add_element(c)
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S = S_list.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        
        step += 1
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # Reset if we've been stuck too long
        if no_improve_count > 5000:
            # Restart with best found so far
            # Clear current state
            while S_list:
                remove_element(S_list[-1])
            # Rebuild from best
            for x in best_S:
                add_element(x)
            no_improve_count = 0
            T = max(T, 0.5)
    
    # Return best found
    return best_S
